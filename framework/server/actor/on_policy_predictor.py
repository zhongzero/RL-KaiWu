#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import traceback
import yaml
import threading
import multiprocessing
from multiprocessing import Value
from framework.common.utils.rainbow_utils import RainbowUtils
from framework.common.checkpoint.model_file_save import ModelFileSave
import lz4.block
from framework.common.utils.tf_utils import *
import numpy as np
import os
import schedule
import datetime
from framework.server.actor.predictor import Predictor
from framework.common.utils.common_func import TimeIt, set_schedule_event, make_single_dir, actor_learner_aisrv_count, get_host_ip, stop_process_by_pid, decompress_data, compress_data
from framework.common.config.algo_conf import AlgoConf
from framework.common.config.config_control import CONFIG
from framework.common.algorithms.model_wrapper_builder import ModelWrapperBuilder
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from framework.common.pybind11.zmq_ops.zmq_ops import ZMQPullSocket

from framework.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from framework.common.checkpoint.model_file_sync import ModelFileSync
from framework.common.alloc.alloc_proxy import AllocProxy
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.ipc.zmq_util import ZmqServer
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label

class OnPolicyPredictor(Predictor, multiprocessing.Process):

    def __init__(self, send_server, recv_server, name):
        super().__init__(send_server, recv_server, name)

        self.names, self.dtypes, self.shapes = self.tensor_spec()

        # 进程启动的序号
        self.index = -1

        '''
        actor采用批处理从zmq_server获取, 故记录了此时队列长度, 从队列里获取的耗时, 
        为了减少损耗, 只是记录统计周期最后一次的值
        1. actor从zmq-server获取的队列长度, 最大为配置值, 需要查看平时是多少
        2. 从zmq-server的队列里获取数据时批处理耗时
        3. actor批处理预测耗时
        4. actor将预测结果发送给aisrv的批处理耗时
        5. actor加载最新的Model文件耗时
        '''
        self.actor_from_zmq_queue_size = 0
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_batch_predict_cost_time_ms = 0
        self.actor_load_last_model_cost_ms = 0
        self.actor_load_last_model_succ_cnt = 0

        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.max_compress_time = 0

        '''
        从actor_server获取的需要预测的数据, 每次处理完成需要清空
        因为存在每次按照batch_size或者按照超时时间来读取, 那这里采用单独的线程来读取数据, 规避超时时间的限制

        '''
        if CONFIG.pipeline_process_sync:
            self.predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)
        
        if CONFIG.actor_server_predict_server_different_queue:
            self.predict_request_queue_from_actor_server = None
    
    '''
    返回predict_request_queue
    '''
    def get_predict_request_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_request_queue
        
        return None
    
    '''
    返回predict_result_queue
    '''
    def get_predict_result_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_result_queue
        
        return None

    def tensor_spec(self):
        names, dtypes, shapes = [], [], []
        for name, array_spec in self.policy_conf.state.state_space().items():
            names.append(name)
            dtypes.append(tf.as_dtype(array_spec.dtype))
            shapes.append(tf.TensorShape((None,) + array_spec.shape))

        names.extend([KaiwuDRLDefine.CLIENT_ID_TENSOR,
                     KaiwuDRLDefine.COMPOSE_ID_TENSOR])
        dtypes.extend([tf.int32, tf.int32])

        # 注意这里COMPOSE_ID_TENSOR的修改需要同步修改这里
        shapes.extend([tf.TensorShape((None,)), tf.TensorShape((None, 3))])
        return names, dtypes, shapes

    def create_mode_wraper(self):

        with TimeIt() as ti:
            # network
            network = self.policy_conf.actor_network(
                self.policy_conf.state.state_space(),
                self.policy_conf.action.action_space()
            )

            # model
            name = "%s_%s" % (CONFIG.app, CONFIG.algo)
            model = AlgoConf[CONFIG.algo].actor_model(network, name)

            self.model_wrapper = ModelWrapperBuilder().create_model_wrapper(model, self.logger)

            if KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.build_predict_graph(self.input_tensors)
                self.model_wrapper.add_predict_hooks(self.predict_hooks())
                self.model_wrapper.create_predict_session()

                self.global_step = self.model_wrapper.get_global_step()

            elif KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.model.build_model()

            elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
                pass

            elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
                pass

            elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.model.build_model()
                self.model_wrapper.model.init_model()

            else:
                self.logger.error(f'predict error use_which_deep_learning_framework {CONFIG.use_which_deep_learning_framework}, only suport {KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, \
                    {KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, {KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}')

                return

            self.logger.info(
                f'predict start, model_wrapper is {self.model_wrapper.name}')

    '''
    input_tensors作为TensorFlow 预测和训练的入口
    '''

    def input_tensors(self):
        # 等待输入
        def wait_for_inputs(queue_size):
            with tf.control_dependencies(enqueue_ops):
                tf.no_op()

            self.logger.debug(
                f'predict current predict_input_queue size is {input_queue.size()}')
            return input_queue.size()

        # actor上zmq的端口需要和aisrv上的zmq端口一致
        receiver = ZMQPullSocket(
            f'tcp://{CONFIG.ip_address}:{CONFIG.zmq_server_op_port}', self.dtypes, hwm=CONFIG.zmq_ops_hwm)
        self.logger.info(
            f'predict zmq-ops server start at {CONFIG.ip_address}:{CONFIG.zmq_server_op_port}')

        enqueue_tensors = [tensor if not CONFIG.use_rnn else tf.expand_dims(tensor, axis=1)
                           for tensor in receiver.pull()]
        input_shapes = [(shape[1:] if not CONFIG.use_rnn else
                         tf.TensorShape([1] + shape[1:].as_list())) for shape in self.shapes]

        # 利用TensorFlow的FIFOQueue
        input_queue = tf.queue.FIFOQueue(
            CONFIG.predict_input_queue_size,
            self.dtypes,
            input_shapes,
            name='predict_input_queue'
        )

        enqueue_op = input_queue.enqueue_many(enqueue_tensors)
        enqueue_ops = [enqueue_op] * CONFIG.predict_input_threads

        # 创建了CONFIG.predict_input_threads线程, 每个线程里运行的是enqueue_op操作
        tf.compat.v1.train.add_queue_runner(
            tf.compat.v1.train.QueueRunner(
                input_queue,
                enqueue_ops=enqueue_ops
            )
        )

        qsize_tensor = input_queue.size()

        # TensorFlow while loop处理
        self.dequeue_size_tensor = tf.while_loop(
            lambda queue_size: tf.less(
                queue_size, CONFIG.predict_input_queue_deq_min),
            wait_for_inputs,
            [qsize_tensor]
        )

        self.dequeue_tensors = input_queue.dequeue_many(
            self.dequeue_size_tensor)

        return dict(zip(self.names, self.dequeue_tensors))

    def predict_hooks(self):
        return []

    '''
    actor周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    '''

    def rainbow_activate(self):

        self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
        self.rainbow_activate_single_process(CONFIG.svr_name)

    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(
            process_name)
        if result_code:
            self.logger.error(
                f'predict read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(
                f'predict read_from_rainbow failed, data is None or data len is 0')
            return

        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(process_name, to_change_key_values)

    '''
    actor加载从learn上同步最新的model文件
    1. 对于tensortrt的, 在/data/ckpt/sgame_5v5_ppo/convert_models_actor下加载
    2. 其他, 在/data/ckpt/sgame_5v5_ppo/models下加载
    '''

    def load_last_new_model(self):
        if KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            if CONFIG.self_play_actor:
                models_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/convert_models_{CONFIG.svr_name}/trt_weights.wts2_old'
            else:
                models_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/convert_models_{CONFIG.svr_name}/trt_weights.wts2'

            # 判断文件不存在提前返回
            if not os.path.exists(models_path):
                return

        else:
            '''
            train模式, 加载最新model文件的地址
            eval模式, 加载指定的eval_model_dir
            '''
            # 加载最新model文件的地址
            models_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/models'
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                models_path = CONFIG.eval_model_dir

        try:
            # 调用业务加载最新模型, 可能会出现错误
            with TimeIt() as ti:
                self.model_wrapper.load_last_new_model(models_path)

            if self.actor_load_last_model_cost_ms < ti.interval * 1000:
                self.actor_load_last_model_cost_ms = ti.interval * 1000

            '''
            训练模式, 修改为DEBUG模式, 减少日志打印
            评估模式, 关键信息打印INFO日志
            '''
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                self.logger.info(
                    f'predict load_last_new_model from {models_path} success')
            else:
                self.logger.debug(
                    f'predict load_last_new_model from {models_path} success')
                
            self.actor_load_last_model_succ_cnt += 1

        except Exception as e:
            self.logger.error(
                f'predict load_last_new_model from {models_path} failed, error is {str(e)}')
            
            # 如果是eval模式, 加载失败就停止actor预测进程, 其他模式会周期性的加载model文件, 不做报错退出
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                self.logger.info(f'predict run mode is KaiwuDRLDefine.RUN_MODEL_EVAL load_last_new_model from {models_path} failed, so exit')
                self.process_pid_list.append(os.getpid())
                stop_process_by_pid(self.process_pid_list)

    def predict_stat_reset(self):
        self.actor_batch_predict_cost_time_ms = 0
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0
        self.actor_load_last_model_cost_ms = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.max_compress_time = 0

    '''
    这里增加predict的统计项
    '''

    def predict_stat(self):

        if int(CONFIG.use_prometheus):

            predict_request_queue_size = 0
            predict_result_queue_size = 0
            try:
                predict_request_queue_size = self.predict_request_queue.qsize()
                predict_result_queue_size = self.predict_result_queue.qsize()
            except Exception as e:
                pass

            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: self.model_wrapper.predict_stat,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE: self.actor_from_zmq_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS: self.actor_from_zmq_queue_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS: self.actor_batch_predict_cost_time_ms,
                KaiwuDRLDefine.ACTOR_TCP_AISRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS: self.actor_load_last_model_cost_ms,
                KaiwuDRLDefine.ACTORLOAD_LAST_MODEL_SUCC_CNT: self.actor_load_last_model_succ_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME : self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_REQUEST_QUEUE_SIZE : predict_request_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_RESULT_QUEUE_SIZE : predict_result_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME : self.max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE : self.max_compress_size,

            }

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data[KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_ERROR_CNT] = self.on_policy_pull_from_modelpool_error_cnt
                monitor_data[KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_SUCCESS_CNT] = self.on_policy_pull_from_modelpool_success_cnt
                monitor_data[KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_ERROR_COUNT] = self.actor_change_model_version_error_count
                monitor_data[KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_SUCCESS_COUNT] = self.actor_change_model_version_success_count
                
            self.monitor_proxy.put_data(monitor_data)

        # 指标复原, 计算的是周期性的上报指标
        self.predict_stat_reset()

        # self.logger.debug(f'predict now predict count is {self.model_wrapper.predict_stat}')

    '''
    根据不同的启动方式进行处理:
    1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
    2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
    '''

    def start_actor_process_by_type(self):

        # 按照需要引入ModelFileSave
        self.model_file_saver = ModelFileSave()
        self.model_file_saver.start_actor_process_by_type(self.logger)

    '''
    框架运行前创建必要的文件目录
    '''

    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)
        make_single_dir(f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}')

    def before_run(self):

        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/actor_predict_pid{self.current_pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",  CONFIG.svr_name)
        self.logger.info(f'predict process start at pid is {self.current_pid}')

        self.make_dirs()

        # 支持间隔N分钟, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, 
                                    CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, self.logger)
            self.logger.info(f'predict RainbowUtils {self.rainbow_utils.identity}')
            
            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 根据不同启动方式来进行处理
        self.start_actor_process_by_type()
    
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            self.process_pid_list = []

        # model_wrapper
        with TimeIt() as ti:
            self.create_mode_wraper()

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            if CONFIG.actor_server_async:
                self.process_pid_list.append(self.send_server.pid)
                self.process_pid_list.append(self.recv_server.pid)
            else:
                self.process_pid_list.append(self.send_server.pid)

        '''
        如果actor采用tensorrt, 则需要启动下面进程来进行并行化处理:
        1. CPU到GPU拷贝进程
        2. GPU到CPU拷贝进程
        '''

        '''
        if  KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            self.actor_tensorrt_cpu2gpu = ActorTensorRTCPU2GPU(self.server)
            self.actor_tensorrt_cpu2gpu.start()

            self.actor_tensort_gpu2cpu = ActorTensorRTGPU2CPU(self.server)
            self.actor_tensort_gpu2cpu.start()
        
        '''

        '''
        model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程
        如果是on-plocy算法则需要保存下来learner同步过来最新的model文件ID, 如果是off-policy则不需要
        为了编程方便, 都统一设置下
        '''
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            if self.index == 0:
                self.model_file_sync_wraper = ModelFileSyncWrapper()
                self.model_file_sync_wraper.init()

        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.model_file_sync_wraper = ModelFileSync()
            self.model_file_sync_wraper.make_model_dirs(self.logger)

            self.current_sync_model_version_from_learner = -1
            self.zmq_server = ZmqServer(CONFIG.ip_address, int(CONFIG.zmq_server_port) + 100)
            self.zmq_server.bind()
            self.logger.info(f'predict zmq server on-policy bind at {CONFIG.ip_address} : {int(CONFIG.zmq_server_port) + 100} for learner')

            # 下面是统计告警指标
            self.on_policy_pull_from_modelpool_error_cnt = 0
            self.on_policy_pull_from_modelpool_success_cnt = 0
            self.actor_change_model_version_error_count = 0
            self.actor_change_model_version_success_count = 0

        else:
            pass

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc) and self.index == 0:
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                self.process_pid_list.append(self.alloc_proxy.pid)

        # 注册定时器任务
        set_schedule_event(
            CONFIG.prometheus_stat_per_minutes, self.predict_stat)
        
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            # on-policy条件下是在actor同步到model文件后开始load_last_new_model, 其他条件下是周期性加载
            if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                set_schedule_event(
                    CONFIG.model_file_sync_per_minutes, self.load_last_new_model)
                
        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            self.load_last_new_model()
        else:
            pass

        # 获取本机IP
        self.host = get_host_ip()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

    '''
    actor上的预测predict主函数, 使用TensorRT
    '''

    def predict_tensorrt(self, datas):
        batch_size = len(datas)

        # 数据整理
        state_dict = {}
        state_space = self.policy_conf.state.state_space()
        for i, key in enumerate(state_space.keys()):
            state_dict[key] = [datas[j][i].flatten()
                               for j in range(batch_size)]

        res_msgs = []
        for i in range(batch_size):
            # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                datas[i][-1][0][3] = self.current_sync_model_version_from_learner

            res_msgs.append({
                KaiwuDRLDefine.CLIENT_ID_TENSOR: datas[i][-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: datas[i][-1]
            })

        sizes = []
        try:
            pred = self.model_wrapper.predict(state_dict, batch_size)
            if pred:
                for i in range(batch_size):
                    res_msgs[i]['pred'] = [p[i] for p in pred]
                    sizes.append(len(pred))

        except Exception as e:
            self.logger.error(
                f"predict failed to run {self.name} predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}")

        # self.logger.debug(f'after predict, size is {sizes}')

        return sizes, res_msgs

    '''
    actor上的预测predict主函数, 使用框架的predict
    '''

    def predict_tensorflow(self):
        size = 0
        pred = None

        batch_size = 1

        try:
            extra_tensors={
                KaiwuDRLDefine.CLIENT_ID_TENSOR: self.dequeue_tensors[-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: self.dequeue_tensors[-1]}
            
            pred = self.model_wrapper.predict(extra_tensors, batch_size)
            size = next(iter(pred.values())).shape[0]

            pred['s'] = np.array([self.global_step] * size)

        except Exception as e:
            self.logger.error(
                f"predict failed to run {self.name} predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}")

        # self.logger.debug(f'after predict, size is {size}, pred is {pred}')

        return size, pred

    def pytorch_predict(self, datas):

        # 组装batch
        # arr 对应state_space中的key
        arr_size = len(datas[0])
        batch_size = len(datas)
        data = []
        for i in range(arr_size):
            data.append([datas[j][i] for j in range(batch_size)])

        # 数据整理
        state_dict = {}
        state_space = self.policy_conf.state.state_space()
        for i, key in enumerate(state_space.keys()):
            # state_dict[key] = [data[i][j].flatten() for j in range(batch_size)]
            state_dict[key] = [data[i][j] for j in range(batch_size)]

        res_msgs = []
        for i in range(batch_size):
            # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                datas[i][-1][0][3] = self.current_sync_model_version_from_learner

            res_msgs.append({
                KaiwuDRLDefine.CLIENT_ID_TENSOR: datas[i][-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: datas[i][-1]
            })

        sizes = []
        try:
            # (format_action, network_sample_info, lstm_info) = pred
            pred = self.model_wrapper.predict(state_dict, batch_size)
            if pred:
                for i in range(batch_size):
                    res_msgs[i]['pred'] = [p[i] for p in pred]
                    sizes.append(len(pred))

        except Exception as e:
            self.logger.error(
                f"predict failed to run {self.name} predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}")

        # self.logger.debug(f'after predict, size is {sizes}')

        return sizes, res_msgs

    '''
    actor上的预测predict主函数, 使用业务的predict
    '''

    def predict_simple(self, datas):

        # 组装batch
        # arr 对应state_space中的key
        batch_size = len(datas)

        # 数据整理
        state_dict = {}
        state_space = self.policy_conf.state.state_space()
        for i, key in enumerate(state_space.keys()):
            state_dict[key] = [datas[j][i].flatten()
                               for j in range(batch_size)]

        res_msgs = []
        for i in range(batch_size):
            # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                datas[i][-1][0][3] = self.current_sync_model_version_from_learner
            
            res_msgs.append({
                KaiwuDRLDefine.CLIENT_ID_TENSOR: datas[i][-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: datas[i][-1]
            })

        sizes = []
        try:
            # self.logger.debug(f'predict actor from aisrv is {state_dict}')
            pred = self.model_wrapper.predict(state_dict, batch_size)
            # self.logger.debug(f'predict actor send to aisrv is {pred}')
            if pred:
                for i in range(batch_size):
                    res_msgs[i]['pred'] = [p[i] for p in pred]
                    sizes.append(len(pred))

        except Exception as e:
            self.logger.error(
                f"predict failed to run {self.name} predictor, as {e}, traceback.print_exc() is {traceback.format_exc()}")

        # self.logger.debug(f'after predict, size is {sizes}')

        return sizes, res_msgs

    '''
    流程:
    判断GPU队列里是否为空:
    1. 队列为空, 等待下次操作
    2. 队列非空, 开始处理预测请求, 并且返回actor_server预测响应    
    '''

    def predict_tensorrt_direct(self):
        size = 0
        pred = None

        # 处理actor --> aisrv的回包
        self.send_server.put_predict_result_data([size, pred])
    
    '''
    从actor_server进程提供的队列收集预测数据, 以线程形式, 暂时不用
    '''
    def get_predict_data_from_actor_server_by_threading(self):
        while True:
            self.get_predict_data_from_actor_server()


    '''
    actor_server获取的预测请求数据放入到on_policy_predictor里
    '''
    def put_to_predict_queue(self, predict_data):
        if not predict_data:
            return
        
        if self.predict_request_queue.full():
            return
        
        self.predict_request_queue.put(predict_data)
    

    '''
    on_policy_predictor的预测结果数据放入到本地后, actor_server从本地拿走
    '''
    def get_predict_result_data(self):
        return self.predict_result_queue.get()

    '''
    从actor_server进程提供的队列收集预测数据, 以函数形式
    1. 如果是pipeline_process_sync为False则从actor_server队列里获取
    2. 如果是pipeline_process_sync为True则从本地队列里获取
    控制条件依据pipeline_process_sync的值:
    1. 如果是False:
        1.1 单次批处理predict_batch_size
        1.2 设置的超时时间
    2. 如果是True:
        2.1 尽最大努力获取数据
        2.2 超过predict_batch_size跳出, 平滑操作
    '''
    def get_predict_data_from_actor_server(self):
        datas = []

        with TimeIt() as it:
            if not CONFIG.pipeline_process_sync:

                # 按照时间间隔和批处理大小收包
                start_time = time.time()
                while len(datas) < int(CONFIG.predict_batch_size):

                    # 区分从哪里获取数据
                    data = None
                    if not CONFIG.actor_server_predict_server_different_queue:
                        data = self.recv_server.get_from_to_predict_queue()
                    else:
                        try:
                            data = self.predict_request_queue_from_actor_server.get()
                        except Exception as e:
                            pass
                    
                    if data:
                        # 增加压缩和解压缩耗时
                        with TimeIt() as ti:
                            decompressed_data = decompress_data(data)
                        if self.max_decompress_time < ti.interval:
                            self.max_decompress_time = ti.interval

                        datas.append(decompressed_data)

                    # 收包超时时强制退出, 平滑处理
                    if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                        break

            else:

                # 最大限度收包
                while not self.predict_request_queue.empty():
                    datas.append(self.predict_request_queue.get())

                    # 最大predict_batch_size的跳出去, 平滑处理
                    if len(datas) > int(CONFIG.predict_batch_size):
                        break
        
        # 如果本次没有数据, 提前返回, 不需要进行处理
        datas_length = len(datas)
        if not datas_length:
            self.process_run_idle_count += 1
            return datas
        
        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing get_predict_data_from_actor_server end")
        
        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < datas_length:
            self.actor_from_zmq_queue_size = datas_length

        if self.actor_from_zmq_queue_cost_time_ms < it.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = it.interval * 1000
        
        return datas

    '''
    预测函数
    '''
    def predict(self, datas):

        if not datas or not len(datas):
            return
        
        # 返回的数据格式
        size = 0
        pred = None

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing predict start")

        if KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_simple(datas)

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorflow()

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_simple(datas)

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
            pass

        elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorrt(datas)

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        else:
            self.logger.error(f'predict error use_which_deep_learning_framework {CONFIG.use_which_deep_learning_framework}, only suport {KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, \
                {KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, {KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}')

            return

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing predict end")

        # 处理actor --> aisrv的回包
        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing predict put actor_server predict result start")

        '''
        处理actor->aisrv的响应回包
        '''
        if not CONFIG.pipeline_process_sync:
            if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                 self.send_response_to_aisrv(size, pred)
            else:
                self.send_response_to_aisrv_simple_fast(size, pred)
        else:
            self.predict_result_queue.put([size, pred])

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing predict put actor_server predict result end")

    '''
    zmq_ops模式下发送响应包
    '''
    def send_response_to_aisrv(self, size, pred):

        if CONFIG.distributed_tracing:
            self.logger.info(f'actor_server distributed_tracing send_response_to_aisrv start', g_not_server_label)
    
        client_ids = pred.pop(KaiwuDRLDefine.CLIENT_ID_TENSOR)
        compose_ids = pred.pop(KaiwuDRLDefine.COMPOSE_ID_TENSOR)

        with TimeIt() as ti:
            dict_obj = {}
            for i in range(size):
                send_data = {k: v[i] for k, v in pred.items()}
                client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                list_obj = dict_obj.setdefault(client_id, [])
                list_obj.append((tuple(compose_id), send_data))
                # self.logger.debug(f'actor_server zmq server will send a new msg {compose_id} to {client_id}', g_not_server_label)
            
            for client_id, send_data in dict_obj.items():
                try:
                    self.zmq_server.recv(block=False, binary=True)
                except Exception as e:
                    pass
                
                self.zmq_server.send(str(client_id), send_data, binary=True)
                self.send_to_aisrv_succ_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(f'actor_server distributed_tracing zmq server send a new msg to {compose_id} success', g_not_server_label)

        if CONFIG.distributed_tracing:
            self.logger.info(f'actor_server distributed_tracing send_response_to_aisrv end', g_not_server_label)
        
    '''
    actor给aisrv的回包组包处理, 免得阻塞actor_server进程
    '''
    def send_response_to_aisrv_simple_fast(self, size, preds):
        if CONFIG.distributed_tracing:
            self.logger.info('actor_server distributed_tracing send_response_to_aisrv_simple_fast start', g_not_server_label)

        dict_obj = {}

        for j, pred in enumerate(preds):
            client_ids = pred[KaiwuDRLDefine.CLIENT_ID_TENSOR]
            compose_ids = pred[KaiwuDRLDefine.COMPOSE_ID_TENSOR]

            if CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                send_data = pred['pred']
                client_id = client_ids[0]
                compose_id = compose_ids[0]
                res = [(tuple(compose_id), send_data)]
                dict_obj[client_id] = res
            else:
                send_data = {
                    'format_action': pred['pred'][0],
                    'network_sample_info': pred['pred'][1],
                    'lstm_info': pred['pred'][2]
                }

                for i in range(size[j] - 2):
                    client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                    compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                    list_obj = dict_obj.setdefault(client_id, [])
                    list_obj.append((tuple(compose_id), send_data))

        for client_id, send_data in dict_obj.items():

            # 压缩耗时和压缩包大小
            with TimeIt() as ti:
                compressed_data = compress_data(send_data)

            if self.max_compress_time < ti.interval:
                self.max_compress_time = ti.interval

            compress_msg_len = len(compressed_data)
            if self.max_compress_size < compress_msg_len:
                self.max_compress_size = compress_msg_len

            # 这里直接放置的是client_id, compressed_data对
            self.send_server.put_predict_result_data([client_id, compressed_data])

            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                with self.send_server.predict_result_condition:
                    self.send_server.predict_result_condition.notify()

            if CONFIG.distributed_tracing:
                self.logger.info(f'actor_server distributed_tracing zmq server send a new msg to {client_id} success', g_not_server_label)

        if CONFIG.distributed_tracing:
            self.logger.info('actor_server distributed_tracing send_response_to_aisrv_simple_fast end')

    '''
    actor采用tensorrt前提下流水线处理
    '''

    def run_once_tesnorrt(self):
        # 步骤1, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 进行预测, 并且获取预测响应
        self.predict_tensorrt_direct()

    def run_once(self):

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, actor上执行on-policy流程
        self.actor_on_policy_process()

        # 步骤3, 从zmq/zmq-ops上获取data/tensor进行预测, 这里按照批处理获取数据, 尽最大努力去拿取队列里的数据, 如果没有则跳出该循环
        datas = self.get_predict_data_from_actor_server()
        if datas:

            # 步骤4, 预测
            self.predict(datas)

        # Model文件同步操作, learner --> actor, 采用单独的进程处理
    
    '''
    actor上执行on-policy流程
    '''
    def actor_on_policy_process(self):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.actor_on_policy_process_detail()

    '''
    actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制

    '''
    def actor_get_model_from_modelpool(self):
        all_pull_model_success = False
        retry_count = 0

        while not all_pull_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            pull_model_success = self.model_file_sync_wraper.pull_checkpoint_from_model_pool(self.logger)
            if not pull_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_pull_model_success = True
                self.logger.info(f'predict learner pull_checkpoint_from_model_pool success')
                break
            
            retry_count += 1
        
        return all_pull_model_success

    '''
    actor上的on-policy的处理流程:
    1. 同步model_version请求
    1.1 获取来自learnerd model文件同步请求
    1.2 actor重新从modelpool获取文件
    1.2.1 如果成功则继续剩余流程 
    1.2.2 失败则返回learner的明确失败的结果, learner根据情况决定是否让aisrv执行更新model_version操作, actor等待下一次model_version改变再走该流程
    1.2.2.1 如果actor返回给learner执行model_version失败, 则learner不能让aisrv执行修改model_version操作
    1.2.2.2 如果actor返回给learner执行model_version成功, 则learner让aisrv执行修改model_version操作
    1.3 actor加载最新model文件
    1.4 朝learner发送model文件同步响应
    2. 心跳请求
    2.1 心跳响应
    '''
    def actor_on_policy_process_detail(self):
        try:
            # 获取来自learner的 model文件同步请求
            client_id, message = self.zmq_server.recv(block=False, binary=False)
            if message:
                if message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST:

                    '''
                    actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制

                    '''
                    actor_get_model_file_success = False
                    for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
                        if self.actor_get_model_from_modelpool():
                            actor_get_model_file_success = True
                            break
                    
                    '''
                    根据actor从modelpool拉取model文件执行下面流程:
                    1. 成功, actor加载最新model文件, 更新当前self.current_sync_model_version_from_learner值, 回复learner响应
                    2. 失败, actor告警指标++, 回复learner响应
                    '''
                    actor_execute_on_policy_success = False
                    if not actor_get_model_file_success:
                        self.logger.error(f'predict learner pull_checkpoint_from_model_pool failed, so return, not change model_version: {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE]}')
                        self.on_policy_pull_from_modelpool_error_cnt += 1
                    else:
                        # actor加载最新model文件
                        self.load_last_new_model()
                        actor_execute_on_policy_success = True
                        self.current_sync_model_version_from_learner = message[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE]
                        self.on_policy_pull_from_modelpool_success_cnt += 1
                        self.logger.info(f"predict learner ask actor to set model_version: {self.current_sync_model_version_from_learner} success")

                    # actor朝learner发送model文件同步响应
                    send_data = {
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE,
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE: actor_execute_on_policy_success
                                }
                    if actor_execute_on_policy_success:
                        self.actor_change_model_version_success_count += 1
                    else:
                        self.actor_change_model_version_error_count += 1
                    
                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.info(f"predict learner ask actor to {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]} success")

                elif message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:
                    # actor朝learner发送心跳响应
                    send_data = {
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE,
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE
                                }
                    
                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.debug(f"predict learner ask actor to {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]} success")

                else:
                    self.logger.error(f'predict learner learner_model_sync_req not support message_type {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]}, so return')
                    return

        except Exception as e:
            pass
    
    def set_index(self, index):
        self.index = index

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    def set_predict_request_queue_from_actor_server(self, predict_request_queue_from_actor_server):
        if not predict_request_queue_from_actor_server:
            return
        
        self.predict_request_queue_from_actor_server = predict_request_queue_from_actor_server

    def run(self):
        self.before_run()

        while not self.model_wrapper.should_stop():
            try:
                self.run_once()

                # 因为在pipeline_process_sync模式下一直从本地收包容易导致CPU100%, 而在非pipeline_process_sync模式下有收包超时时间反而不容易发生
                if CONFIG.pipeline_process_sync:
                    # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于存在actor的按照时间间隔去预测, 故这里不休眠, 后期修改为事件机制
                    if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                        time.sleep(CONFIG.idle_sleep_second)

                        # process_run_count置0, 规避溢出
                        self.process_run_idle_count = 0

            except Exception as e:
                self.logger.error(
                    f"failed to run {self.name} predict. exit. Error is: {e}, traceback.print_exc() is {traceback.format_exc()}")

        if CONFIG.actor_server_async:
            self.send_server.stop()
            self.recv_server.stop()
            self.logger.info('predict self.send_server.stop self.recv_server.stop success')
        else:
            self.send_server.stop()
            self.logger.info('predict self.send_server.stop success')

        self.model_wrapper.close()
        self.logger.info('predict self.model_wrapper.close success')

        # 非on-policy的才需要主动关闭self.model_file_sync_wraper
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            self.model_file_sync_wraper.stop()
            self.logger.info('predict self.model_file_sync_wraper.stop success')
