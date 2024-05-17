#!/usr/bin/env python3
# -*- coding:utf-8 -*-



'''
@Project :kaiwu-fwk 
@File    :onpolicy_predictor_pipeline.py
@Author  :drewjiang
@Date    :2022/11/17 11:02

'''

import os
import sys
import time
import traceback
import yaml
from framework.common.utils.rainbow_utils import RainbowUtils
from framework.common.checkpoint.model_file_save import ModelFileSave
import lz4.block
from framework.common.utils.tf_utils import *
import schedule
from pydoc import locate
from framework.server.actor.predictor import Predictor
from framework.common.utils.common_func import TimeIt, set_schedule_event, make_single_dir, actor_learner_aisrv_count, get_host_ip, python_exec_shell, get_gpu_machine_type
from framework.common.config.algo_conf import AlgoConf
from framework.common.config.config_control import CONFIG
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from framework.common.pybind11.zmq_ops.zmq_ops import ZMQPullSocket

from framework.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from framework.common.alloc.alloc_proxy import AllocProxy
from framework.common.monitor.monitor_proxy import MonitorProxy


class OnPolicyPredictor_Pipeline(Predictor):

    def __init__(self, server, name):
        super().__init__(server, name)

    '''
    actor周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    '''
    def rainbow_activate(self):
        
        self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
        self.rainbow_activate_single_process(CONFIG.svr_name)

    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            self.logger.error(f'predict read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(f'predict read_from_rainbow failed, data is None or data len is 0')
            return
            
        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(process_name, to_change_key_values)
        self.logger.info(f"predict {process_name} CONFIG save_to_file success")

    
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
    
    '''
    启动C++常驻进程, 注意这里的路径对齐
    '''
    def start_cpp_daemon(self, version):

        if version == "v5":
            cmd = 'sh tools/actor_cpp_server_start.sh'
        else:
            cmd = 'sh tools/actor_cpp_server_start_v4.sh'
            
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            return False
        
        self.logger.info(f'predict C++ Daemon Process starts success, cmd is {cmd}')
        return True
    
    def before_run(self):

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

        '''
        专门处理aisrv <--> actor的消息类启动:
        
        actor采用python进程方式:
        1. server启动, 处理预测请求/预测响应收发

        actor采用上层为python进程, 下层为cpp Daemon进程形式:
        1. server即receive_server启动, 处理预测请求收操作
        2. send_server启动, 处理预测请求发操作

        actor采用C++常驻进程处理数据收发,则不需要启动python侧的数据收发进程
        '''
        if not CONFIG.cpp_daemon_send_recv_zmq_data:
            self.server.start()

        '''
        如果actor采用tensorrt, 则需要启动下面进程来进行并行处理:
        1. python和C++常驻进程方式
        2. python常驻进程方式, C++提供调用接口方式:
           a. 同步方式, 启动多个预测进程, 每个预测进程里预测是同步执行的
           b. 异步方式, 启动多个预测进程, 每个预测进程里的请求拷贝,预测是异步执行的
        '''
        assert  KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework
        
        # C++端, 常驻进程启动, 如果无法启动C++进程, 则本进程退出
        if not CONFIG.cpp_daemon_send_recv_zmq_data:
            version= "v4"
        else:
            version = "v5"
        
        if not self.start_cpp_daemon(version):
            self.logger.error(f'predict C++ Daemon Process starts failed, please see the log')
            sys.exit(-1)

        # C++和python进程不能同时启动
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)

        # 下面是进程集合, 指标不一样, 需要分开统计
        self.actor_tensort_cpu2gpu_processes = []
        self.actor_tensort_gpu2cpu_processes = []

        # 如果zmq的数据收发放在C++侧, 则不需要启动下面的进程
        if not CONFIG.cpp_daemon_send_recv_zmq_data:
            # 根据配置设置不同的actor_server
            if int(CONFIG.actor_server_async):
                actor_server_receive = self.server.get_receive_server()
                actor_server_send = self.server.get_send_server()
            else:
                actor_server_receive = self.server
                actor_server_send = self.server

            for i in range(int(CONFIG.actor_cpu_2gpu_thread_num)):

                # python端, 拷贝预测请求队列进程启动
                p_cpu2gpu = ActorTensorRTCPU2GPU(actor_server_receive, self.policy_conf)
                self.actor_tensort_cpu2gpu_processes.append(p_cpu2gpu)
                p_cpu2gpu.start()
                self.logger.info(f'predict python ActorTensorRTCPU2GPU {i}th Process starts success')
                # python端, 拷贝预测响应队列进程启动
                p_gpu2cpu = ActorTensorRTGPU2CPU(actor_server_send)
                self.actor_tensort_gpu2cpu_processes.append(p_gpu2cpu)
                p_gpu2cpu.start()
                self.logger.info(f'predict python ActorTensorRTGPU2CPU {i}th  Process starts success')
        
        
        # model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程
        self.model_file_sync_wraper = ModelFileSyncWrapper()
        self.model_file_sync_wraper.init()

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        if int(CONFIG.use_prometheus):
            # 启动独立的进程, 负责actor与普罗米修斯交互
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()
            
        # 注册定时器任务
        if CONFIG.cpp_daemon_send_recv_zmq_data:
            os.chdir(CONFIG.tensorrt_engine_dir)
            
            # 获取GPU机型
            gpu_machine_type = get_gpu_machine_type()
            actor_py_server = locate(f'framework.server.cpp.dist.actor.{gpu_machine_type}.actor_server.actor_py_server')

            self.lib = actor_py_server()
            self.logger.info(f'predict C++ lib start success')

            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.predict_cpp_stat)
        else:
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.predict_stat)

        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()
    
    '''
    python侧从C++侧获取统计数据, 并且进行上报普罗米修斯
    1. predict_succ_cnt, 预测成功数目
    2. ACTOR_TCP_AISRV, 现场计算
    '''
    def predict_cpp_stat(self):
        result = self.lib.get_cpp_server_stat_data()
        if not result or not len(result):
            return
        
        # 进行上报, 注意取出来的数据需要强制转换下数据类型
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT : result.get(KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT),
                KaiwuDRLDefine.ACTOR_TCP_AISRV : actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT: result.get(KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT),
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT : result.get(KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT),
                KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_SUCC_CNT : result.get(KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_SUCC_CNT),
                KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_ERROR_CNT : result.get(KaiwuDRLDefine.MONITOR_ACTOR_RECEIVEFROM_AISRV_ERROR_CNT),
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE : result.get(KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE),
                KaiwuDRLDefine.MONITOR_TENSORRT_REFIT_SUC_CNT :  result.get(KaiwuDRLDefine.MONITOR_TENSORRT_REFIT_SUC_CNT),
                KaiwuDRLDefine.MONITOR_TENSORRT_REFIT_ERR_CNT :  result.get(KaiwuDRLDefine.MONITOR_TENSORRT_REFIT_ERR_CNT),
            }

            self.monitor_proxy.put_data(monitor_data)

            # 指标周期性复原
            self.lib.cpp_server_stat_data_reset()
    
    '''
    增加predict统计, 原则上是将所有进程的统计值进行下面处理:
    1. predict_cnt, 多个进程相加, 取和
    2. actor_from_zmq_queue_size, 取最大值
    3. actor_from_zmq_queue_cost_time_ms, 取最大值
    4. actor_batch_predict_cost_time_ms, 取最大值
    5. push_to_cuda_queue_cost_time_ms, 取最大值
    6. ACTOR_TCP_AISRV, 现场计算
    7. actor_load_last_model_cost_ms, 取最大值
    '''
    def predict_stat(self):

        monitor_data = {}

        predict_cnt = 0
        actor_from_zmq_queue_size = 0
        actor_from_zmq_queue_cost_time_ms = 0
        push_to_cuda_queue_cost_time_ms = 0

        # 处理各个进程的统计值
        if not int(CONFIG.python_cpp_daemon):
            actor_batch_predict_cost_time_ms = 0
            actor_load_last_model_cost_ms = 0

            for predict_process in self.predict_processes:
                p_predict_cnt, p_actor_from_zmq_queue_size, p_actor_from_zmq_queue_cost_time_ms, p_actor_batch_predict_cost_time_ms, \
                    p_push_to_cuda_queue_cost_time_ms, p_actor_load_last_model_cost_ms = predict_process.get_predict_stat_data()

                predict_cnt += p_predict_cnt
                if actor_from_zmq_queue_size < p_actor_from_zmq_queue_size:
                    actor_from_zmq_queue_size = p_actor_from_zmq_queue_size
                if actor_from_zmq_queue_cost_time_ms < p_actor_from_zmq_queue_cost_time_ms:
                    actor_from_zmq_queue_cost_time_ms = p_actor_from_zmq_queue_cost_time_ms
                if actor_batch_predict_cost_time_ms < p_actor_batch_predict_cost_time_ms:
                    actor_batch_predict_cost_time_ms = p_actor_batch_predict_cost_time_ms
                if push_to_cuda_queue_cost_time_ms < p_push_to_cuda_queue_cost_time_ms:
                    push_to_cuda_queue_cost_time_ms = p_push_to_cuda_queue_cost_time_ms
                if actor_load_last_model_cost_ms < p_actor_load_last_model_cost_ms:
                    actor_load_last_model_cost_ms = p_actor_load_last_model_cost_ms
                
                # 完成统计指标重置操作
                predict_process.predict_stat_reset()
            
            if int(CONFIG.use_prometheus):
                monitor_data[KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS] = actor_batch_predict_cost_time_ms
                monitor_data[KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS] = actor_load_last_model_cost_ms

        else:
            actor_tensorrt_cpu_send_to_gpu_succ_cnt = 0
            actor_tensorrt_cpu_send_to_gpu_error_cnt = 0

            for predict_process in self.actor_tensort_cpu2gpu_processes:
                p_actor_tensorrt_cpu_send_to_gpu_succ_cnt, p_actor_tensorrt_cpu_send_to_gpu_error_cnt, \
                p_push_to_cuda_queue_cost_time_ms, p_actor_from_zmq_queue_size, p_actor_from_zmq_queue_cost_time_ms = predict_process.get_cpu_send2gpu_stat()

                ''' 
                在python和C++都是常驻进程时, predict_cnt 的计算方式为 actor_tensorrt_cpu_send_to_gpu_succ_cnt * actor_from_zmq_queue_size, 多个进程累加
                但是每分钟里actor_from_zmq_queue_size是变化的, 该值不一定准确, 故在该场景下该指标不做统计
                '''
                #predict_cnt += p_actor_tensorrt_cpu_send_to_gpu_succ_cnt * p_actor_from_zmq_queue_size

                actor_tensorrt_cpu_send_to_gpu_succ_cnt += p_actor_tensorrt_cpu_send_to_gpu_succ_cnt
                actor_tensorrt_cpu_send_to_gpu_error_cnt += p_actor_tensorrt_cpu_send_to_gpu_error_cnt
                if push_to_cuda_queue_cost_time_ms < p_push_to_cuda_queue_cost_time_ms:
                    push_to_cuda_queue_cost_time_ms = p_push_to_cuda_queue_cost_time_ms
                if actor_from_zmq_queue_size < p_actor_from_zmq_queue_size:
                    actor_from_zmq_queue_size = p_actor_from_zmq_queue_size
                if actor_from_zmq_queue_cost_time_ms < p_actor_from_zmq_queue_cost_time_ms:
                    actor_from_zmq_queue_cost_time_ms = p_actor_from_zmq_queue_cost_time_ms
                
                # 完成统计指标重置操作
                predict_process.cpu_send2gpu_stat_reset()

            if int(CONFIG.use_prometheus):
                monitor_data[KaiwuDRLDefine.ACTOR_TENSORRT_CPU2GPU_SUCC_CNT] = actor_tensorrt_cpu_send_to_gpu_succ_cnt
                monitor_data[KaiwuDRLDefine.ACTOR_TENSORRT_CPU2GPU_ERR_CNT] = actor_tensorrt_cpu_send_to_gpu_error_cnt

            actor_tensorrt_gpu_send_to_cpu_succ_cnt = 0
            actor_tensorrt_gpu_send_to_cpu_error_cnt = 0

            for predict_process in self.actor_tensort_gpu2cpu_processes:
                p_actor_tensorrt_gpu_send_to_cpu_succ_cnt, p_actor_tensorrt_gpu_send_to_cpu_error_cnt = predict_process.get_gpu_send2cpu_stat()
                actor_tensorrt_gpu_send_to_cpu_succ_cnt += p_actor_tensorrt_gpu_send_to_cpu_succ_cnt
                actor_tensorrt_gpu_send_to_cpu_error_cnt += p_actor_tensorrt_gpu_send_to_cpu_error_cnt

                predict_process.gpu_send2cpu_stat_reset()

            if int(CONFIG.use_prometheus):
                monitor_data[KaiwuDRLDefine.ACTOR_TENSORRT_GPU2CPU_SUCC_CNT] = actor_tensorrt_gpu_send_to_cpu_succ_cnt
                monitor_data[KaiwuDRLDefine.ACTOR_TENSORRT_GPU2CPU_ERR_CNT ] = actor_tensorrt_gpu_send_to_cpu_error_cnt
        
        # 进行上报
        if int(CONFIG.use_prometheus):

            # 放置公共的统计监控指标
            monitor_data[KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT ] = predict_cnt
            monitor_data[KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE] = actor_from_zmq_queue_size
            monitor_data[KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS] = actor_from_zmq_queue_cost_time_ms
            monitor_data[KaiwuDRLDefine.ACTOR_TCP_AISRV] = actor_learner_aisrv_count(self.host, CONFIG.svr_name)
            monitor_data[KaiwuDRLDefine.MONITOR_PUSH_TO_CUDA_QUEUE_COST_TIME_MS] = push_to_cuda_queue_cost_time_ms

            self.monitor_proxy.put_data(monitor_data)

    def run_once(self):

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

    def loop(self):
        self.before_run()

        while True:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0
            
            except Exception as e:
                self.logger.error(f"failed to run {self.name} predict. exit. Error is: {e}, traceback.print_exc() is {traceback.format_exc()}")
        
                self.server.stop()
                self.logger.info('predict self.server.stop success')

                self.model_file_sync_wraper.stop()
                self.logger.info('predict self.model_file_sync_wraper.stop success')
