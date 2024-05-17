#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from pydoc import locate
from multiprocessing import Value
import time
import traceback
import re
from framework.common.utils.rainbow_utils import RainbowUtils
import schedule
import os
import yaml
from framework.common.utils.tf_utils import *
from framework.server.learner.trainer import Trainer
from framework.common.replay_buffer.replay_buffer_wrapper import ReplayBufferWrapper
from framework.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from framework.common.checkpoint.model_file_sync import ModelFileSync
from framework.common.config.config_control import CONFIG
from framework.common.config.algo_conf import AlgoConf
from framework.common.algorithms.model_wrapper_builder import ModelWrapperBuilder
from framework.common.checkpoint.model_file_save import ModelFileSave
from framework.common.utils.common_func import TimeIt, set_schedule_event, make_single_dir, actor_learner_aisrv_count, get_host_ip, get_uuid
from framework.common.alloc.alloc_proxy import AllocProxy
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.ipc.zmq_util import ZmqServer, ZmqClient
from framework.common.alloc.alloc_utils import AllocUtils

class OnPolicyTrainer(Trainer):
    @property
    def tensor_names(self):
        raise NotImplementedError

    @property
    def tensor_dtypes(self):
        raise NotImplementedError

    @property
    def tensor_shapes(self):
        raise NotImplementedError

    def __init__(self, name):
        super(OnPolicyTrainer, self).__init__(name)

        self.cached_local_step = -1

        self.local_step = Value('d', -1)
    
    def create_model_wrapper(self):
        '''
        {
            "ppo": {
                "actor_model": "framework.common.algorithms.model.Model",
                "learner_model": "framework.common.algorithms.model.Model",
                "trainer": "framework.server.learner.ppo_trainer.PPOTrainer",
                "predictor": "framework.server.actor.ppo_predictor.PPOPredictor",
                "expr_processor": "framework.common.algorithms.ppo_processor.PPOProcessor",
                "default_config": "framework.common.algorithms.ppo.PPODefaultConfig"
            }
        }
        '''

        with TimeIt() as ti:
            # network
            network = self.policy_conf.learner_network(
                 self.policy_conf.state.state_space(),
                 self.policy_conf.action.action_space()
            )

            # model
            name = "%s_%s" % (CONFIG.app, CONFIG.algo)
            self.model = AlgoConf[CONFIG.algo].learner_model(network, name, CONFIG.svr_name)

            self.model_wrapper = ModelWrapperBuilder().create_model_wrapper(self.model, self.logger)
            if KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.build_train_graph(self.input_tensors)
                self.model_wrapper.add_chief_only_hooks(self.chief_only_hooks())
                self.model_wrapper.add_train_hooks(self.train_hooks())
                self.model_wrapper.create_train_session()
                self.model_wrapper.sess._tf_sess().run(self.replay_buffer_wrapper.extra_initializer_ops())
                self.local_step.value = self.model_wrapper.get_global_step()

            # 在tensorflow_simple和tensorrt模式下, 采用的learner逻辑是一致的
            elif KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework or KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.set_dataset(self.replay_buffer_wrapper)
                self.model_wrapper.build_model()
            
            elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
                self.model_wrapper.set_dataset(self.replay_buffer_wrapper)

            elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
                pass

            else:
                self.logger.error(f'train error use_which_deep_learning_framework {CONFIG.use_which_deep_learning_framework}, only suport {KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, \
                {KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, {KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}')
                
                return

            self.logger.info(f'train process start, model_wrapper is {self.model_wrapper.name}')
    
    # 当作为主learner时, 需要保存ckpt文件
    def chief_only_hooks(self):
        with tf.device(f"{self.model_wrapper.learner_device}/cpu:0"):
            return [self.model.ckpt_saver_hook()]

    def train_hooks(self):
        with tf.device(f"{self.model_wrapper.learner_device}/cpu:0"):
            return self.replay_buffer_wrapper.train_hooks(self.model_wrapper.local_step)
    
    '''
    根据不同的启动方式进行处理:
    1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
    2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
    '''
    def start_learner_process_by_type(self):

        # 按照需要引入ModelFileSave
        self.model_file_saver = ModelFileSave()
        self.model_file_saver.start_actor_process_by_type(self.logger)
    
    '''
    learner周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    '''
    def rainbow_activate(self):
        
        self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
        self.rainbow_activate_single_process(CONFIG.svr_name)
    
    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            self.logger.error(f'train read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(f'train read_from_rainbow failed, data is None or data len is 0')
            return
            
        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(process_name, to_change_key_values)
        self.logger.info(f"train {process_name} CONFIG save_to_file success")
    
    '''
    learn上的训练train函数流程, 返回是否真实的训练
    '''
    def train_detail(self):

        '''
        直接调用业务返回的数据格式上报, 框架不关心具体的类型和值, 格式是map
        '''
        with TimeIt() as ti:
            app_monitor_data, has_model_file_changed, model_file_id = self.model_wrapper.train()
            if app_monitor_data and isinstance(app_monitor_data, dict):
                self.app_monitor_data = app_monitor_data

        if self.batch_train_cost_time_ms < ti.interval * 1000:
            self.batch_train_cost_time_ms = ti.interval * 1000
        
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY and has_model_file_changed:
            self.current_sync_model_version_from_learner = model_file_id

            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_TIME_INTERVAL:
                self.learner_on_policy_process(True)

    '''
    aisrv启动的learner的on-policy流程
    '''
    def learner_on_policy_process_by_aisrv(self):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                self.learner_on_policy_process_by_aisrv_detail()

    '''
    learner访问aisrv
    1. 心跳请求, 对于心跳响应的返回值不同则处理方式不同
    1.1 如果心跳响应里某个aisrv要求learner需要走on-policy流程
    1.2 如果心跳响应里没有aisrv要求learner需要走on-policy流程
    '''
    def on_policy_learner_connect_to_aisrv(self):
        if not self.aisrv_zmq_client_map:
            return
        
        '''
        如果是需要aisrv发起来的on-policy流程, 此时需要朝aisrv发送自己的client_id
        '''
        learner_send_to_aisrv_heartbeat_success_count = 0
        for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
            send_data = {
                            KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
                            KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST
                        }
            zmq_client.send(send_data, binary=False)
            self.logger.debug(f"train send heartbeat request to aisrv: {aisrv_ip} success")

        '''
        同步等待心跳响应回包, 这里因为aisrv在心跳包里会带上是否让learner启动on-policy流程, 故需要注意的点:
        1. 计算心跳回包和计算aisrv启动on-policy的数量需要分开
        2. end_time只能增加1次, 否则进入死循环, 无法退出
        3. 理论上因为aisrv在CONFIG.on_policy_timeout_seconds时间里判断是否有on-policy流程, 故站在leaner的角度看是需要获取所有的aisrv的响应回包
           理论上该值为len(self.aisrv_zmq_client_map) * CONFIG.on_policy_timeout_seconds, 但是配置过大导致learner的主循环阻塞, 故折中设置为2 * CONFIG.on_policy_timeout_seconds
        '''
        end_time = time.time() + 2 * CONFIG.on_policy_timeout_seconds
        success_recv_aisrv_ip = []
        any_recv_aisrv_ask_on_policy_success = False
        update_end_time = False
        while time.time() < end_time:
            for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
                if aisrv_ip not in success_recv_aisrv_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            # 收到了aisrv让learner启动on-policy流程时, 需要发送确认响应
                            if recv_data[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE:
                                success_recv_aisrv_ip.append(aisrv_ip)

                                # 判断aisrv发送的需要learner的on-policy请求
                                if recv_data[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE][KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST]:
                                    any_recv_aisrv_ask_on_policy_success = True
                                    self.logger.info(f"train learner recv aisrv {aisrv_ip} ask to execute on-policy request success")
                                
                                learner_send_to_aisrv_heartbeat_success_count += 1
                                self.logger.debug(f"train learner recv aisrv {aisrv_ip}  heartbeat response success")
                
                            else:
                                pass

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            '''
            跳出循环的条件:
            1. 有aisrv发送on-policy请求, 并且已经满足比例的aisrv收到on-policy的请求则跳出循环
            2. 没有aisrv发送on-policy请求, 并且已经满足所有的aisrv的心跳收到请求则跳出循环
            '''
            if any_recv_aisrv_ask_on_policy_success:
                if len(success_recv_aisrv_ip) / len(self.aisrv_zmq_client_map) >= CONFIG.on_policy_quantity_ratio:
                    break
            else:
                if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
                    break

            # 如果有任何aisrv发送了on-policy的请求, 则满足最后那个aisrv发送请求的最大延长时间
            if any_recv_aisrv_ask_on_policy_success and not update_end_time:
                end_time = time.time() + CONFIG.on_policy_timeout_seconds
                update_end_time = True
        
        # 如果本周期内有aisrv发起了让learner去执行on-policy流程的通知, 但是个数不相等的话即接入告警, 否则走on-policy流程
        if any_recv_aisrv_ask_on_policy_success:
            if len(success_recv_aisrv_ip) /  len(self.aisrv_zmq_client_map) < CONFIG.on_policy_quantity_ratio:
                keys1 = set(success_recv_aisrv_ip)
                keys2 = set(self.aisrv_zmq_client_map.keys())

                # 增加告警和容灾
                self.on_policy_learner_recv_aisrv_error_count += 1

                self.logger.error(f'train process learner not recv aisrv ask on-policy request ips: {list(keys2-keys1)}')
            else:
                self.on_policy_learner_recv_aisrv_success_count += 1

                # 开始执行train的操作, 然后再让learner走on-policy流程
                is_train_success = self.train()

                self.learner_on_policy_process(is_train_success)
        else:
            if learner_send_to_aisrv_heartbeat_success_count == len(self.aisrv_zmq_client_map):
                self.logger.info(f"train learner recv all aisrv heartbeat response success, count: {learner_send_to_aisrv_heartbeat_success_count}")

            else:
                # 由于无法收到aisrv的请求, 那么此时不确定aisrv的情况是怎么样, 故清空self.aisrv_zmq_client_map, 重新拉取看下效果
                self.logger.error(f"train learner not recv all aisrv heartbeat response, retry next time")

                self.aisrv_zmq_client_map.clear()
                self.on_policy_learner_get_aisrv_address()

    '''
    on-policy的流程, 从aisrv角度启动
    1. 获取所有aisrv发起的需要执行on-policy流程的个数, 即返回来的zmq信息消息数目 = 从alloc服务获取的返回的aisrv列表
    2. 如果1满足, 则开启on-ploicy流程
    3. 如果1不满足, 则等待到超时时间后即告警, 后期做容灾处理
    '''
    def learner_on_policy_process_by_aisrv_detail(self):

        # 处理learner <--> aisrv之间的心跳请求/响应
        self.on_policy_learner_connect_to_aisrv()

        # 处理learner <--> actor之间的心跳请求/响应
        self.on_policy_learner_connect_to_actor()

    '''
    on-policy需要启动流程:
    1. 由于本次训练是依靠样本消耗比, 故本次不一定能训练, 根据是否训练下面操作:
    1.1 训练成功则:
    1.1.1 清空样本池, 不会失败
    1.1.2 learner推送model文件到modelpool
    1.1.2.1 如果成功则继续剩余流程 
    1.1.2.2 失败则告警指标增加
    1.1.3 learner通知aisrv最新model文件版本号
    1.1.3.1 如果成功则继续剩余流程
    1.1.3.2 如果不成功则告警, 下一步做容灾
    1.1.4 learner等待aisrv获取最新model文件版本号完毕通知
    1.14.1 如果成功则继续剩余流程
    1.1.4.2 如果不成功则告警, 下一步做容灾
    1.1.5 learner通知actor从modelpool拉取model文件
    1.1.5.1 如果成功则继续剩余流程
    1.1.5.2 如果不成功则告警, 下一步做容灾
    1.1.6 learner等待actor确认加载model文件完毕通知
    1.1.6.1 如果成功则继续剩余流程
    1.1.6.2 如果不成功则告警, 下一步做容灾
    1.2 训练不成功
    1.2.1 learner通知aisrv最新model文件版本号
    1.2.1.1 如果成功则本次完成
    1.2.1.2 如果不成功则告警, 下一步做容灾
    '''
    def learner_on_policy_process(self, is_train_success):

        # 清空样本池, 如果本次有进行训练才能清空样本池, 否则不需要清空, 如果强制清空, 下次learner可能会卡在reverb读写上面

        if is_train_success:
            self.replay_buffer_wrapper.reset(self.cached_local_step, self.model_wrapper.tf_sess)
            self.logger.info(f'train learner have train, so reverb reset success')
        else:
            self.logger.info(f'train learner not have train, so reverb not need reset')

        '''
        消息格式:
        message_type: xxxx
        message_value: yyyy
        '''
        send_data = {
                        KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST, 
                        KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE: self.current_sync_model_version_from_learner, 
                    }
        
        if is_train_success:
        
            # learner推送model文件到modelpool, 有重试机制
            learner_push_model_file_success = False
            for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
                if self.learner_push_model_to_modelpool():
                    learner_push_model_file_success = True
                    break
            '''
            如果本次leaner推送到modelpool失败时, learner自身来说可以下一次再推送model文件重试, 并且可以下一次再走on-policy流程, 下面处理方法优缺点
            1. 告警指标++, 并且需要同步aisrv, actor最新的model_version
            1.1 缺点: actor上的model_version和实际的model文件不一致
            1.2 优点: aisrv上的从actor同步到的model_version和从learner同步到的model_version是一致的, aisrv的筛选样本的逻辑不会出问题
            2. 告警指标++, 同步aisrv但是不同步actor最新model_version
            2.1 优点: 减少1次learner和actor的model_version通信
            2.2 缺点: aisrv上的从actor同步到的model_version和从learner同步到的model_version是不一致的, aisrv的筛选样本的逻辑出现问题, 训练无法继续

            目前选择方法1
            '''
            if not learner_push_model_file_success:
                self.logger.error(f'train process learner push_checkpoint_to_model_pool failed, so return')
                self.on_policy_push_to_modelpool_error_count += 1
            
            else:
                self.on_policy_push_to_modelpool_success_count += 1
                # on_policy_learner_change_model_version_cnt代表是真实的model_version次数, 故只有在真实的同步时计数
                self.on_policy_learner_change_model_version_cnt += 1

            '''
            learner通知actor, aisrv更新model文件版本号, 先更新actor端, 再更新aisrv端, 
            原因: 
            1. 如果先更新aisrv端的model版本号, 如果actor再更新model文件失败, aisrv就开始过滤掉样本, 从而引起没有新的样本发送给learner
            2. 如果先更新actor的model文件版本号, 如果actor更新失败, 则回复给aisrv, 不进行该次的model_version更新给aisrv, 则此轮的aisrv会按照旧的model文件版本号发送样本到learner
            '''
            self.learner_send_and_recv_actor_model_version_request_and_response(send_data)

        # 无论is_train_success正确与否, 这里都需要learner和aisrv同步信息
        self.learner_send_and_recv_aisrv_model_version_request_and_response(send_data)
        
        self.logger.info(f'train process learner on_policy complete success')

    # learner朝aisrv发送model_version请求和收取响应
    def learner_send_and_recv_aisrv_model_version_request_and_response(self, send_data):
        if not send_data:
            return False
        
        for aisrv_ip, zmq_client in self.aisrv_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.logger.info(f'train process learner send model_version sync request to aisrv: {aisrv_ip}, model_version: {self.current_sync_model_version_from_learner}')
        
        # learner等待aisrv获取最新model文件版本号完毕通知
        learner_recv_all_aisrv_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            if self.recv_model_sync_response(self.aisrv_zmq_client_map):
                learner_recv_all_aisrv_success = True
                break

        if learner_recv_all_aisrv_success:
            self.on_policy_learner_recv_aisrv_success_count += 1
            self.logger.info(f'train process learner recv all the aisrv newest model sync resp')
            return True
        
        else:
            self.logger.error(f'train process learner recv not all the aisrv newest model sync resp')

            # 增加告警和容灾
            self.on_policy_learner_recv_aisrv_error_count += 1
            return False

    # learner朝actor发送model_version请求和收取响应
    def learner_send_and_recv_actor_model_version_request_and_response(self, send_data):
        if not send_data:
            return False
        
        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.logger.info(f'train process learner send model_version sync request to actor: {actor_ip}, model_version: {self.current_sync_model_version_from_learner}')

        # learner等待actor确认加载model文件完成通知, 错误情况接入监控告警
        learner_recv_all_actor_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            if self.recv_model_sync_response(self.actor_zmq_client_map):
                learner_recv_all_actor_success = True
                break

        if learner_recv_all_actor_success:
            self.logger.info(f'train process learner recv all the actor newest model sync resp')
            self.on_policy_learner_recv_actor_success_cnt += 1
            return True
        
        else:
            self.logger.error(f'train process learner recv not all the actor newest model sync resp')

            # 增加告警和容灾
            self.on_policy_learner_recv_actor_error_cnt += 1
            return False
    
    # learner推送model文件到modelpool去, 加上重试机制
    def learner_push_model_to_modelpool(self):
        all_push_model_success = False
        retry_count = 0

        while not all_push_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            push_model_success = self.model_file_sync_wraper.push_checkpoint_to_model_pool(self.logger)
            if not push_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_push_model_success = True
                self.logger.info(f'train learner learner_push_model_to_modelpool success')
                break
            
            retry_count += 1
        
        return all_push_model_success

    '''
    获取发出去的model_version同步请求的响应
    1. learner <--> aisrv
    2. learner <--> actor
    '''
    def recv_model_sync_response(self, zmq_client_map):
        if not zmq_client_map:
            return True

        # learner等待actor确认加载model文件完成通知, 错误情况接入监控告警
        success_cnt = 0
        retry_count = 0
        # aisrv/actor会返回结果, 但是结果里有正确和错误的区分, 故采用下面2个变量实现
        response_success_ip = {}
        model_version_change_ip = {}

        '''
        重试时间即等于retry_count * CONFIG.idle_sleep_second
        1. actor是在主循环里加载, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2. aisrv的超时时间设置如下:
        2.1 如果不是按照单局或者单帧的, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2.2 如果是按照单局或者单帧的, 采用的值需要大于2 * CONFIG.on_policy_timeout_seconds
        '''
        while success_cnt != len(zmq_client_map) and retry_count < int(CONFIG.on_policy_error_retry_count):
            for ip, zmq_client in zmq_client_map.items():
                # 如果已经成功的不需要重复获取响应
                if ip not in response_success_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            if recv_data[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE:
                                response_success_ip[ip] = ip

                                # 每个aisrv/actor明确返回model_version修改结果
                                if recv_data[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE]:
                                    model_version_change_ip[ip] = ip
                                    success_cnt += 1
                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)
            
            retry_count += 1
        
        if success_cnt != len(zmq_client_map):
            keys1 = set(model_version_change_ip.keys())
            keys2 = set(zmq_client_map.keys())

            self.logger.error(f'train process learner model sync not recv resp or recv error resp ips: {keys2-keys1}')
            return False

        return True

    '''
    learner --> actor的model文件同步, 目前采用的是model pool, 后期考虑优化, 当前的actor的local step, 同步learner上的global_step
    '''
    def model_file_sync(self):

        self.logger.debug(f'train process after model file sync, current global step is {self.model_wrapper.get_global_step()}')

    # 监控项置位
    def train_stat_reset(self):
        self.batch_train_cost_time_ms = 0
        self.sample_send_and_consume_ratio = 0

    '''
    这里增加train的统计项
    '''
    def train_stat(self):

        '''
        样本的生成速度: reverb的间隔时间里insert的样本数,注意插入次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗速度: 训练的次数 * batch_size, 注意训练的次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗/生产比 = 样本消耗的速度 / 样本的生产的速度
        '''
        train_count = self.model_wrapper.train_stat
        reverb_insert_count = self.replay_buffer_wrapper.get_insert_stats()

        reverb_current_size = self.replay_buffer_wrapper.get_current_size()

        self.sample_product_rate = (reverb_insert_count - self.last_reverb_insert_count)
        self.sample_consume_rate = (train_count - self.last_train_count) * int(CONFIG.train_batch_size)

        if self.sample_product_rate == 0:
            self.sample_send_and_consume_ratio = 0
        else:
            self.sample_send_and_consume_ratio =  self.sample_consume_rate  / self.sample_product_rate 
        
        self.last_train_count = train_count
        self.last_reverb_insert_count = reverb_insert_count

        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_REVERB_READY_SIZE: reverb_current_size,
                KaiwuDRLDefine.MONITOR_TRAIN_SUCCES_CNT: self.model_wrapper.train_stat,
                KaiwuDRLDefine.MONITOR_TRAIN_GLOBAL_STEP: self.model_wrapper.get_global_step(),
                KaiwuDRLDefine.MONITOR_BATCH_TRAIN_COST_TIME_MS: self.batch_train_cost_time_ms,
                KaiwuDRLDefine.LEARNER_TCP_AISRV : actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.SAMPLE_SEND_AND_CONSUME_RATIO : self.sample_send_and_consume_ratio,
                KaiwuDRLDefine.SAMPLE_PRODUCT_RATE : self.sample_product_rate,
                KaiwuDRLDefine.SAMPLE_CONSUME_RATE : self.sample_consume_rate,
            }

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data[KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_ERROR_CNT] = self.on_policy_push_to_modelpool_error_count
                monitor_data[KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_SUCCESS_CNT] = self.on_policy_push_to_modelpool_success_count
                monitor_data[KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_ERROR_CNT] = self.on_policy_learner_recv_aisrv_error_count
                monitor_data[KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_SUCCESS_CNT] = self.on_policy_learner_recv_aisrv_success_count
                monitor_data[KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_ERROR_CNT] = self.on_policy_learner_recv_actor_error_cnt
                monitor_data[KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_SUCCESS_CNT] = self.on_policy_learner_recv_actor_success_cnt

            # 按照业务数据返回的map格式直接赋值, 然后去普罗米修斯监控上设置下展示字段即可
            for key, value in self.app_monitor_data.items():
                monitor_data[key] = float(value)

            self.monitor_proxy.put_data(monitor_data)

        # 指标复原, 计算的是周期性的上报指标
        self.train_stat_reset()

        self.logger.info(f'train process now input ready size is {self.replay_buffer_wrapper.get_current_size()}')
        self.logger.info(f'train process now train count is {self.model_wrapper.train_stat}, tensorflow global step is {self.model_wrapper.get_global_step()}')

    '''
    框架运行前创建必要的文件目录
    '''
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)
        make_single_dir(CONFIG.restore_dir)
        make_single_dir(CONFIG.summary_dir)
        make_single_dir(CONFIG.ckpt_dir)
        make_single_dir(CONFIG.pb_model_dir)
        make_single_dir(f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}')
        
        # 按照需要创造旁路文件
        if int(CONFIG.use_bypass):
            make_single_dir(CONFIG.bypass_dir)
    
    def start_learner_process_by_type(self):

        # 按照需要引入ModelFileSave, 注意此处是多个learner都需要从COS获取模型文件
        self.model_file_saver = ModelFileSave()
        self.model_file_saver.start_actor_process_by_type(self.logger)
    
    '''
    预加载模式功能是指将预先训练好的baseline文件加载到KaiwuDRL里, 只是learner需要处理, actor会通过learner<-->actor之间的model文件同步在某个时间阈值后替换
    1. tensorflow, 该框架自动支持
    2. pytorch, 需要手工调用下函数

    使用方法:
    1. 需要在/data/ckpt/app_algo下放置需要设置的model文件
    2. 修改/data/ckpt/app_algo下checkpoint文件内容, 指向1中的model文件
    '''
    def preload_model_file(self):
        if not int(CONFIG.preload_model):
            return

        if KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework or KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework or KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            self.logger.info(f'train tensorflow preload, not need to call function')

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            if not CONFIG.preload_model_file or not os.path.exists(CONFIG.preload_model_file):
                self.logger.error(f'train pytorch preload, but preload_model_file is empty or not exist')
                return
            
            # 获取上传的model文件的ID, 如果找不到则从0开始, 否则从上传的ID开始, 类似的文件名字为/data/projects/kaiwu-fwk/model.ckpt-2200.pkl 
            match = re.search(r"\d+", CONFIG.preload_model_file)
            id = 0
            if match:
                id = int(match.group())
            
            self.model_wrapper.preload_model_file(CONFIG.preload_model_file, id)
            self.logger.info(f'train pytorch preload success, preload_model_file is {CONFIG.preload_model_file}, id is {id}')
        else:
            self.logger.error(f'train preload just not support {CONFIG.use_which_deep_learning_framework}, support list is KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE, KaiwuDRLDefine.MODEL_TENSORRT, KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX, KaiwuDRLDefine.MODEL_PYTORCH')

    def before_run(self):

        self.make_dirs()

        # 支持间隔N分钟, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, 
                                    CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, self.logger)
            self.logger.info(f'train RainbowUtils {self.rainbow_utils.identity}')

            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 根据不同启动方式来进行处理
        self.start_learner_process_by_type()

        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()

        # 启动独立的进程, 负责learner与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        if int(CONFIG.use_prometheus):
            # 启动独立的进程, 负责learner与普罗米修斯交互
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()
            
            # 注册定时器任务
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.train_stat)

        # replay_buffer
        self.replay_buffer_wrapper = ReplayBufferWrapper(self.tensor_names, self.tensor_dtypes, self.tensor_shapes, self.logger)
        self.replay_buffer_wrapper.init()
        self.replay_buffer_wrapper.extra_threads()

        # model_wrapper, 由于ModelFileSyncWrapper和ModelFileSave需要判断是否是主learner才能进行下一步处理, 故提前到这里进行
        self.create_model_wrapper()

        # model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程处理, 只有主learner进程才会执行
        if self.model_wrapper.is_chief:
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.model_file_sync_wraper = ModelFileSyncWrapper()
                self.model_file_sync_wraper.init()

            elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                self.current_sync_model_version_from_learner = -1
                self.model_file_sync_wraper = ModelFileSync()
                self.model_file_sync_wraper.make_model_dirs(self.logger)

                # 由于aisrv依赖learner先启动, 故learner启动后再去周期性的获取aisrv地址并且建立TCP连接
                set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.on_policy_learner_get_and_connect_aisrv)

                set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.on_policy_learner_get_and_connect_actor)

                self.alloc_util = AllocUtils(self.logger)

                # 格式client_id->zmq_client对象
                self.actor_zmq_client_map = {}
                self.aisrv_zmq_client_map = {}

                # 下面是统计告警指标
                self.on_policy_push_to_modelpool_error_count = 0
                self.on_policy_push_to_modelpool_success_count = 0
                self.on_policy_learner_recv_aisrv_error_count = 0
                self.on_policy_learner_recv_aisrv_success_count = 0
                self.on_policy_learner_recv_actor_error_cnt = 0
                self.on_policy_learner_recv_actor_success_cnt = 0
                self.on_policy_learner_change_model_version_cnt = 0

            else:
                pass

        # model_file_saver, 用于保存模型文件到持久化设备, 比如COS, 采用单独的进程处理, 只有主learner进程才会执行
        if self.model_wrapper.is_chief:
            self.model_file_saver = ModelFileSave()
            self.model_file_saver.start()
        
        # 预先加载模型文件模式
        if int(CONFIG.preload_model):
            self.preload_model_file()

        # 如果是pytorch, 则默认第一次保存文件
        if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
            self.model_wrapper.save_param()

        '''
        统计监控指标
        1. 批处理的训练耗时
        '''
        self.batch_train_cost_time_ms = 0
        self.last_input_ready_count = 0
        self.batch_train_cost_time_ms = 0
        self.sample_send_and_consume_ratio = 0
        self.last_reverb_insert_count = 0
        self.last_train_count = 0
        self.last_input_ready_count = 0
        self.sample_product_rate = 0
        self.sample_consume_rate = 0

        # 业务算法类监控值是个map形式
        self.app_monitor_data = {}

        self.logger.info(f"train process {self.name} trainer global step {self.local_step.value} load app {CONFIG.app} algo {CONFIG.algo} model")

    '''
    learner获取actor地址并且建立TCP连接, 包括下面的操作:
    1. 获取actor地址, 分是否使用alloc服务
    2. 根据1中获取actor地址情况进行处理
    2.1 如果1中获取actor地址失败, 则下次重试
    2.2 如果1中获取actor地址成功, 则本次执行
    '''
    def on_policy_learner_get_and_connect_actor(self):

        # 如果self.actor_zmq_client_map为空则走获取actor地址流程
        if not self.actor_zmq_client_map:
            self.on_policy_learner_get_actor_address()

    '''
    on-policy场景下, learner获取actor地址
    '''
    def on_policy_learner_get_actor_address(self):
        
        '''
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务
        '''
        actor_address = [KaiwuDRLDefine.LOCAL_HOST_IP]
        if int(CONFIG.use_alloc):
            self.alloc_util.registry()
            actor_address = self.alloc_util.get_all_address_by_srv_name(KaiwuDRLDefine.SERVER_ACTOR)
            if not actor_address or not len(actor_address):
                self.logger.error(f"train get actor_address error, retry next time")
                return
            else:
                self.logger.info(f"train get actor_address success,  actor address: {actor_address}")
        else:
            self.logger.info(f"train set use_alloc False, so actor use {KaiwuDRLDefine.LOCAL_HOST_IP}")
        
        for address in actor_address:
            client_id = get_uuid()
            actor_ip = address.split(':')[0]
            zmq_client = ZmqClient(str(client_id), actor_ip, int(CONFIG.zmq_server_port) + 100)
            zmq_client.connect()
            self.actor_zmq_client_map[f'{actor_ip}:{int(CONFIG.zmq_server_port) + 100}'] = zmq_client

    '''
    on-policy场景下, learner与actor地址建立连接, 周期性的发送/接收心跳保活请求/响应
    '''
    def on_policy_learner_connect_to_actor(self):
        if not self.actor_zmq_client_map:
            return
        
        learner_send_to_actor_heartbeat_success_count = 0
        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            send_data = {
                            KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
                            KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST
                        }
            zmq_client.send(send_data, binary=False)
            self.logger.debug(f"train send heartbeat request to actor: {actor_ip} success")

            # 同步等待心跳响应回包
            retry_count = 0
            while retry_count < int(CONFIG.on_policy_error_retry_count):
                try:
                    recv_data = zmq_client.recv(block=False, binary=False)
                    if recv_data:
                        if recv_data[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE:
                            self.logger.debug(f"train recv heartbeat response to actor: {actor_ip} success")
                            learner_send_to_actor_heartbeat_success_count += 1
                            break
                except Exception as e:
                    # 减少CPU争用
                    time.sleep(CONFIG.idle_sleep_second)
                
                retry_count += 1
        
        # 以为心跳的请求频率比较高, 打印日志比较耗时, 故采用debug日志
        if learner_send_to_actor_heartbeat_success_count == len(self.actor_zmq_client_map):
            self.logger.debug(f"train learner recv all actor heartbeat response success, count: {learner_send_to_actor_heartbeat_success_count}")

        else:
            # 由于无法收到actor的请求, 那么此时不确定actor的情况是怎么样, 故清空self.actor_zmq_client_map, 重新拉取看下效果
            self.logger.error(f"train learner not recv all actor heartbeat response, retry next time, learner_send_to_actor_heartbeat_success_count {learner_send_to_actor_heartbeat_success_count} != len(actor_zmq_client_map) {len(self.actor_zmq_client_map)}")

            self.actor_zmq_client_map.clear()
            self.on_policy_learner_get_actor_address()

    '''
    learner获取aisrv地址并且建立TCP连接, 包括下面的操作:
    1. 获取aisrv地址, 分是否使用alloc服务
    2. 根据1中获取aisrv地址情况进行处理
    2.1 如果1中获取aisrv地址失败, 则下次重试
    2.2 如果1中获取aisrv地址成功, 则本次执行
    '''
    def on_policy_learner_get_and_connect_aisrv(self):

        # 如果self.aisrv_zmq_client_map为空则走获取aisrv地址流程
        if not self.aisrv_zmq_client_map:
            self.on_policy_learner_get_aisrv_address()

    '''
    on-policy场景下, learner从alloc服务获取aisrv的地址
    '''
    def on_policy_learner_get_aisrv_address(self):

        '''
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务
        '''
        aisrv_address = [KaiwuDRLDefine.LOCAL_HOST_IP]
        if int(CONFIG.use_alloc):
            self.alloc_util.registry()
            # on-policy情况下learner需要启动与aisrv的通信, 采用在aisrv 8000端口号 + 100的端口上监听, learner为client, aisrv为server
            aisrv_address = self.alloc_util.get_all_address_by_srv_name(KaiwuDRLDefine.SERVER_AISRV)
            if not aisrv_address or not len(aisrv_address):
                self.logger.error(f"train get aisrv_address error, retry next time")
                return
            else:
                self.logger.info(f"train get aisrv_address success, aisrv address: {aisrv_address}")
        else:
            self.logger.info(f"train set use_alloc False, so aisrv use {KaiwuDRLDefine.LOCAL_HOST_IP}")

        for address in aisrv_address:
            client_id = get_uuid()
            aisrv_ip = address.split(':')[0]
            zmq_client = ZmqClient(str(client_id), aisrv_ip, int(CONFIG.aisrv_server_port) + 100)
            zmq_client.connect()

            self.aisrv_zmq_client_map[f'{aisrv_ip}:{int(CONFIG.aisrv_server_port) + 100}'] = zmq_client

    '''
    训练的规则:
    1. 当reverb设置最大的size, 采用FIFO模式
    2. 当满足batch_size即开始训练, 对reverb不做主动清空操作, 从reverb里拿取的数据是随机的, 这样增加了训练次数, 新的数据进来采用FIFO去替换掉旧的
    '''
    def train(self):
        reverb_insert_count = self.replay_buffer_wrapper.get_insert_stats()
        current_size = self.replay_buffer_wrapper._replay_buffer.total_size(self.replay_buffer_wrapper._reverb_client)
        
        # 标志本次是否真实的train
        is_train_success = False

        '''
        这里需要区分下:
        1. 如果是on-policy, 必须等current_size大于batch_size才能进入到self.train_detail逻辑, 否则因为aisrv在等learner的on-policy响应, learner在等aisrv产生样本, 就出现死锁
        2. 如何是off-policy, 不需要current_size大于batch_size, 当learner在读取reverb样本时, 阻塞了aisrv也能给learner发送样本
        '''
        # 步骤2, 从reverb server获取样本数据
        '''
        learner满足训练条件的情况:
        1. off-policy
        1.1 积累一定数量
        1.2 大于等于样本消耗比
        1.3 大于batch_size
        2. on-policy
        2.1 大于batch_size
        '''
        # if self.input_ready():
        condition = False
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            condition = (current_size > int(CONFIG.replay_buffer_capacity)//int(CONFIG.preload_ratio)) and \
            (reverb_insert_count - self.last_input_ready_count) > (int(CONFIG.train_batch_size) / int(CONFIG.production_consume_ratio))
        
        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            condition = current_size >= int(CONFIG.train_batch_size)
 
        else:
            pass

        if condition:
            # 步骤3, 训练
            self.train_detail()

            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.last_input_ready_count = reverb_insert_count
            is_train_success = True
        
        return is_train_success

    '''
    learner的单次流程如下:
    1. 执行定时器操作
    2. 执行训练步骤
    3. on-policy情况下, 执行从aisrv开始的流程
    '''
    def run_once(self):

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        '''
        步骤2, 执行训练, 主要是下面的情况:
        1. 如果是on-policy
        1.1 如果是按照learner角度多帧的, 直接训练
        1.2 如果是按照aisrv角度单帧/单局的, 不要调用self.train训练, 而是采用self.learner_on_policy_process_by_aisrv推动
        1.3 其他的情况, 后期扩展, 直接训练
        2. 如果是off-policy, 直接训练
        3. 其他的情况, 后期扩展, 直接训练
        '''
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_TIME_INTERVAL:
                self.train()
            elif CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                pass
            else:
                self.train()

        elif CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            self.train()
            
        else:
            self.train()

        # 步骤3, 收集从aisrv来的发起on-policy请求
        self.learner_on_policy_process_by_aisrv()

        # Model文件保存, 同步已经采用单个进程方式进行

    def input_tensors(self):
        return self.replay_buffer_wrapper.input_tensors()

    '''
    learner满足训练条件的情况:
    1. off-policy
    1.1 积累一定数量
    1.2 大于等于样本消耗比
    1.3 大于batch_size
    2. on-policy
    2.1 大于batch_size
    '''
    def input_ready(self):
        return self.replay_buffer_wrapper.input_ready(None)

    def loop(self):
        self.before_run()

        while not self.model_wrapper.should_stop():
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0
                
            except Exception as e:
                self.logger.error(f"train process failed to run {self.name} trainer. exit. Error is: {e}, traceback.print_exc() is {traceback.format_exc()}")
                break
        
        self.model_wrapper.close()
        self.logger.info('train self.server.stop success')

        # 非on-policy的才需要主动关闭self.model_file_sync_wraper
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
            self.model_file_saver.stop()
            self.logger.info('train self.model_wrapper.close success')

            self.model_file_sync_wraper.stop()
            self.logger.info('train self.model_file_sync_wraper.stop success')
