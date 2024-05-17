#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv.py
# @brief
# @author kaiwu
# @date 2022-04-25


import os
import schedule
import threading
import socketserver as ss
import socket
import multiprocessing
import traceback
import datetime
import flatbuffers
import time
import yaml
import collections
import copy
from framework.interface.exception import ClientQuitException
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.common_func import TimeIt, is_list_eq, list_diff, set_schedule_event, compress_data, decompress_data, actor_learner_aisrv_count, get_host_ip, get_uuid
from framework.common.ipc.connection import Connection
from framework.common.utils.common_func import Context
from framework.common.config.app_conf import AppConf
from framework.server.aisrv.flatbuffer.kaiwu_msg import Request, ReqMsg
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from framework.server.aisrv.kaiwu_rl_helper import KaiWuRLHelper
from framework.server.aisrv.msg_buff import MsgBuff
from framework.common.utils.slots import Slots
from framework.common.alloc.alloc_utils import AllocUtils
from framework.common.alloc.alloc_proxy import AllocProxy
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.rainbow_utils import RainbowUtils
import lz4.block
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.ipc.zmq_util import ZmqClient, ZmqServer

'''
因为server_activate在server_bind后开始运行, 故需要早于server_activate调用, 设置socketserver的网络参数
'''
def socketserver_setoption():
    ss.TCPServer.allow_reuse_address = True
    # ss.ForkingTCPServer.allow_reuse_address = True

socketserver_setoption()

# ForkingTCPServers是继承的TCPServer
class AiSrv(ss.ForkingTCPServer):

    def __init__(self, server_address, RequestHandlerClass):
        # 便于Aisrv和AiHandler进程之间通信, 因为只是需要model_version版本号, 采用Value数据结构
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.active_handlers_model_version = multiprocessing.Value('i', -1)
            self.active_handlers_model_version_lock = multiprocessing.RLock()

        if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:

            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前处于已经被pause的kaiwu_rl_helper线程数量, 采用Value数据结构
            self.kaiwu_rl_helper_pause_count = multiprocessing.Value('i', 0)
            self.kaiwu_rl_helper_pause_count_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前处于已经被continue的kaiwu_rl_helper线程数量, 采用Value数据结构
            self.kaiwu_rl_helper_continue_count = multiprocessing.Value('i', 0)
            self.kaiwu_rl_helper_continue_count_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 表示可以继续暂停掉的线程, 布尔类型, 采用Value数据结构
            self.kaiwu_rl_helper_continue_value = multiprocessing.Value('b', False)
            self.kaiwu_rl_helper_continue_value_lock = multiprocessing.RLock()

            '''
            因为单局/单帧会多次continue和pause, 故单个handler需要单独控制的, 故采用Array数据结构, 不能采用Value数据结构
            '''
            # 便于Aisrv主进程和AiHandler进程之间通信, 用于多次执行continue操作, 布尔数组类型, 采用Array数据结构
            self.have_set_kaiwu_rl_helper_continue = multiprocessing.Array('b', [False] * CONFIG.max_tcp_count)
            self.have_set_kaiwu_rl_helper_continue_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 用于多次执行pause操作, 布尔数组类型, 采用Array数据结构
            self.have_set_kaiwu_rl_helper_pause = multiprocessing.Array('b', [False] * CONFIG.max_tcp_count)
            self.have_set_kaiwu_rl_helper_pause_lock = multiprocessing.RLock()

            # 便于Aisrv主进程和AiHandler进程之间通信, 记录当前活着的handler数量, 采用Value数据结构
            self.active_handlers_alive_count = multiprocessing.Value('i', 0)
            self.active_handlers_alive_count_lock = multiprocessing.RLock()

        super().__init__(server_address, RequestHandlerClass)

    def server_activate(self) -> None:
        super().server_activate()

        ss.ForkingMixIn.max_children = CONFIG.max_tcp_count

        # 设置日志Log配置
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/aisrv_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'AiSrv')
        self.logger.info(f'AiSrv is start at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, pid is {pid}, run_mode is {CONFIG.run_mode}, self_play is {CONFIG.self_play}')

        # 设置Context
        self.simu_ctx = Context()

        # aisrv handler进程使用
        self.simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))

        # aisrv进程启动时, 从七彩石获取配置
        if int(CONFIG.use_rainbow):
            self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, 
                                CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, self.logger)
        
            self.logger.info(f'AiSrv RainbowUtils {self.rainbow_utils.identity}')

            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
            self.rainbow_activate_single_process(CONFIG.svr_name)

        # aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址
        if int(CONFIG.use_alloc):
            self.get_actor_learner_ip_from_alloc()

        ''' 
        实例配置如下
        {
            "hero": {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "rl_helper": "app.gorge_walk.environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
                "policies": {
                "train_one": {
                    "policy_builder" : "framework.server.aisrv.async_policy.AsyncBuilder",
                    "algo": "ppo",
                    "state": "app.gym.gym_proto.GymState",
                    "action": "app.gym.gym_proto.GymAction",
                    "reward": "app.gym.gym_proto.GymReward",
                    "actor_network": "app.gym.gym_network.GymDeepNetwork",
                    "learner_network": "app.gym.gym_network.GymDeepNetwork",
                    "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                }
            }
        }
        '''

        try:
            policies_builder = {}
            policies_conf = AppConf[CONFIG.app].policies
            for policy_name, policy_conf in policies_conf.items():
                policies_builder[policy_name] = policy_conf.policy_builder(policy_name, self.simu_ctx)
            
            self.simu_ctx.policies_builder = policies_builder

            self.simu_ctx.kaiwu_rl_helper = AppConf[CONFIG.app].rl_helper

        except Exception as e:
            self.logger.exception(
                f'AiSrv server start exception: {str(e)}, traceback.print_exc() is {traceback.format_exc()}')
            return
        
        # 启动独立的进程, 负责aisrv与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()
        
        # 启动独立的线程, 负责aisrv与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        # 在on-policy的情况下需要启动zmq server, aisrv为server, learner为client
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.zmq_server = ZmqServer(CONFIG.aisrv_ip_address, int(CONFIG.aisrv_server_port) + 100)
            self.zmq_server.bind()
            self.logger.info(f'AiSrv on-policy aisrv bind at {int(CONFIG.aisrv_server_port) + 100} for learner')

            # 统计信息
            self.all_kaiwu_rl_helper_pause_error_count = 0
            self.all_kaiwu_rl_helper_pause_success_count = 0
            self.all_kaiwu_rl_helper_continue_error_count = 0
            self.all_kaiwu_rl_helper_continue_success_count = 0
            self.aisrv_change_model_version_error_count = 0
            self.aisrv_change_model_version_success_count = 0

            # 记录learner所在的client_id
            self.client_id = None

            # learner需要朝aisrv发送了心跳请求后才能得到client_id进行通信
            self.learner_have_send_heartbeat_request_success = False

            # 设置统计监控信息
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.aisrv_main_process_stat)

            # 如果是aisrv按照单帧或者单局的进行, 则表示本次需要推动learner的on-policy流程, 执行完成后就设置为False, 进行下一轮
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                self.aisrv_ask_learner_to_execute_on_policy = False

    '''
    业务的上报, aisrv主线程
    '''
    def aisrv_main_process_stat(self):
        if int(CONFIG.use_prometheus):
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data = {}
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_ERROR_COUNT] = self.all_kaiwu_rl_helper_pause_error_count
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_SUCCESS_COUNT] = self.all_kaiwu_rl_helper_pause_success_count
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_ERROR_COUNT] = self.all_kaiwu_rl_helper_continue_error_count
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_SUCCESS_COUNT] = self.all_kaiwu_rl_helper_continue_success_count
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_ERROR_COUNT] = self.aisrv_change_model_version_error_count
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_SUCCESS_COUNT] = self.aisrv_change_model_version_success_count
                
                self.monitor_proxy.put_data(monitor_data)

    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            self.logger.error(f'AiSrv read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(f'AiSrv read_from_rainbow failed, data is None or data len is 0')
            return
            
        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        
        CONFIG.save_to_file(process_name, to_change_key_values)
        self.logger.info(f"AiSrv {process_name} CONFIG save_to_file success")

    '''
    增加aisrv从alloc获取IP地址的逻辑, 为了和以前从配置文件加载的方式结合, 采用操作步骤如下:
    1. 每隔CONFIG.alloc_process_per_seconds拉取, 最大CONFIG.socket_retry_times次后报错, 当返回有具体的数据则跳出循环
    2. 针对返回的actor和learner地址, 修改内存和配置文件里的值
    '''
    def get_actor_learner_ip_from_alloc(self):

        # alloc 工具类, aisrv上与alloc交互操作
        self.alloc_util = AllocUtils(self.logger)

        # 需要先注册本地aisrv地址后, 再拉取actor, learner地址
        code, msg = self.alloc_util.registry()
        if code:
            self.logger.info(f"AiSrv alloc interact registry success")
        else:
            self.logger.error(f"AiSrv alloc interact registry fail, will retry next time, error_code is {msg}")
            return

        # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times:
            if not int(CONFIG.self_play):
                actor_address, learner_address, _, _ = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, None)
                if not actor_address or not learner_address:
                    time.sleep(int(CONFIG.socket_timeout))
                    retry_num += 1
                else:
                    break
            else:
                # 对于self-play模式, self_play_set下的learner不是强要求的
                self_play_actor_address, self_play_old_actor_address, self_play_learner_address, self_play_old_learner_address = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)

                if not self_play_actor_address or not self_play_learner_address or not self_play_old_actor_address:
                    time.sleep(int(CONFIG.socket_timeout))
                    retry_num += 1
                else:
                    break

        # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
        if retry_num >= CONFIG.socket_retry_times:
            self.logger.error(
                f'AiSrv server get actor and learner address retry times more than {CONFIG.socket_retry_times}, will start with configure file')
            return
        
        # 修改配置文件
        if not int(CONFIG.self_play):
            self.change_configure_content(actor_address, learner_address, None, None, None, None)
        else:
            self.change_configure_content(None, None, self_play_actor_address, self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address)


    '''
    修改conf/system/aisrv_system.json里的配置项目, 如下:
    1. actor_addrs
    2. actor_proxy_num
    3. learner_addrs
    4. learner_proxy_num
    5. self_play_actor_proxy_num
    6. self_play_old_actor_proxy_num
    7. self_play_learner_proxy_num
    8. self_play_old_learner_proxy_num
    '''
    def change_configure_content(self, actor_addrs, learner_addrs, self_play_actor_address, self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address):
        
        # 写回配置文件内容
        to_change_key_values = {}

        # 将当前的配置文件的内容读成字典, 内存修改后, 再写回, 如果解析出错, 则提前报错返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)
        except Exception as e:
            self.logger.error(f'alloc_proxy get actor and learner address from conf failed, error is {str(e)}', g_not_server_label)

            return
        
        '''
        处理实例如下:
        actor_addrs = {"train_one": ["127.0.0.1:8001"], "train_two": ["127.0.0.1:8002"]}
        learner_addrs = {"train_one": ["127.0.0.1:9000"], "train_two": ["127.0.0.1:9001"]}
        '''
        
        if not int(CONFIG.self_play):
            if not actor_addrs and not learner_addrs:
                return
            
            # 如果actor_addrs不空则处理, 否则跳过
            if actor_addrs:
                actor_proxy_num = len(actor_addrs)
                old_actor_address_map[CONFIG.policy_name] = actor_addrs
                to_change_key_values['actor_proxy_num'] = actor_proxy_num
                to_change_key_values['actor_addrs'] = old_actor_address_map

            # 如果learner_addrs不空则处理, 否则跳过
            if learner_addrs:
                learner_proxy_num = len(learner_addrs)
                old_learner_address_map[CONFIG.policy_name] = learner_addrs
                to_change_key_values['learner_proxy_num'] = learner_proxy_num
                to_change_key_values['learner_addrs'] = old_learner_address_map
            
            # 修改配置文件内容落地
            if actor_addrs or learner_addrs:
                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiSrv {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

        else:
            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return
            
            if self_play_actor_address:
                self_play_actor_proxy_num = len(self_play_actor_address)
                old_actor_address_map[CONFIG.self_play_policy] = self_play_actor_address
                to_change_key_values['self_play_actor_proxy_num'] = self_play_actor_proxy_num
        
            if self_play_old_actor_address:
                self_play_old_actor_proxy_num = len(self_play_old_actor_address)
                old_actor_address_map[CONFIG.self_play_old_policy] =  self_play_old_actor_address
                to_change_key_values['self_play_old_actor_proxy_num'] = self_play_old_actor_proxy_num
            
            to_change_key_values['actor_addrs'] = old_actor_address_map

            if self_play_learner_address:
                self_play_learner_proxy_num = len(self_play_learner_address)
                old_learner_address_map[CONFIG.self_play_policy] =  self_play_learner_address
                to_change_key_values['self_play_learner_proxy_num'] = self_play_learner_proxy_num
            
            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(self_play_old_learner_address)
                old_learner_address_map[CONFIG.self_play_old_policy] =  self_play_old_learner_address
                to_change_key_values['self_play_old_learner_proxy_num'] = self_play_old_learner_proxy_num
            
            to_change_key_values['learner_addrs'] = old_learner_address_map

            # 修改配置文件内容落地
            if self_play_actor_address or self_play_learner_address or self_play_old_actor_address or self_play_old_learner_address:
                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiSrv {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

    # 在kaiwu_rl_helper_pause_count的大小
    def get_kaiwu_rl_helper_pause_count(self):
        kaiwu_rl_helper_pause_count = 0
        with self.kaiwu_rl_helper_pause_count_lock:
            kaiwu_rl_helper_pause_count = self.kaiwu_rl_helper_pause_count.value
        
        return kaiwu_rl_helper_pause_count
    
    # 获取self.kaiwu_rl_helper_continue_value的值, 便于handler进程continue kaiwu_rl_helper线程
    def get_kaiwu_rl_helper_continue_value(self):
        kaiwu_rl_helper_continue_value = False
        with self.kaiwu_rl_helper_continue_value_lock:
            kaiwu_rl_helper_continue_value = self.kaiwu_rl_helper_continue_value.value

        return kaiwu_rl_helper_continue_value
    
    # 获取self.have_set_kaiwu_rl_helper_continue的值, 便于单个aisrv handler处理
    def get_have_set_kaiwu_rl_helper_continue(self, slot_id):
        have_set_kaiwu_rl_helper_continue = False
        with self.have_set_kaiwu_rl_helper_continue_lock:
            have_set_kaiwu_rl_helper_continue = self.have_set_kaiwu_rl_helper_continue[slot_id]
        
        return have_set_kaiwu_rl_helper_continue

    # 设置self.have_set_kaiwu_rl_helper_continue的值, 便于单个aisrv handler处理
    def set_have_set_kaiwu_rl_helper_continue(self, slot_id, value):
        with self.have_set_kaiwu_rl_helper_continue_lock:
            self.have_set_kaiwu_rl_helper_continue[slot_id] = value

    # 获取self.have_set_kaiwu_rl_helper_pause的值, 便于单个aisrv handler处理
    def get_have_set_kaiwu_rl_helper_pause(self, slot_id):
        have_set_kaiwu_rl_helper_pause = False
        with self.have_set_kaiwu_rl_helper_pause_lock:
            have_set_kaiwu_rl_helper_pause = self.have_set_kaiwu_rl_helper_pause[slot_id]
        
        return have_set_kaiwu_rl_helper_pause
    
    # 设置self.have_set_kaiwu_rl_helper_pause的值, 便于单个aisrv handler处理
    def set_have_set_kaiwu_rl_helper_pause(self, slot_id, value):
        with self.have_set_kaiwu_rl_helper_pause_lock:
            self.have_set_kaiwu_rl_helper_pause[slot_id] = value
    
    # 在kaiwu_rl_helper_continue_count的大小
    def get_kaiwu_rl_helper_continue_count(self):
        kaiwu_rl_helper_continue_count = 0
        with self.kaiwu_rl_helper_continue_count_lock:
            kaiwu_rl_helper_continue_count = self.kaiwu_rl_helper_continue_count.value

        return kaiwu_rl_helper_continue_count
    
    # 获取self.active_handlers_model_version的值, 便于handler进程model_version更改
    def get_active_handlers_model_version_value(self):
        active_handlers_model_version_value = 0
        with self.active_handlers_model_version_lock:
            active_handlers_model_version_value = self.active_handlers_model_version.value

        return active_handlers_model_version_value

    # 获取self.active_handlers_alive_count的值, 即活着的handler进程
    def get_active_handlers_alive_count(self):
        active_handlers_alive_count_value = 0
        with self.active_handlers_alive_count_lock:
            active_handlers_alive_count_value = self.active_handlers_alive_count.value

        return active_handlers_alive_count_value

    # 设置self.active_handlers_alive_count的值, 主要是单个handler进程退出或者增加使用
    def set_active_handlers_alive_count_value(self, value):
        with self.active_handlers_alive_count_lock:
            self.active_handlers_alive_count.value += value

    # on-policy情况下按照单帧/单局的逻辑
    def on_policy_by_episode(self):
        self.on_policy_by_episode_detail()

    '''
    如果是on-policy时, 则流程如下:
    1. 获取所有的kaiwu_rl_helper的暂停信息, 判断依据:self.kaiwu_rl_helper_pause_count的大小等于active_handlers_alive_count大小
    2. 如果1满足, 则朝learner传递开始走on-policy的信息
    3. 如果2满足, 则等待到超时时间后即告警, 后期做容灾处理
    '''
    def on_policy_by_episode_detail(self):
        # 如果learner还没有发送心跳请求, 则learner与aisrv之间无法进行通信, 故不走该逻辑
        if not self.learner_have_send_heartbeat_request_success:
            return
        
        # 如果当前的self.aisrv_ask_learner_to_execute_on_policy已经为True, 说明还没有走完aisrv让learner执行on-policy流程, 故本次就提前返回
        if self.aisrv_ask_learner_to_execute_on_policy:
            return
        
        '''
        在超时时间内, 只要aisrv收到了1个kaiwu_rl_helper发送的on-policy流程开始的请求, 那么就按照最大超时时间收集所有的kaiwu_rl_helper的请求
        并且只是增加1次, 否则无法跳出循环
        '''
        end_time = time.time() + CONFIG.on_policy_timeout_seconds
        any_kaiwu_rl_helper_pause_success = False
        update_end_time = False
        kaiwu_rl_helper_pause_count = 0
        active_handle_count = 0
        while time.time() < end_time:
            # 新增/减少handler, 新增/减少kaiwu_rl_helper线程是动态的
            kaiwu_rl_helper_pause_count = self.get_kaiwu_rl_helper_pause_count()
            active_handle_count = self.get_active_handlers_alive_count()
            if kaiwu_rl_helper_pause_count > 0:
                any_kaiwu_rl_helper_pause_success = True
            else:
                # 减少CPU争用
                time.sleep(CONFIG.idle_sleep_second)

            # 达到一定比例即跳出循环
            if any_kaiwu_rl_helper_pause_success and (kaiwu_rl_helper_pause_count / active_handle_count >= CONFIG.on_policy_quantity_ratio):
                break

            if any_kaiwu_rl_helper_pause_success and not update_end_time:
                end_time = time.time() + CONFIG.on_policy_timeout_seconds
                update_end_time = True
        
        # 超时后即告警处理
        if any_kaiwu_rl_helper_pause_success:
            if kaiwu_rl_helper_pause_count / active_handle_count < CONFIG.on_policy_quantity_ratio:

                # 设置本次需要learner执行on-policy逻辑的标志位
                self.aisrv_ask_learner_to_execute_on_policy = False

                self.all_kaiwu_rl_helper_pause_error_count += 1
                self.logger.error(f"AiSrv on_policy_by_episode kaiwu_rl_helper_pause_count {kaiwu_rl_helper_pause_count} / active_handle_count {active_handle_count} < {CONFIG.on_policy_quantity_ratio}, so aisrv not ask learner to execute on-policy process this time, try next time")
                return

            else:
                # 设置本次需要learner执行on-policy逻辑的标志位
                self.aisrv_ask_learner_to_execute_on_policy = True
                self.all_kaiwu_rl_helper_pause_success_count += 1
                self.logger.info(f"AiSrv on_policy_by_episode kaiwu_rl_helper_pause_count {kaiwu_rl_helper_pause_count} / active_handle_count {active_handle_count} >= {CONFIG.on_policy_quantity_ratio}, so self.aisrv_ask_learner_to_execute_on_policy is True")
        else:
            # 设置本次需要learner执行on-policy逻辑的标志位
            self.aisrv_ask_learner_to_execute_on_policy = False
    
    # on-policy情况下, aisrv与learner之间的通信
    def on_policy_aisrv_communication_with_learner(self):
        # 在aisrv发起的on-policy流程主要是针对每局或者每帧, 此时也需要收集从learner发给aisrv的model_version修改信息
        try:
            self.client_id, message = self.zmq_server.recv(block=False, binary=False)
            if message:
                ''' 
                需要区分消息类型
                1. 如果是learner的heartbeat请求, 主要是为获取client_id, 增加是否需要开启on-policy流程, 需要回包给learner
                2. 如果是learner的model_version请求, 主要是更新model_version, 需要回包给learner
                3. 待扩展
                '''
                if message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:
                    message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] = KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE

                    # 便于以后扩展, 采用map形式
                    send_data = {
                                    KaiwuDRLDefine.ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST: self.aisrv_ask_learner_to_execute_on_policy
                                }
                    message[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE] = send_data

                    self.learner_have_send_heartbeat_request_success = True

                    # aisrv返回给learner确认信息
                    self.zmq_server.send(str(self.client_id), message, binary=False)

                    self.logger.debug(f"AiSrv learner ask aisrv {self.client_id} msg_type: {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]} success")

                elif message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST:
                    model_version = message[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE]

                    message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE] = KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE

                    aisrv_process_model_version_success = self.aisrv_process_when_model_version_change(model_version)
                    if aisrv_process_model_version_success:
                        self.aisrv_change_model_version_success_count += 1
                    else:
                        self.aisrv_change_model_version_error_count += 1

                    message[KaiwuDRLDefine.ON_POLICY_MESSAGE_VALUE] = aisrv_process_model_version_success

                    # aisrv返回给learner确认信息
                    self.zmq_server.send(str(self.client_id), message, binary=False)

                    self.logger.info(f"AiSrv learner ask aisrv {self.client_id} msg_type: {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]} success")

                else:
                    self.logger.error(f"AiSrv learner ask aisrv not support msg_type: {message[KaiwuDRLDefine.ON_POLICY_MESSAGE_TYPE]}")
                    return

        except Exception as e:
            pass
    
    '''
    learner让aisrv走on-policy流程, aisrv其具体的有下面操作:
    1. 更新kaiwu_rl_helper的model_version
    2. aisrv主进程设置继续kaiwu_rl_helper线程标志位为True
    3. aisrv的handler进程看见2中的标准位, 继续kaiwu_rl_helper线程
    4. aisrv主进程收集所有的handler进程标志位
    4.1 如果不等于则超时处理
    4.2 如果等于则执行5
    5. aisrv主进程设置继续kaiwu_rl_helper线程标志位为False
    6. aisrv主进程将self.kaiwu_rl_helper_continue_count全部赋值为False
    7. 给learner回响应包代表本次流程OK
    8. 初始化部分指标, 下个周期再进入

    返回True表示执行成功, 返回False表示执行失败, 需要做容错处理
    '''
    def aisrv_process_when_model_version_change(self, model_version):
        aisrv_process_model_version_success = True

        # 设置handler_version
        with self.active_handlers_model_version_lock:
            self.active_handlers_model_version.value = model_version

        # 按照单帧/单局需要走的逻辑
        if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
            # 设置kaiwu_rl_helper线程继续的标志位
            with self.kaiwu_rl_helper_continue_value_lock:
                self.kaiwu_rl_helper_continue_value.value = True

            retry_count = 0
            kaiwu_rl_helper_continue_count = 0
            active_handlers_alive_count = 0
            while retry_count < int(CONFIG.on_policy_error_retry_count):
                # 需要实时计算值
                kaiwu_rl_helper_continue_count = self.get_kaiwu_rl_helper_continue_count()
                active_handlers_alive_count = self.get_active_handlers_alive_count()
                if kaiwu_rl_helper_continue_count / active_handlers_alive_count >= CONFIG.on_policy_quantity_ratio:
                    break
                else:
                    # 减少CPU争用
                    time.sleep(CONFIG.idle_sleep_second)

                retry_count += 1
            
            # 如果超过了最大重试次数, 则计入监控, 后期做容错处理
            if retry_count >= int(CONFIG.on_policy_error_retry_count):
                self.logger.error(f"AiSrv main process not recv all handler continue kaiwu_rl_helper response, kaiwu_rl_helper_continue_count {kaiwu_rl_helper_continue_count}, active_handlers_alive_count {active_handlers_alive_count}")
                aisrv_process_model_version_success = False
                self.all_kaiwu_rl_helper_continue_error_count += 1
            else:
                self.logger.info(f"AiSrv main process recv all handler continue kaiwu_rl_helper response")
                self.all_kaiwu_rl_helper_continue_success_count += 1
            
            # 无论本次执行成功与否, 下面变量都reset下, 便于下个周期进行处理, 否则就出现aisrv的死锁状态
            self.aisrv_ask_learner_to_execute_on_policy = False
            with self.kaiwu_rl_helper_pause_count_lock:
                self.kaiwu_rl_helper_pause_count.value = 0

            with self.kaiwu_rl_helper_continue_value_lock:
                self.kaiwu_rl_helper_continue_value.value = False

            with self.kaiwu_rl_helper_continue_count_lock:
                self.kaiwu_rl_helper_continue_count.value = 0

            with self.have_set_kaiwu_rl_helper_continue_lock:
                self.have_set_kaiwu_rl_helper_continue[:] = [False] * len(self.have_set_kaiwu_rl_helper_continue)

            with self.have_set_kaiwu_rl_helper_pause_lock:
                self.have_set_kaiwu_rl_helper_pause[:] = [False] * len(self.have_set_kaiwu_rl_helper_pause)
        
        return aisrv_process_model_version_success

    '''
    aisrv启动的learner的on-policy流程
    '''
    def aisrv_on_policy_process(self):
        if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            return
        
        '''
        在on-policy的while True循环里, 主要流程:
        1. 如果是按照对局或者单帧的, 需要走self.on_policy_by_episode()函数
        2. aisrv与learner之间通信交互
        '''
        while True:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                self.on_policy_by_episode()

            # aisrv与learner之间的通信交互函数
            self.on_policy_aisrv_communication_with_learner()

    '''
    主循环, 加上我们的循环后再调用socketserver的主循环
    '''
    def serve_forever(self):

        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            # 采用线程来调用socketserver的主循环
            server_thread = threading.Thread(target=super().serve_forever)
            server_thread.daemon = True
            server_thread.start()

            # aisrv上执行on-policy流程
            self.aisrv_on_policy_process()
        
        else:
            # 默认调用socketserver的serve_forever函数
            super().serve_forever()

class AiSrvHandle(ss.BaseRequestHandler):
    __slots__ = ("logger", "conn", "simu_ctx", "slots", "slot_id", "msg_buff", "data_queue", "kaiwu_rl_helper", "min_slot_id",
                 "monitor_proxy")

    def setup(self) -> None:
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/aisrv_handle_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'aisrvhandle')
        self.logger.info(f'aisrvhandle start at pid {pid}')

        self.conn = Connection(self.request)

        # Context, 从AiSrv继承过来
        self.simu_ctx = self.server.simu_ctx
        self.simu_ctx.exit_flag = multiprocessing.Value('b', False)

        # 设置客户端连接地址
        self.simu_ctx.client_address = str(self.client_address)
        self.slots = self.simu_ctx.slots
        self.slot_id = self.slots.get_slot()
        self.simu_ctx.slot_id = self.slot_id

        # 设置aisrv上对客户端的消息buff, 匹配速度
        self.msg_buff = MsgBuff(self.simu_ctx)
        self.simu_ctx.msg_buff = self.msg_buff
        
        # 负责统计kaiwu_rl_helper中产生的对局数据，线程和主进程间只用数据dequeue即可，减少cpu消耗
        self.data_queue = collections.deque(maxlen=CONFIG.max_queue_len)
        self.simu_ctx.data_queue = self.data_queue

        '''
         aisrv下每1个客户端启动1个KaiWuRLHelper对象, 封装了强化学习流程
         1. 如果是主循环的内容在业务侧, 调用self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper
         2. 如果是主循环的内容在框架侧, 调用self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)
         '''
        self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper(self.simu_ctx)

        # self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)
        self.logger.info(f'aisrvhandle use kaiwu_rl_helper: {self.kaiwu_rl_helper}')

        self.kaiwu_rl_helper.daemon = True
        self.kaiwu_rl_helper.start()

        self.min_slot_id, _ = self.slots.get_min_max_slot_id()
        self.logger.info(f'aisrvhandle established connection from {self.client_address}, slot id is {str(self.slot_id)}, min_slot_id is {self.min_slot_id}')

        current_actor_addrs, current_learner_addrs = self.kaiwu_rl_helper.get_current_actor_learner_address()
        self.logger.info(f'aisrvhandle current_actor_addrs is {current_actor_addrs}, current_learner_addrs is {current_learner_addrs}')

        # 启动独立的线程, 负责aisrv与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            if self.slot_id % 2 == 0:
                self.monitor_proxy = MonitorProxy(self.logger)
                self.monitor_proxy.start()

        return super().setup()
    
    '''
    设置每个handler的model_version
    '''
    def set_handler_model_version(self, model_version):
        self.kaiwu_rl_helper.from_learner_model_version = model_version
        self.logger.info(f'aisrvhandle on-policy set_handler_model_version success, model_version: {model_version}')

    '''
    aisrv在处理actor和learner的动态扩缩容逻辑
    '''
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return
        
        current_actor_addrs, current_learner_addrs = self.kaiwu_rl_helper.get_current_actor_learner_address()

        read_from_file_content = CONFIG.read_from_file(CONFIG.svr_name, ['actor_addrs', 'learner_addrs'])
        
        # 本次读取的文件内容错误, 则跳过本次处理下次再进行处理
        try:
            new_actor_addrs = read_from_file_content['actor_addrs'][CONFIG.policy_name]
            new_learner_addrs = read_from_file_content['learner_addrs'][CONFIG.policy_name]
        except Exception as e:
            self.logger.info(f'aisrvhandle load actor address and learner address err, {str(e)}')
            return

        self.aisrv_with_different_actor_learner(current_actor_addrs, new_actor_addrs, current_learner_addrs,  new_learner_addrs)
    
    # actor和learner的IP区别判断, 采用2个参数进行返回
    def check_actor_ip_and_learner_ip_change(self, actor_address, old_actor_address, learner_address, old_learner_addrs):
        actor_ip_change = False
        learner_ip_change = False

        if not actor_address and not learner_address:
            return actor_ip_change, learner_ip_change

        if actor_address:
            if not is_list_eq(actor_address, old_actor_address):
                actor_ip_change = True            
        
        if learner_address:
            if not is_list_eq(learner_address, old_learner_addrs):
                learner_ip_change = True
        
        return actor_ip_change, learner_ip_change

    def aisrv_with_different_actor_learner(self, current_actor_addrs, new_actor_addrs, current_learner_addrs, new_learner_addrs):

        actor_ip_change, learner_ip_change = self.check_actor_ip_and_learner_ip_change(new_actor_addrs, current_actor_addrs, 
                                                        new_learner_addrs, current_learner_addrs)

        if not actor_ip_change and not learner_ip_change:
            return
        
        # actor地址有变化
        if actor_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(current_actor_addrs, new_actor_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增actor_proxy
                actor_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have, 
                                                None, None)
            
            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少actor_ip
                actor_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(KaiwuDRLDefine.PROCESS_REDUCE, list_B_have_A_not_have, 
                                                None, None)

        # learner地址有变化
        if learner_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(new_learner_addrs, current_learner_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增learner_proxy
                learner_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(None, None, KaiwuDRLDefine.PROCESS_ADD, 
                                                list_A_have_B_not_have)
                
            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少learner_ip
                learner_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(None, None, KaiwuDRLDefine.PROCESS_REDUCE, 
                                               list_B_have_A_not_have)

        # 修改配置文件内容落地
        if actor_add_result and actor_reduce_result and learner_add_result and learner_reduce_result:
            self.logger.info("aisrvhandle aisrv_with_different_actor_learner change finish sucess")

    '''
    业务的上报, 选择存在的最小slot_id进行上报
    '''
    def aisrv_stat(self):
        if int(CONFIG.use_prometheus):
            if self.slot_id % 2 == 0:
                monitor_data = {}
                monitor_data[KaiwuDRLDefine.AISRV_TCP_BATTLESRV ] = actor_learner_aisrv_count(self.host, CONFIG.svr_name)
                
                #reward_value = self.kaiwu_rl_helper.get_current_reward_value()
                
                # 目前只传了一个episode的reward sum值
                # 如果reward_value能上报则上报, 不能上报则不需要上报
                if len(self.data_queue)>0:
                    reward_value = sum(self.data_queue)/len(self.data_queue)
                    monitor_data[KaiwuDRLDefine.REAWRD_VALUE] = reward_value

                '''
                收集业务的算法指标上报
                '''
                app_monitor_data = self.kaiwu_rl_helper.app_monitor_data
                for key, value in app_monitor_data.items():
                    monitor_data[key] = float(value)

                self.monitor_proxy.put_data(monitor_data)
                    

    def before_run(self):

        # 支持每局结束前, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, 
                                    CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, self.logger)
            
            self.logger.info(f'aisrvhandle RainbowUtils {self.rainbow_utils.identity}')

            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
            self.rainbow_activate_single_process(CONFIG.svr_name)
        
        '''
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        '''
        if CONFIG.actor_learner_expansion:
            set_schedule_event(int(CONFIG.alloc_process_per_seconds), self.aisrv_with_new_actor_learner_change)
        
        '''
        设置了aisrv上报业务训练数据
        '''
        if int(CONFIG.use_prometheus):
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.aisrv_stat)
        
        # 获取本机IP
        self.host = get_host_ip()

        '''
        采用单独的线程来执行aisrv_handler_on_policy_process逻辑
        '''
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                # 活着的handler增加1个
                self.server.set_active_handlers_alive_count_value(1)
                self.logger.info(f'aisrvhandle add a handler, slot_id: {self.slot_id}')

                # 将aisrv_handler_on_policy_process放在单独的线程里处理
                server_thread = threading.Thread(target=self.aisrv_handler_on_policy_process)
                server_thread.daemon = True
                server_thread.start()

    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            self.logger.error(f'aisrvhandle read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(f'aisrvhandle read_from_rainbow failed, data is None or data len is 0')
            return
            
        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(process_name, to_change_key_values)
        self.logger.info(f"aisrvhandle {process_name} CONFIG save_to_file success")

    def finish(self) -> None:
        self.logger.info("aisrvhandle finish")

        # join可能会导致线程卡死，无法释放
        self.kaiwu_rl_helper.join()
        
        #最后再上报一次
        self.aisrv_stat()
        time.sleep(1)
 
        # 回收slot_id
        self.slots.put_slot(self.slot_id)

        # handler的finish方法都会调用
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                self.server.set_active_handlers_alive_count_value(-1)
                self.logger.info(f'aisrvhandle reduce a handler, slot_id: {self.slot_id}')

        super().finish()
        self.logger.info('aisrvhandle lost connection from {}', str(self.client_address))
    
    '''
    设置kaiwu_rl_helper继续执行
    '''
    def set_kaiwu_rl_helper_continue(self):
        self.kaiwu_rl_helper.process_continue()

    '''
    更新self.kaiwu_rl_helper_pause_count的值, 主要操作为增加1或者减少1
    '''
    def set_kaiwu_rl_helper_should_pause_value(self, value):
        with self.server.kaiwu_rl_helper_pause_count_lock:
                self.server.kaiwu_rl_helper_pause_count.value += value
    
    '''
    更新self.kaiwu_rl_helper_continue_count的值, 主要操作为增加1或者减少1
    '''
    def set_kaiwu_rl_helper_should_continue_value(self, value):
        with self.server.kaiwu_rl_helper_continue_count_lock:
            self.server.kaiwu_rl_helper_continue_count.value += value

    '''
    aisrv_handler的流程如下:
    1. 如果需要更新kaiwu_rl_helper的model_version, 则更新
    2. 如果是按照步或者对局来执行on-policy的, 则如果需要执行继续kaiwu_rl_helper操作则执行
    3. 如果是按照步或者对局来执行on-policy的, 收集获取到的暂停的kaiwu_rl_helper的数量
    '''
    def aisrv_handler_on_policy_process(self):
        if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            return
        
        # 记录上次更新的model_version
        self.last_active_handlers_model_version = -1
        # 记录是否已经做过pause操作, 只有做过1次pause操作的才能做continue操作
        self.have_execute_pause = False

        while not self.simu_ctx.exit_flag.value:
            # 如果本次需要更新的model_version和AisrvHandler维护的model_version一致, 则不需要更新, 否则更新
            current_active_model_version = self.server.get_active_handlers_model_version_value()
            if self.last_active_handlers_model_version == current_active_model_version:
                # 本次没有model_version更新则短暂休息下, 规避CPU耗用
                time.sleep(CONFIG.idle_sleep_second)
            else:
                # 如果不一致说明learner已经让aisrv更新了新的model_version了, 则代表learner已经走了on-policy的流程
                self.set_handler_model_version(current_active_model_version)
                self.last_active_handlers_model_version = current_active_model_version
        
            # 按照每帧/每局逻辑下继续进程处理
            if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE or CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                '''
                aisrv主进程让handler进程之间的处理, 包括下面的流程:
                1. 如果aisrv主进程让handler进程执行continue操作, 则
                1.1 判断该handler是否做过pause操作, 只有做过pause操作的才能做continue操作, 即顺序是pause->continue->pause->continue ......
                1.2 判断kaiwu_rl_helper是否处于continue状态, 如果是则不用操作, 不是则继续处理
                1.3 判断self.have_set_kaiwu_rl_helper_pause是否为True?
                1.3.1 如果是True, 本次计数已经完成不再计数
                1.3.2 如果是False, 则执行下面的操作
                1.3.2.1 调用self.set_kaiwu_rl_helper_continue()让kaiwu_rl_helper线程继续执行
                1.3.2.2 对aisrv主线程继续的变量进行计数 + 1
                1.3.2.3 设置self.have_set_kaiwu_rl_helper_continue为True
                2. 如果handler进程的kaiwu_rl_helper线程已经执行了pause操作, 则
                2.1 判断kaiwu_rl_helper是否处于暂停状态
                2.2 判断self.have_set_kaiwu_rl_helper_pause是否为True?
                2.2.1 如果是True, 本次计数已经完成不再计数
                2.2.2 如果是False, 本次计数, 并且设置self.have_set_kaiwu_rl_helper_pause为True
                2.3 如果aisrv需要handler进程continue时, 此时让self.have_set_kaiwu_rl_helper_pause为False

                '''
                # aisrv主进程需要handler进程的kaiwu_rl_helper线程执行continue操作
                if self.server.get_kaiwu_rl_helper_continue_value():
                    if self.have_execute_pause:
                        if not self.server.get_have_set_kaiwu_rl_helper_continue(self.slot_id):
                            self.set_kaiwu_rl_helper_continue()
                            self.set_kaiwu_rl_helper_should_continue_value(1)

                            self.server.set_have_set_kaiwu_rl_helper_continue(self.slot_id, True)

                # 如果aisrv的handler进程的kaiwu_rl_helper线程处于pause状态
                elif self.kaiwu_rl_helper.get_process_in_pause_statues():
                    if not self.server.get_have_set_kaiwu_rl_helper_pause(self.slot_id):
                        self.set_kaiwu_rl_helper_should_pause_value(1)

                        self.server.set_have_set_kaiwu_rl_helper_pause(self.slot_id, True)

                        self.have_execute_pause = True
                
                # 未来扩展
                else:
                    pass

    def handle_once(self) -> None:
                    
        # 步骤1, 例行任务
        schedule.run_pending()

        # 步骤2, 网络收发包
        try:
            with TimeIt() as ti:
                # recv msg, 从网络上获取一个请求响应包的数据
                recv_msg = self.conn.recv_msg()
                recv_msg = recv_msg.tobytes()
                # 增加LZ4压缩/解压缩
                recv_msg = decompress_data(recv_msg, serialize=False)

            with TimeIt() as ti:
                # 放入到aisrv本地缓冲区MsgBuff里
                send_msg = self.msg_buff.update(recv_msg)
                # 如果有需要给gamecore的回包, 则处理回包
                if send_msg:
                    # 增加LZ4压缩/解压缩
                    send_msg = compress_data(send_msg, serialize=False)

                    self.conn.send_msg(send_msg)
            
        except socket.timeout as e:
            self.logger.error(f'aisrvhandle socket.timeout, traceback.print_exc() is {traceback.format_exc()}')
        except ClientQuitException as e:
            self.logger.error(f'aisrvhandle ClientQuitException error msg is {e.message}')
            self.simu_ctx.exit_flag.value = True

            # 当bt异常退出时候，需要强制回一个结束包，来结束handler线程，释放资源否则会出现内存泄漏问题
            self.ep_end_req()
            
            # 安全退出KaiWuRLHelper
            self.kaiwu_rl_helper.stop()

            return

    def handle(self) -> None:

        # before_run
        self.before_run()

        # 主循环
        try:
            while not self.simu_ctx.exit_flag.value:
                self.handle_once()

        except Exception as e:
            self.logger.error(
                f"aisrvhandle failed to handle message {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            self.simu_ctx.exit_flag.value = True

            # 其他报错同理需要回一个结束包，来结束线程，释放资源
            self.ep_end_req()
            
            # 安全退出KaiWuRLHelper
            self.kaiwu_rl_helper.stop()
            raise e

    def ep_end_req(self):
        builder = flatbuffers.Builder(0)
        ep_end_req = KaiwuMsgHelper.encode_ep_end_req(builder, b'0', 0, b'')
        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_end_req, ep_end_req)
        builder.Finish(req)
        req_msg = builder.Output()
        
        self.msg_buff.input_q.put(req_msg, )
        self.logger.info("aisrvhandle send Ep end frame")
