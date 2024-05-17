#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import json
import multiprocessing
import datetime
import os
import traceback
import schedule
import sys
import time
import yaml
import copy
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.config.config_control import CONFIG
from framework.common.utils.rainbow_utils import RainbowUtils
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.alloc.alloc_proxy import AllocProxy, AllocUtils
from framework.common.utils.common_func import python_exec_shell, make_single_dir, set_schedule_event, actor_learner_aisrv_count, get_host_ip

class AiServer(multiprocessing.Process):
    def __init__(self, ) -> None:
        super(AiServer, self).__init__()

    def rainbow_activate_single_process(self, process_name):
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            self.logger.error(f'AiServer read_from_rainbow failed, msg is {result_msg}')
            return

        if not data or not len(data):
            self.logger.error(f'AiServer read_from_rainbow failed, data is None or data len is 0')
            return
            
        # 更新内存里的值, 再更新配置文件
        to_change_key_values = yaml.load(data[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(process_name, to_change_key_values)
        self.logger.info(f"AiServer {process_name} CONFIG save_to_file success")

    '''
    aisrv在处理actor和learner的动态扩缩容逻辑
    '''
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

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
            self.logger.info(f"AiServer alloc interact registry success")
        else:
            self.logger.error(f"AiServer alloc interact registry fail, will retry next time, error_code is {msg}")
            return

        # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times:
            if not int(CONFIG.self_play):
                actor_address, learner_address, _, _ = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)
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
                f'AiServer server get actor and learner address retry times more than {CONFIG.socket_retry_times}, will start with configure file')
            return
        
        # 修改配置文件
        if not int(CONFIG.self_play):
            self.change_configure_content(actor_address, learner_address, None, None, None, None)
        else:
            self.change_configure_content(None, None, self_play_actor_address, self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address)

    '''
    C++ 常驻进程进程配置文件修改
    '''
    def save_to_file(self, process_name, to_change_key_values):
        if not to_change_key_values or not process_name:
            return
        
        # 先删除actor_addrs,learner_addrs,self_play, actor_proxy_num, learner_proxy_num
        cmd = f"sed -i '/actor_addrs/d' {CONFIG.cpp_aisrv_configure}; sed -i '/learner_addrs/d' {CONFIG.cpp_aisrv_configure}; sed -i '/self_play/d' {CONFIG.cpp_aisrv_configure} \
        ; sed -i '/actor_proxy_num/d' {CONFIG.cpp_aisrv_configure}; sed -i '/learner_proxy_num/d' {CONFIG.cpp_aisrv_configure};"
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            self.logger.error(f'AiServer python_exec_shell failed, cmd is {cmd}, error msg is {result_str}')
            return
    
        # 由于self_play是在main里配置, 这里根据返回的actor_addrs和learner_addrs来决定其值
        actor_addrs_json = json.loads(to_change_key_values.get('actor_addrs'), strict=False)
        self_play = 0
        if len(actor_addrs_json) == 2:
            self_play = 1
        to_change_key_values['self_play'] = self_play

        # 去掉actor_proxy_num和learner_proxy_num参数
        del to_change_key_values['actor_proxy_num']
        del to_change_key_values['learner_proxy_num']
        
        # 追加文件写操作
        with open(CONFIG.cpp_aisrv_configure, 'a', encoding=KaiwuDRLDefine.UTF_8) as f:
            for key, value in to_change_key_values.items():
                # gflags严格要求key=value形式, 不能留空格
                f.write(f'--{key}={value}\n')
                self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} {key} {value}")
        
        self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} CONFIG save_to_file success")

    '''
    修改conf/system/aisrv_system.toml里的配置项目, 如下:
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

        # 将当前的配置文件的内容读成json串, 内存修改后, 再写回json内容, 如果解析json串出错, 则提前报错返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)
        except Exception as e:
            self.logger.error(f'AiServer get actor and learner address from conf failed, error is {str(e)}', g_not_server_label)

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
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

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
                CONFIG.self_play_learner_proxy_num = self_play_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_policy] =  self_play_learner_address
                to_change_key_values['self_play_learner_proxy_num'] = self_play_learner_proxy_num
            
            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(self_play_old_learner_address)
                CONFIG.self_play_old_learner_proxy_num = self_play_old_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_old_policy] =  self_play_old_learner_address
                to_change_key_values['self_play_old_learner_proxy_num'] = self_play_old_learner_proxy_num
            
            to_change_key_values['learner_addrs'] = old_learner_address_map

            # 修改配置文件内容落地
            if self_play_actor_address or self_play_learner_address or self_play_old_actor_address or self_play_old_learner_address:
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)
                
                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")
    
    def run(self) -> None:
        
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
                self.logger.error(f"AiServer failed to run {self.name} . exit. Error is: {e}, traceback.print_exc() is {traceback.format_exc()}")

    def run_once(self) -> None:
        
        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

    '''
    框架运行前创建必要的文件目录
    '''
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)

    # 启动C++常驻进程
    def start_cpp_daemon(self):
        cmd = 'sh tools/aisrv_cpp_server_start.sh'
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            return False
        
        self.logger.info(f'AiServer C++ Daemon Process starts success, cmd is {cmd}')
        return True

    # 从C++ server获取监控信息
    def cpp_stat(self):
        result = self.lib.get_cpp_server_stat_data()
        if not result or not len(result):
            return
        
        # 进行上报, 注意取出来的数据需要强制转换下数据类型
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.AISRV_TCP_BATTLESRV : actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT: result.get(KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT),
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT : result.get(KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT),
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT : result.get(KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_SUC_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_SUC_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_ERR_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_ERR_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_SUC_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_SUC_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_ERR_CNT : result.get(KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_ERR_CNT),
                KaiwuDRLDefine.MONITOR_AISRV_MAX_PROCESSING_TIME : result.get(KaiwuDRLDefine.MONITOR_AISRV_MAX_PROCESSING_TIME),

            }

            self.monitor_proxy.put_data(monitor_data)

            # 指标周期性复原
            self.lib.cpp_server_stat_data_reset()

    def before_run(self):

        # 设置日志Log配置
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/aiserver_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'AiServer')
        self.logger.info(f'AiServer is start at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, pid is {pid}, run_mode is {CONFIG.run_mode}, self_play is {CONFIG.self_play}')

        self.make_dirs()

        # aisrv进程启动时, 从七彩石获取配置
        if int(CONFIG.use_rainbow):
            self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, 
                                CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, self.logger)
        
            self.logger.info(f'AiServer RainbowUtils {self.rainbow_utils.identity}')

            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN)
            self.rainbow_activate_single_process(CONFIG.svr_name)

        # aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址
        if int(CONFIG.use_alloc):
            self.get_actor_learner_ip_from_alloc()

        # C++和python进程不能同时启动
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)
        
        # 需要等alloc获取服务正常后开始启动C++进程
        if not self.start_cpp_daemon():
            self.logger.error(f'AiServer C++ Daemon Process starts failed, please see the log')
            sys.exit(-1)

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        # 启动独立的进程, 负责actor与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        '''
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        '''
        if CONFIG.actor_learner_expansion:
            set_schedule_event(int(CONFIG.alloc_process_per_seconds), self.aisrv_with_new_actor_learner_change)
        
        # 设置python调用C++的类库
        os.chdir('/data/projects/kaiwu-fwk/framework/server/cpp/dist/aisrv/')
        from framework.server.cpp.dist.aisrv.aisrv_server import cpp_aisrv_server

        self.lib = cpp_aisrv_server()
        self.logger.info(f'AiServer C++ lib start success')

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.cpp_stat)
        
        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()
