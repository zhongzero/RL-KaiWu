#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import datetime
import os
import time
import traceback
import schedule
import copy
import json
from framework.common.config.config_control import CONFIG
from framework.common.alloc.alloc_utils import AllocUtils
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.common_func import is_list_eq, set_schedule_event, python_exec_shell
from framework.common.monitor.prometheus_utils import PrometheusUtils
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine


'''
该类主要是KaiwuDRL上的aisrv、actor、learner与alloc交互的进程, 独立出进程, 减少核心路径消耗
1. aisrv, 服务发现, IP分配
2. actor, 服务发现
3. learner, 服务发现
'''


class AllocProxy(multiprocessing.Process):

    def __init__(self) -> None:
        super(AllocProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/alloc_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'alloc_proxy')
        self.logger.info(f'alloc_proxy start at pid {pid}, Due to the large amount of logs, the log is printed only when the registration is wrong. ', g_not_server_label)

        # alloc 工具类, 与alloc交互操作
        self.alloc_util = AllocUtils(self.logger)

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.prometheus_utils = PrometheusUtils(self.logger)

        self.set_event_alloc_interact()

        self.process_run_count = 0

    def check_actor_learner_from_alloc_init(self):
        '''
        如果是非self_play模式, 旧的actor和learner只是2个list
        如果是self_play模式, 旧的actor和learner要分新旧policy的, 参见配置文件conf/framework/aisrv.toml
        '''
        if not int(CONFIG.self_play):
            self.old_actor_address = []
            self.old_learner_address = []
        else:
            self.old_self_play_actor_address = []
            self.old_self_play_old_actor_address = []
            self.old_self_play_learner_address = []
            self.old_self_play_old_learner_address = []

        ''' 
         从配置读取后进行复制
        '''
        try:
            # 如果是非self_play模式, 则按照policy_name来获取actor和learner地址
            if not int(CONFIG.self_play):
                self.old_actor_address = CONFIG.actor_addrs[CONFIG.policy_name].copy()
                self.old_learner_address = CONFIG.learner_addrs[CONFIG.policy_name].copy()
            # 如果是self_play模式, 则按照self_play_policy和self_play_old_policy来获取actor和learner地址
            else:
                self.old_self_play_actor_address = CONFIG.actor_addrs[CONFIG.self_play_policy].copy()
                self.old_self_play_old_actor_address = CONFIG.actor_addrs[CONFIG.self_play_old_policy].copy()
                self.old_self_play_learner_address = CONFIG.learner_addrs[CONFIG.self_play_policy].copy()
                self.old_self_play_old_learner_address = CONFIG.learner_addrs[CONFIG.self_play_old_policy].copy()
        except Exception as e:
            self.logger.error(
                f'alloc_proxy get actor and learner address from conf failed, error is {str(e)}', g_not_server_label)

    def set_event_alloc_interact(self):
        set_schedule_event(int(CONFIG.alloc_process_per_seconds), self.alloc_interact, 'seconds')
    
    '''
    aisrv/actor/learner进程与alloc交互
    '''

    def alloc_interact(self):
        code, msg = self.alloc_util.registry()
        # 服务发现的每隔N秒进行, 导致打印的日志比较多, 这里采用出错时打印方法
        if not code:
            self.logger.error(f"alloc_proxy alloc interact registry fail, will retry next time, error_msg is {msg}", g_not_server_label)

            # 如果本次的注册失败, 表明alloc服务不稳定, 不需要进行下一步操作, 等下一次再操作
            return

        if KaiwuDRLDefine.SERVER_AISRV == CONFIG.svr_name:
            # 对比项目reset
            self.check_actor_learner_from_alloc_init()

            # aisrv需要从alloc拉取最新的actor和learner地址, 然后更新内存里的配置, 再更新配置文件, 更新到最新的与新的actor和learner之间的连接
            self.check_actor_learner_from_alloc()

    '''
    aisrv间隔的从alloc获取到actor和learner的最新地址
    1. 如果和本地的没有变化, 则跳过
    2. 如果有变化, 则需要更新内存里配置, 更新配置文件, 其他handle再加载配置文件进行更新
    '''

    def check_actor_learner_from_alloc(self):
        # 对于aisrv来说, 是需要拉取actor,learner的IP列表, 是数组类型
        if not int(CONFIG.self_play):
            actor_address, learner_address, _, _ = self.alloc_util.get_actor_learner_ip(
                CONFIG.set_name, None)

            # 注意参数, 前2个是非self-play使用, 后面是self-play使用
            actor_ip_change, learner_ip_change, _, _ = self.check_actor_ip_and_learner_ip_change(
                actor_address, learner_address, None, None, None, None)

            if not actor_ip_change and not learner_ip_change:
                return

            # 如果有actor和learner地址变化, 则落地到配置文件, 更改config_file_change值, 则sosckerserver handle进行处理
            self.save_to_config_file(
                actor_address, learner_address, None, None, None, None)

        else:
            self_play_actor_address, self_play_old_actor_address, self_play_learner_address, self_play_old_learner_address = self.alloc_util.get_actor_learner_ip(
                CONFIG.set_name, CONFIG.self_play_set_name)

            # 注意参数, 前2个是非self-play使用, 后面是self-play使用
            self_actor_ip_change, self_actor_old_ip_change, self_learner_ip_change, self_learner_ip_old_change = self.check_actor_ip_and_learner_ip_change(None, None, self_play_actor_address,
                                                                                                                                                           self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address)

            if not self_actor_ip_change and not self_actor_old_ip_change and not self_learner_ip_change:
                return

            # 如果self-play的地址有变化, 则落地配置文件
            self.save_to_config_file(None, None, self_play_actor_address, self_play_learner_address,
                                     self_play_old_actor_address, self_play_old_learner_address)

    def save_to_config_file(self, actor_addrs, learner_addrs, self_play_actor_address, self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address):

        # 写回配置文件内容
        to_change_key_values = {}

        # 临时保存当前actor和learner地址, 报错则提前返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)
        except Exception as e:
            self.logger.error(
                f'alloc_proxy get actor and learner address from conf failed, error is {str(e)}', g_not_server_label)

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

                self.logger.info(f"alloc_proxy {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success", g_not_server_label)

        else:
            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return

            if self_play_actor_address:
                self_play_actor_proxy_num = len(self_play_actor_address)
                old_actor_address_map[CONFIG.self_play_policy] = self_play_actor_address
                to_change_key_values['self_play_actor_proxy_num'] = self_play_actor_proxy_num

            if self_play_old_actor_address:
                self_play_old_actor_proxy_num = len(
                    self_play_old_actor_address)
                old_actor_address_map[CONFIG.self_play_old_policy] = self_play_old_actor_address
                to_change_key_values['self_play_old_actor_proxy_num'] = self_play_old_actor_proxy_num

            to_change_key_values['actor_addrs'] = old_actor_address_map

            if self_play_learner_address:
                self_play_learner_proxy_num = len(self_play_learner_address)
                old_learner_address_map[CONFIG.self_play_policy] = self_play_learner_address
                to_change_key_values['self_play_learner_proxy_num'] = self_play_learner_proxy_num

            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(
                    self_play_old_learner_address)
                old_learner_address_map[CONFIG.self_play_old_policy] = self_play_old_learner_address
                to_change_key_values['self_play_old_learner_proxy_num'] = self_play_old_learner_proxy_num

            to_change_key_values['learner_addrs'] = old_learner_address_map

            # 修改配置文件内容落地
            if self_play_actor_address or self_play_learner_address or self_play_old_actor_address or self_play_old_learner_address:
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"alloc_proxy {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success", g_not_server_label)
    
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
            self.logger.error(f'alloc_proxy python_exec_shell failed, cmd is {cmd}, error msg is {result_str}')
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
                self.logger.info(f"alloc_proxy {CONFIG.cpp_aisrv_configure} {key} {value}")
        
        self.logger.info(f"alloc_proxy {CONFIG.cpp_aisrv_configure} CONFIG save_to_file success")

    '''
    actor和learner的IP区别判断, 采用2个参数进行返回, 注意函数输入
    1. 如果是非self_play模式, 则对比actor和learner地址
    2. 如果是self_play模式, 则需要对比新旧actor和learner地址
    '''

    def check_actor_ip_and_learner_ip_change(self, actor_address, learner_address, self_play_actor_address,
                                             self_play_learner_address, self_play_old_actor_address, self_play_old_learner_address):

        # 需要区分self-play和非self-play的
        if not int(CONFIG.self_play):
            actor_ip_change = False
            learner_ip_change = False

            if not actor_address and not learner_address:
                return actor_ip_change, learner_ip_change, False, False

            if actor_address:
                if not is_list_eq(actor_address, self.old_actor_address):
                    actor_ip_change = True

            if learner_address:
                if not is_list_eq(learner_address, self.old_learner_address):
                    learner_ip_change = True

            # 本次对比完成后, 则更新actor和learner列表
            self.old_actor_address = actor_address
            self.old_learner_address = learner_address

            return actor_ip_change, learner_ip_change, False, False
        else:
            self_actor_ip_change = False
            self_actor_old_ip_change = False
            self_learner_ip_change = False
            self_learner_ip_old_change = False

            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return self_actor_ip_change, self_actor_old_ip_change, self_learner_ip_change, self_learner_ip_old_change

            if self_play_actor_address:
                if not is_list_eq(self_play_actor_address, self.old_self_play_actor_address):
                    self_actor_ip_change = True

            if self_play_old_actor_address:
                if not is_list_eq(self_play_old_actor_address, self.old_self_play_old_actor_address):
                    self_actor_old_ip_change = True

            if self_play_learner_address:
                if not is_list_eq(self_play_learner_address, self.old_self_play_learner_address):
                    self_learner_ip_change = True

            if self_play_old_learner_address:
                if not is_list_eq(self_play_old_learner_address, self.old_self_play_old_learner_address):
                    self_learner_ip_old_change = True

            # 本次对比完成后, 则更新actor和learner列表
            self.old_self_play_actor_address = self_play_actor_address
            self.old_self_play_old_actor_address = self_play_old_actor_address
            self.old_self_play_learner_address = self_play_learner_address
            self.old_self_play_old_learner_address = self_play_old_learner_address

            return self_actor_ip_change, self_actor_old_ip_change, self_learner_ip_change, self_learner_ip_old_change

    def run_once(self):

        # 启动定时器
        schedule.run_pending()

    '''
    进程停止函数
    '''

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('alloc_proxy AllocProxy stop success',
                         g_not_server_label)

    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(
                    f'alloc_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
