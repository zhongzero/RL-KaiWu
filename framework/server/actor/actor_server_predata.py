#!/usr/bin/env python3
# -*- coding:utf-8 -*-



'''
预处理
actor_server获取数据后, 放在actor_server本地队列, 该类需要将本地队列的数据放在on_policy_predictor的本地队列去, 减少on_policy_predict的获取队列数据时的操作耗时
'''

import os
try:
    import _pickle as pickle
except:
    import pickle

import multiprocessing
import datetime
import traceback
import schedule
import time
from framework.common.config.config_control import CONFIG
from framework.common.utils.slots import Slots
from framework.common.pybind11.zmq_ops import *
from framework.common.utils.common_func import TimeIt, Context, set_schedule_event, compress_data, decompress_data
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.monitor.monitor_proxy import MonitorProxy

'''
该类主要用于actor_server --> on_policy_predictor之间的消息处理:
1. 从数据actor_server的收包方向里的队列数据读出
2. 将数据放入on_policy_predictor的收包方向的队列
'''
class ActorServerPreData(multiprocessing.Process):
    def __init__(self, zmq_receive_server, on_policy_predictor) -> None:
        super(ActorServerPreData, self).__init__()

        self.zmq_receive_server = zmq_receive_server
        self.on_policy_predictor = on_policy_predictor

        # 停止标志位
        self.exit_flag = multiprocessing.Value('b', False)
        
        # 统计数字
        self.max_decompress_time = 0
    
    '''
    返回类的名字, 便于确认调用关系
    '''
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/actor_server_predata_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
        self.logger.info(f'actor_server process pid is {pid}, class name is {self.get_class_name()}', g_not_server_label)

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()
    
    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):
        
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)
    
    def zmq_stat_reset(self):
        self.max_decompress_time = 0

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME : self.max_decompress_time,
            }

            self.monitor_proxy.put_data(monitor_data)

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()
    
    # 操作数据
    def actor_server_predata(self):
        data = self.zmq_receive_server.get_from_to_predict_queue()
        if data:
            # 增加压缩和解压缩耗时
            with TimeIt() as ti:
                decompressed_data = decompress_data(data)
            if self.max_decompress_time < ti.interval:
                self.max_decompress_time = ti.interval

            self.on_policy_predictor.put_to_predict_queue(decompressed_data)
        else:
            self.process_run_idle_count += 1

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_server_predata()
        
        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()
    
    def run(self):
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                '''
                # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于actor的zmq_server是比较忙碌的, 这里暂时不做人为休眠, 后期修改为事件提醒机制
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0
                '''
                    
            except Exception as e:
                self.logger.error(f'actor_server ActorServer run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)

    '''
    停止进程
    '''
    def stop(self):
        self.exit_flag.value = True



