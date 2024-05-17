#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import queue
import datetime
import os
import signal
import time
import traceback
from framework.common.config.config_control import CONFIG
from framework.common.monitor.prometheus_utils import PrometheusUtils
from framework.common.logging.kaiwu_logger import KaiwuLogger


'''
此类用于aisrv, actor, learner进程与监控产品(当前是普罗米修斯, 后期可以按照需要调整)
 独立出进程, 减少核心路径消耗
'''


class MonitorProxy(multiprocessing.Process):
    def __init__(self) -> None:
        super(MonitorProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

        '''
        该队列使用场景:
        1. aisrv,actor,learner主进程向该队列放置监控的数据
        2. monitor_proxy从该队列拿出需要监控的数据, 发往普罗米修斯
        '''
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)
        
        self.walk = None 

        # 按照时间间隔判断进程是否存活
        self.last_detection_time = time.time()

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/monitor_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
        self.logger.info(f'monitor_proxy process start at pid is {pid}')

        # PrometheusUtils 工具类, 与普罗米修斯交互操作
        self.prometheus_utils = PrometheusUtils(self.logger)
        self.process_run_count = 0

    '''
    monitor_data采用map形式, 即key/value格式, 监控指标/监控值
    '''

    def put_data(self, monitor_data):
        if not monitor_data:
            self.logger.error(f'monitor_proxy monitor_data is None')
            return

        if self.msg_queue.full():
            self.logger.error(f'monitor_proxy queue is full, return')
            return
        else:
            self.msg_queue.put(monitor_data)

    '''
    采用queue.Queue类的get方法, 减少CPU损耗
    '''
    def get_data(self):
        try:
            return self.msg_queue.get_nowait()
        except queue.Empty:
            return None

    def send_to_prometheus(self, monitor_data):
        if not monitor_data:
            return

        if not isinstance(monitor_data, dict):
            self.logger.error(f'monitor_proxy monitor_data is not dict, return')
            return

        for monitor_name, montor_value in monitor_data.items():
            if isinstance(montor_value, list):
                for i in range(len(montor_value)):
                    self.prometheus_utils.gauge_use(
                        CONFIG.svr_name, monitor_name, monitor_name, montor_value[i])
            else:
                self.prometheus_utils.gauge_use(
                    CONFIG.svr_name, monitor_name, monitor_name, montor_value)

        self.prometheus_utils.push_to_prometheus_gateway()
        self.logger.debug(f'monitor_proxy push_to_prometheus_gateway success')

    def run_once(self):

        # 获取需要监控的数据
        monitor_data = self.get_data()
        if monitor_data:
            self.send_to_prometheus(monitor_data)
        else:
            # 如果本次为空, 则self.process_run_count += 1, 尽快获得休息时间, 减少CPU损耗
            self.process_run_count += 1

        # 进程alive检测
        if self.walk:
            stop = False
            now = time.time()

            if now - self.last_detection_time >= CONFIG.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                if now - self.walk.last_recv_gamecore_time >= CONFIG.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                    self.logger.error(
                        f"pid {self.walk_battlesrv_pid} and alloc pid {self.walk_alloc_id} will killed because of gamecore response timeout: {now - self.walk.last_recv_gamecore_time} seconds")
                    stop = True
                
                if now - self.walk.last_recv_aisrv_time >= CONFIG.battlesrv_recv_response_from_gamecore_aisrv_seconds:
                    self.logger.error(
                        f"pid {self.walk_battlesrv_pid} and alloc pid {self.walk_alloc_id} will killed because of aisrv response timeout: {now - self.walk.last_recv_aisrv_time} seconds")
                    stop = True
            
                if stop:
                    # 安全停止gamecore
                    self.walk.controller.stop_game()

                    # 需要先停止alloc进程, 再停止battlesrv进程
                    if self.walk_alloc_id  > 0:
                        os.kill(self.walk_alloc_id, signal.SIGKILL)
                
                    if self.walk_battlesrv_pid > 0:
                        os.kill(self.walk_battlesrv_pid, signal.SIGKILL)

                self.last_detection_time = now

    def set_walk(self, walk, battlesrv_pid, alloc_pid):
        self.walk = walk

        self.walk_battlesrv_pid = battlesrv_pid
        self.walk_alloc_id = alloc_pid
    '''
    进程停止函数
    '''

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('monitor_proxy MonitorProxy stop success')

    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                time.sleep(CONFIG.idle_sleep_second)

            except Exception as e:
                self.logger.error(
                    f'monitor_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}')
