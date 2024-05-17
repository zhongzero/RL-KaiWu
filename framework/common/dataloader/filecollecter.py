#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import datetime
import os
import time
import traceback
import json
import schedule
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.common_func import is_list_eq, set_schedule_event, python_exec_shell
from framework.common.monitor.prometheus_utils import PrometheusUtils
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine


'''
该类主要是负责样本文件传输, 需要传入要传递到的IP和目录上
'''
class FileCollecter(multiprocessing.Process):
    
    def __init__(self, ip, dir) -> None:
        super(FileCollecter, self).__init__(ip, dir)

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

        # 目标IP
        self.ip = ip

        # 目标目录
        self.dir = dir
    
    def before_run(self):
        
        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/file_collecter_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'file_collecter')
        self.logger.info(f'file_collecter start at pid {pid}', g_not_server_label)

        # 访问普罗米修斯的类
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.process_run_count = 0

    def run_once(self):
        
        # 启动定时器
        schedule.run_pending()
    
    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('file_collecter FileCollecter stop success', g_not_server_label)
    
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
                self.logger.error(f'file_collecter run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
