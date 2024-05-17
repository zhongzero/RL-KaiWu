#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import multiprocessing
import sys
import traceback
import lz4.block
import schedule
import datetime
import time
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.ipc.reverb_util import RevervbUtil
from framework.common.utils.common_func import set_schedule_event
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine

class LearnerProxy(multiprocessing.Process):
    def __init__(self, policy_name, learner_addr, context) -> None:
        super(LearnerProxy, self).__init__()

        self.policy_name = policy_name
        
        '''
        支持业务自定义和从alloc获取的情况
        1. 默认端口
        2. alloc服务下发的IP和端口
        3. 从配置文件读取的IP和端口
        '''
        self.learner_addr = learner_addr[0]
        self.learner_port = learner_addr[1]

        '''
        aisrv里主线程放进该Queue, learnproxy采用reverb client发送给reverb server
        这里必须使用manager.queue对象, 否则多进程中传输会出现问题
        '''
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 需要发送给learner的样本数据
        self.train_data = None

        # 需要发送给learner样本的优先级
        self.train_data_prioritezeds = []

        self.context = context

        # reverb 工具类, aisrv上采用reverb client将数据发送给learn进程上的reverb server
        self.revervb_util = None
        self.reverb_table_names = None

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)
    
    def put_data(self, slot_id, train_data, train_data_prioritezeds):
        # 这里不需要指定是哪个battsvr和agent出现的数据, 故只是发送训练数据即可
        if self.msg_queue.full():
            return
        
        self.msg_queue.put((train_data, train_data_prioritezeds))

    def get_data(self):
        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            if not self.msg_queue.empty():
                self.train_data, self.train_data_prioritezeds = self.msg_queue.get()
        
        except Exception as e:
            self.train_data = None
            self.train_data_prioritezeds.clear()
    
    '''
    返回reverb server的IP和端口
    '''
    def get_reverb_ip(self):
        return f'{self.learner_addr}:{self.learner_port}'
    
    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/learner_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'learner_proxy')
        self.logger.info(f'learner_proxy start at pid {pid}', g_not_server_label)

        # 必须放在这里赋值, 否则reverb client会卡住
        self.revervb_util = RevervbUtil(f'{self.learner_addr}:{self.learner_port}', self.logger)

        self.reverb_table_names = ['{}_{}'.format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]
        self.logger.info(f'learner_proxy send reverb server tables is {self.reverb_table_names}', g_not_server_label)

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        self.send_to_reverb_server_stat()

        self.process_run_count = 0
        
        # aisrv朝learner发送的最大样本大小 
        self.max_sample_size = 0

    def reverb_server_stat(self):
        succ_cnt, error_cnt = self.revervb_util.get_send_to_reverb_server_stat()
        
        if int(CONFIG.use_prometheus):
            
             # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except Exception as e:
                msg_queue_size = 0

            monitor_data  = {
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT : succ_cnt,
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_ERR_CNT : error_cnt, 
                KaiwuDRLDefine.MONITOR_MAX_SAMPLE_SIZE : self.max_sample_size,
                KaiwuDRLDefine.MONITOR_AISRV_LEARNER_PROXY_QUEUE_LEN : msg_queue_size
            }

            self.monitor_proxy.put_data(monitor_data)

        # self.logger.info(f'learner_proxy send reverb server stat, succ_cnt is {succ_cnt}, error_cnt is {error_cnt}', g_not_server_label)

    # 定时器采用schedule, need pip install schedule
    def send_to_reverb_server_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.reverb_server_stat)

    def run_once(self):

        # get sample data
        self.get_data()

        # use reverb client send sample data to reverb server
        self.send_msg_use_reverb_client()

        # 重新设置self.train_data为None
        self.train_data = None
        self.train_data_prioritezeds.clear()

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()
    
    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('learner_proxy LearnerProxy stop success', g_not_server_label)
    
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
                self.logger.error(f'learner_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)

    # 发送样本时, 可以对样本进行预处理操作
    def before_send_train_data(self):
        if not self.train_data:
            return

        # 暂时删除step维度
        if 's' in self.train_data.keys():
            del self.train_data['s']

        # 增加lz4压缩
        # compress_train_data = lz4.block.compress(train_data, store_size=False)
    
    def before_send_train_data_simple(self):
        pass
    
        # 增加lz4压缩
        # compress_train_data = lz4.block.compress(train_data, store_size=False)

    # use reverb client send msq to reverb server
    def send_msg_use_reverb_client(self):
        if not self.train_data:
            return

        # 发给reverb server
        self.before_send_train_data_simple()
        self.revervb_util.write_to_reverb_server_simple(self.reverb_table_names, self.train_data, self.train_data_prioritezeds)
        
        # 更新最大样本大小
        input_datas_list = self.train_data
        sample_size = 0
        for agent in input_datas_list:
            sample_size += agent['input_datas'].nbytes

        # 更新最大样本大小
        if sample_size > self.max_sample_size:
            self.max_sample_size = sample_size
