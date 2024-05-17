#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import multiprocessing
import traceback
import lz4.block
import schedule
import datetime
import time
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.ipc.reverb_util import RevervbUtil
from framework.common.utils.common_func import set_schedule_event, get_random
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
# from framework.common.protocol.aisrv_learner_req_resp_pb2 import AisrvLearnerRequest
from framework.common.ipc.zmq_util import ZmqServer
import numpy as np
from guppy import hpy
import psutil, os
import gc
from framework.common.protocol.AisrvLearnerRequest import AisrvLearnerRequest

'''
数据流程如下:
1. aisrv zmq_client --> learner zmq_server
2. learner zmq_server --> learner reverb_client
3. learner reverb_client --> learner reverb_server

由于learner使用的python版本的reverb性能比aisrv使用的C++版本的zmq性能差, 出现收发速度不匹配问题, 故这里learner上的zmq和reverb进程情况如下:
1. zmq_server, 1个
2. reverb client server, N个, N由配置文件项决定
'''


'''
LearnerServerZmq
'''
class LearnerServerZmq(multiprocessing.Process):
    def __init__(self, learner_server_reverbs) -> None:
        super(LearnerServerZmq, self).__init__()

        self.zmq_server = ZmqServer(CONFIG.ip_address,  str(int(CONFIG.reverb_svr_port)-1))

        self.process_run_count = 0
        
        # 停止标志位
        self.exit_flag = multiprocessing.Value('b', False)

        # LearnerServerReverb进程对象集合
        self.learner_server_reverbs = learner_server_reverbs

        # 全局的PB解析对象
        # self.pb_req = AisrvLearnerRequest()

    def get_data_and_send_to_queue(self):

        # get sample data
        try:
            _, data = self.zmq_server.recv(block=True, binary=True)
            if not data:
                return
                
            '''
            lz4 decompress + pb, lz4解压缩大小设置需要和aisrv的learner_proxy对齐
            1. lz4压缩比20
            2. 320帧样本PB序列化大小为11MB
            3. 按照1和2的结果, 则设置为300MB比较安全
            '''
            data = lz4.block.decompress(data, uncompressed_size=CONFIG.lz4_learner_uncompressed_size)

            '''
            # pb 解析
            self.pb_req.ParseFromString(data)
            bs = self.pb_req.batch_size
            datas = np.array(self.pb_req.data).reshape(bs, -1)
            '''

            # fb解析
            request = AisrvLearnerRequest.GetRootAsAisrvLearnerRequest(data, 0)
            bs = request.BatchSize()
            
            # reshape
            datas = np.array(request.DataAsNumpy()).reshape(bs, -1)

            # 发送样本时, 强制转换成float16
            train_data = [{'input_datas': np.array(sample, dtype=np.float16)} for sample in datas]

            # 随机选择发送给learner_server_reverb
            idx = get_random(0, len(self.learner_server_reverbs) - 1)
            self.learner_server_reverbs[idx].put_data(train_data)
        
        except Exception as e:
            # 这里暂时没有请求aisrv请求是正常现象, 下一个循环接着处理
            self.logger.error(f'learner_server_zmq get_data_and_send_to_queue error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
    
    '''
    返回reverb server的IP和端口
    '''
    def get_zmq_ip(self):
        return f'{self.ip_address}:{int(CONFIG.reverb_svr_port)-1}'
    
    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/learner/learner_server_zmq_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'learner_server_zmq')
        
        self.logger.info(f'learner_server_zmq start at pid {pid}, use protocl flatbuffer', g_not_server_label)

        self.zmq_server.bind()

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        self.process_run_count = 0

        # 定时器采用schedule, need pip install schedule
        self.zmq_server_stat_schedule()
    
    '''
    周期性打印内存占用情况
    '''
    def zmq_server_stat(self):
        pass

        #h = hpy()
        #self.logger.info(h.heap())
        #self.logger.info(f'learner_server_zmq gc count {gc.get_count()}')
        #self.logger.info(f'learner_server_zmq memory: {psutil.virtual_memory()}')
        #self.logger.info(f'learner_server_zmq msg count: {self.zmq_server.get_cache_message_count()}')

    # 定时器采用schedule, need pip install schedule
    def zmq_server_stat_schedule(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_server_stat)

    def run_once(self):

        # get sample data
        self.get_data_and_send_to_queue()

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()
    
    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('learner_server_zmq LearnerServerZmq stop success', g_not_server_label)
    
    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                '''
                由于LearnerServerZmq进程里都是IO操作比较多, 这里减少休息时间
                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0
                '''

            except Exception as e:
                self.logger.error(f'learner_server_zmq run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)


'''
LearnerServerReverb
'''
class LearnerServerReverb(multiprocessing.Process):
    def __init__(self, idx) -> None:
        super(LearnerServerReverb, self).__init__()

        self.learner_addr = 'localhost'
        self.learner_port = int(CONFIG.reverb_svr_port)
        self.process_run_count = 0
        
        # 停止标志位
        self.exit_flag = multiprocessing.Value('b', False)

        # 需要发送给learner的样本数据
        self.train_data = None

        # reverb 工具类, aisrv上采用reverb client将数据发送给learn进程上的reverb server
        self.reverb_table_names = None

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

        # 单个reverb_util对象
        self.revervb_util = None 

        # index, 只是对第1个进行上报处理
        self.idx = idx

        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)
    
    def put_data(self, train_data):
        if not train_data or self.msg_queue.full():
            return False

        self.msg_queue.put(train_data)
        return True

    def get_data(self):
        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            if not self.msg_queue.empty():
                self.train_data = self.msg_queue.get()
        
        except Exception as e:
            self.train_data = None
    
    '''
    返回reverb server的IP和端口
    '''
    def get_reverb_ip(self):
        return f'{self.learner_addr}:{self.learner_port}'
    
    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/learner/learner_server_reverb_{self.idx}_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", f'learner_server_reverb_{self.idx}')
        
        self.logger.info(f'learner_server_reverb_{self.idx} start at pid {pid}', g_not_server_label)

        # 必须放在这里赋值, 否则reverb client会卡住
        self.revervb_util = RevervbUtil(f'{self.learner_addr}:{self.learner_port}', self.logger)

        self.reverb_table_names = ['{}_{}'.format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]
        self.logger.info(f'learner_server_reverb_{self.idx} send reverb server tables is {self.reverb_table_names}', g_not_server_label)

        # 访问普罗米修斯的类, 只有第一个才设置
        if int(CONFIG.use_prometheus) and self.idx == 0:
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        # 因为需要打印统计日志代表进程存活, 故都需要设置下
        self.send_to_reverb_server_stat()

        self.process_run_count = 0
        
        # aisrv朝learner发送的最大样本大小 
        self.max_sample_size = 0

    def reverb_server_stat(self):
        total_succ_cnt, total_error_cnt = self.revervb_util.get_send_to_reverb_server_stat()

        # 只有第一个才上报普罗米修斯
        if int(CONFIG.use_prometheus) and self.idx == 0:
            
             # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except Exception as e:
                msg_queue_size = 0

            monitor_data  = {
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT : total_succ_cnt,
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_ERR_CNT : total_error_cnt, 
                KaiwuDRLDefine.MONITOR_MAX_SAMPLE_SIZE : self.max_sample_size,
                KaiwuDRLDefine.MONITOR_LEARNER_ZMQ_REVERB_QUEUE_LEN : msg_queue_size
            }

            self.monitor_proxy.put_data(monitor_data)

        # 打印日志, 是为了确保进程正常, 1分钟打印1次性能可控
        self.logger.info(f'learner_server_reverb_{self.idx} send reverb server stat, succ_cnt is {total_succ_cnt}, error_cnt is {total_error_cnt}', g_not_server_label)

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

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()
    
    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info(f'learner_server_reverb_{self.idx} LearnerServerReverb stop success', g_not_server_label)
    
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
                self.logger.error(f'learner_server_reverb_{self.idx} run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)

    
    # use reverb client send msq to reverb server
    def send_msg_use_reverb_client(self):
        if not self.train_data:
            return
        
        # reverb_client发送
        self.revervb_util.write_to_reverb_server_simple(self.reverb_table_names, self.train_data)

        # 更新最大样本大小
        input_datas_list = self.train_data
        sample_size = 0
        for agent in input_datas_list:
            sample_size += agent['input_datas'].nbytes

        # 更新最大样本大小
        if sample_size > self.max_sample_size:
            self.max_sample_size = sample_size