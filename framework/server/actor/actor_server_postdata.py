#!/usr/bin/env python3
# -*- coding:utf-8 -*-



'''
后处理
predict进程处理后, 放在本地队列, 该进程主要将predictor类本地队列放到actor_server的回包队列上, 这样减少操作耗时
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
class ActorServerPostData(multiprocessing.Process):
    def __init__(self, zmq_send_server, on_policy_predictor) -> None:
        super(ActorServerPostData, self).__init__()

        self.zmq_send_server = zmq_send_server
        self.on_policy_predictor = on_policy_predictor

        # 停止标志位
        self.exit_flag = multiprocessing.Value('b', False)
        
        # 统计数字
        self.max_compress_time = 0
        self.max_compress_size = 0
    
    '''
    返回类的名字, 便于确认调用关系
    '''
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/actor_server_postdata_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
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
        self.max_compress_time = 0
        self.max_compress_size = 0

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME : self.max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE : self.max_compress_size,
            }

            self.monitor_proxy.put_data(monitor_data)

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()
    
    '''
    zmq_ops模式下发送响应包
    '''
    def send_response_to_aisrv(self, size, pred):

        if CONFIG.distributed_tracing:
            self.logger.info(f'actor_server distributed_tracing send_response_to_aisrv start', g_not_server_label)
    
        client_ids = pred.pop(KaiwuDRLDefine.CLIENT_ID_TENSOR)
        compose_ids = pred.pop(KaiwuDRLDefine.COMPOSE_ID_TENSOR)

        with TimeIt() as ti:
            dict_obj = {}
            for i in range(size):
                send_data = {k: v[i] for k, v in pred.items()}
                client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                list_obj = dict_obj.setdefault(client_id, [])
                list_obj.append((tuple(compose_id), send_data))
                # self.logger.debug(f'actor_server zmq server will send a new msg {compose_id} to {client_id}', g_not_server_label)
            
            for client_id, send_data in dict_obj.items():
                try:
                    self.zmq_server.recv(block=False, binary=True)
                except Exception as e:
                    pass
                
                self.zmq_server.send(str(client_id), send_data, binary=True)
                self.send_to_aisrv_succ_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(f'actor_server distributed_tracing zmq server send a new msg to {compose_id} success', g_not_server_label)

        if CONFIG.distributed_tracing:
            self.logger.info(f'actor_server distributed_tracing send_response_to_aisrv end', g_not_server_label)

    '''
    组装操作响应包的数据
    '''
    def send_response_to_aisrv_simple_fast(self, size, preds):
        if CONFIG.distributed_tracing:
            self.logger.info('actor_server distributed_tracing send_response_to_aisrv_simple_fast start', g_not_server_label)

        dict_obj = {}

        for j, pred in enumerate(preds):
            client_ids = pred[KaiwuDRLDefine.CLIENT_ID_TENSOR]
            compose_ids = pred[KaiwuDRLDefine.COMPOSE_ID_TENSOR]

            if CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                send_data = pred['pred']
                client_id = client_ids[0]
                compose_id = compose_ids[0]
                res = [(tuple(compose_id), send_data)]
                dict_obj[client_id] = res
            else:
                send_data = {
                    'format_action': pred['pred'][0],
                    'network_sample_info': pred['pred'][1],
                    'lstm_info': pred['pred'][2]
                }

                for i in range(size[j] - 2):
                    client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                    compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                    list_obj = dict_obj.setdefault(client_id, [])
                    list_obj.append((tuple(compose_id), send_data))

        for client_id, send_data in dict_obj.items():

            # 压缩耗时和压缩包大小
            with TimeIt() as ti:
                compressed_data = compress_data(send_data)
            
            if self.max_compress_time < ti.interval:
                self.max_compress_time = ti.interval

            compress_msg_len = len(compressed_data)
            if self.max_compress_size < compress_msg_len:
                self.max_compress_size = compress_msg_len

            # 放回到actor_server队列里
            self.zmq_send_server.put_predict_result_data([client_id, compressed_data])

            if CONFIG.distributed_tracing:
                self.logger.info(f'actor_server distributed_tracing zmq server send a new msg to {client_id} success', g_not_server_label)

        if CONFIG.distributed_tracing:
            self.logger.info('actor_server distributed_tracing send_response_to_aisrv_simple_fast end')

    # 操作数据
    def actor_server_postdata(self):
        data = self.on_policy_predictor.get_predict_result_data()
        if data:
            size, pred = data

            if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                 self.send_response_to_aisrv(size, pred)
            else:
                self.send_response_to_aisrv_simple_fast(size, pred)
        else:
            self.process_run_idle_count += 1

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_server_postdata()
        
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



