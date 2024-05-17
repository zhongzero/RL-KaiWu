#!/usr/bin/env python3
# -*- coding:utf-8 -*-



'''
actor采用python进程时, actor_server单独进程进行预测请求/响应收发
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
from framework.common.config.config_control import CONFIG
from framework.common.utils.slots import Slots
from framework.common.pybind11.zmq_ops import *
from framework.common.utils.common_func import TimeIt, Context, set_schedule_event, compress_data
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.monitor.monitor_proxy import MonitorProxy

'''
该类主要用于actor <--> aisrv之间的消息处理:
1. aisrv <-- actor方向, 采用self.zmq_client
2. aisrv --> actor方向
2.1 如果是框架定义session, 采用self.zmq_ops_client
2.2 如果是业务定义session, 采用self.zmq_client
'''
class ActorServerSyncSender(multiprocessing.Process):
    def __init__(self, zmq_server) -> None:
        super(ActorServerSyncSender, self).__init__()

        self.zmq_server = zmq_server

        # 单个zmq server处理多个zmq client的请求和响应
        self.client_id = ''

        # 停止标志位
        self.exit_flag = multiprocessing.Value('b', False)
        
        # 上下文Context
        self.context = Context()
        self.context.slots = Slots(CONFIG.max_tcp_count)
        self.context.slot_group_name = "actor_server"
        self.context.slots.register_group(self.context.slot_group_name)

        # 使用队列模式, 获取actor的预测响应, 发给aisrv
        self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        '''
        if int(CONFIG.use_pipeline_predict):
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)
        
        else:
            self.input_pipe, self.output_pipe = multiprocessing.Pipe(duplex=False)
        '''
        

        # 采用队列模式, actor接收来自aisrv的zmq请求
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            self.to_predict_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 统计指标
        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0

        # actor --> aisrv发送时批量处理耗时
        self.actor_send_to_aisrv_batch_cost_time_ms = 0

        # 采用压缩算法时, 压缩耗时, 解压缩耗时, 压缩大小
        self.max_compress_time = 0
        self.max_compress_size = 0

    '''
    actor --> aisrv发送预测请求的响应
    '''
    def put_predict_result_data(self, predict_result_data):
        if predict_result_data:
            # 采用pickle序列化, 因为采用pipe可以去掉序列化和反序列化步骤
            # predict_result_data = pickle.dumps(predict_result_data)

            if self.predict_result_queue.full():
                # 队列满时报错
                return
            else:
                self.predict_result_queue.put(predict_result_data)
    
    '''
    返回类的名字, 便于确认调用关系
    '''
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/actor_server_sender_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)

        self.logger.info(f'actor_server zmq server bind at {CONFIG.ip_address} : {CONFIG.zmq_server_port}', g_not_server_label)
        self.logger.info(f'actor_server process pid is {pid}, class name is {self.get_class_name()}', g_not_server_label)

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        '''
        # 启动类内线程, 处理预测请求/响应收发, 但是效果不明显
        self.actor_receiver = threading.Thread(target=ActorServerSync.actor_receive_msg, args=(self,)) 
        self.actor_receiver.start()
        self.actor_sender = threading.Thread(target=ActorServerSync.actor_send_msg, args=(self,)) 
        self.actor_sender.start()
        '''

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()
    
    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):
        
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)
    
    def zmq_stat_reset(self):

        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0
        
        self.actor_send_to_aisrv_batch_cost_time_ms = 0
        self.max_compress_time = 0
        self.max_compress_size = 0


    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT: self.send_to_aisrv_succ_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT : self.send_to_aisrv_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS : self.actor_send_to_aisrv_batch_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME : self.max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE : self.max_compress_size,
            }
            self.monitor_proxy.put_data(monitor_data)

        '''
        self.logger.debug(f'actor_server zmq stat, send_succ_cnt is {self.send_to_aisrv_succ_cnt}, \
                          send_error_cnt is {self.send_to_aisrv_error_cnt} \
                          recv_succ cnt is {self.recv_from_aisrv_succ_cnt} \
                          recv_error_cnt is {self.recv_from_aisrv_error_cnt}', g_not_server_label)
        '''

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()
    
    '''
    如果本次无预测请求, 返回None; 否则返回数据, 注意需要处理异常, 规避队列get时阻塞时间
    '''
    def get_from_to_predict_queue(self):
        try:
            return self.to_predict_queue.get(block=False)
        except Exception as e:
            return None
    
    def send_response_to_aisrv(self, size, pred):
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
                # self.logger.debug(f'actor_server zmq server send a new msg to {client_id} success', g_not_server_label)

    '''
    按照批处理返回数据处理
    '''
    def send_response_to_aisrv_simple(self, size, preds):
        #assert isinstance(size, list), "actor size is not list"
        #assert isinstance(preds, list), "actor pred is not list"
        #assert len(size) == len(preds), "actor batch prediction"


        batch_size = len(size)
        for j in range(batch_size):
            client_ids = preds[j][KaiwuDRLDefine.CLIENT_ID_TENSOR]
            compose_ids = preds[j][KaiwuDRLDefine.COMPOSE_ID_TENSOR]
            pred = preds[j]['pred']

            with TimeIt() as ti:
                dict_obj = {}
                '''
                1v1预测结果返回的响应包是format_action, network_sample_info, lstm_info, 需要size - 2
                5v5预测结果返回的响应包是network_sample_info, lstm_info, 需要size - 1
                '''
                if CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                    '''
                    for i in range(size[j] - 1):
                        send_data = {
                            'network_sample_info': pred[0],
                            'lstm_info': pred[1]

                        }

                        client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                        compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                        list_obj = dict_obj.setdefault(client_id, [])
                        list_obj.append((tuple(compose_id), send_data))

                        self.logger.debug(f'actor_server zmq server will send a new msg {compose_id} to {client_id}', g_not_server_label)
                    '''
                    send_data = pred
                    client_id = client_ids[0] 
                    compose_id = compose_ids[0] 
                    res = [(tuple(compose_id), send_data)]
                    dict_obj={client_id:res}

                    
                else:
                    
                    for i in range(size[j] - 2):
                        send_data = {
                            'format_action' : pred[0],
                            'network_sample_info' : pred[1],
                            'lstm_info':pred[2]
                        }

                        client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                        compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                        list_obj = dict_obj.setdefault(client_id, [])
                        list_obj.append((tuple(compose_id), send_data))
                        # self.logger.debug(f'actor_server zmq server will send a new msg {compose_id} to {client_id}', g_not_server_label)
                    
                for client_id, send_data in dict_obj.items():

                    # 增加压缩
                    with TimeIt() as ti:
                        compressd_data = compress_data(send_data)

                    # 压缩耗时和压缩包大小
                    if self.max_compress_time < ti.interval:
                        self.max_compress_time = ti.interval
                    
                    compress_msg_len = len(compressd_data)
                    if self.max_compress_size < compress_msg_len:
                        self.max_compress_size = compress_msg_len

                    self.zmq_server.send(str(client_id), compressd_data, binary=True)

                    self.send_to_aisrv_succ_cnt += 1
                    # self.logger.debug(f'actor_server zmq server send a new msg to {client_id} success', g_not_server_label)

    '''
    类中线程, 处理actor --> aisrv方向预测请求
    '''
    def actor_send_msg(self):

        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            if not self.predict_result_queue.empty():
                data = self.predict_result_queue.get()
            else:
                self.process_run_idle_count += 1

                return

        except Exception as e:
            self.process_run_idle_count += 1

            return

        # 采用pickle反序列化, 因为采用pipe可以去掉序列化和反序列化步骤
        # size, pred = pickle.loads(data, encoding='bytes')

        size, pred = data
        if not size or not pred:
            self.process_run_idle_count += 1

            return

        # 获取client_ids和compose_ids
        with TimeIt() as ti:
            if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                self.send_response_to_aisrv(size, pred)
            else:
                self.send_response_to_aisrv_simple(size, pred)
        
        # 获取采集周期里的最大值
        if self.actor_send_to_aisrv_batch_cost_time_ms < ti.interval * 1000:
            self.actor_send_to_aisrv_batch_cost_time_ms = ti.interval * 1000

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_send_msg()
        
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



