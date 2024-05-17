#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import os
import time
import traceback
import numpy as np
import lz4.block
from framework.common.utils.tf_utils import *
import schedule
import datetime
import asyncio
try:
    import _pickle as pickle
except:
    import pickle
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import TimeIt, set_schedule_event
from framework.common.ipc.zmq_util import ZmqClient, ZmqOpsClient
from framework.common.config.app_conf import AppConf
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from framework.common.pybind11.zmq_ops.zmq_ops import dump_arrays
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.common_func import get_uuid, compress_data, decompress_data, get_mean_and_max
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.protocol.aisrv_actor_req_resp_pb2 import AisrvActorRequest
from framework.common.protocol.aisrv_actor_req_resp_pb2 import AisrvActorResponse

class ActorProxy(multiprocessing.Process):

    def __init__(self, policy_name, index, actor_addr, context) -> None:
        super(ActorProxy, self).__init__()

        self.policy_name = policy_name
        '''
        支持业务自定义和从alloc获取的情况
        1. 默认端口
        2. alloc服务下发的IP和端口
        3. 从配置文件读取的IP和端口
        '''
        # 获取ip
        self.actor_address = actor_addr[0]
        # 获取port
        self.actor_port = actor_addr[1]

        '''
        aisrv <--> actor之间, actor是支持多个aisrv的, 故actor需要知道各个aisrv的client_id, 才能准确回包, 故这里采用uuid方式
        该值会透传给actor, 在actor采用zmq进行回包时, 带上该client_id参数
        '''
        self.client_id = get_uuid()

        '''
        aisrv <--> actor之间采用通信方式
        1. aisrv <-- actor方向, 采用self.zmq_client
        2. aisrv --> actor方向
        2.1 如果是框架定义session, 采用self.zmq_ops_client
        2.2 如果是业务定义session, 采用self.zmq_client
        '''
        
        self.zmq_client = ZmqClient(str(self.client_id), self.actor_address, self.actor_port)

        # 如果aisrv --> actor 采用的是zmq-ops方式, 则需要定义zmq-ops-client
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.zmq_ops_client = ZmqOpsClient(str(self.client_id), self.actor_address, CONFIG.zmq_server_op_port)

        # slots
        self.slots = context.slots
        self.slot_group_name = f'{policy_name}_actor_proxy_{index}'
        self.slots.register_group(self.slot_group_name)
        
        # send msg queue
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        self.last_send_heartbeat = 0

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

        # 统计数目
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0

        self.recv_from_actor_succ_cnt = 0
        self.recv_from_actor_error_cnt = 0

        # 采用压缩算法时, 压缩耗时, 解压缩耗时, 压缩大小
        self.max_compress_time = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0
    
    # 需要区分是哪个agent发送的请求
    def put_predict_data(self, slot_id, agent_id, message_id, model_version, predict_data) -> None:
        if not predict_data or self.msg_queue.full():
            return False
        
        self.msg_queue.put(((slot_id, agent_id, message_id, model_version), predict_data))
        return True
    
    # 同一个对局的两个agent数据同时发送过来，需要actor同时处理
    def put_predict_data_v2(self, slot_id, agent_id_0, agent_id_1, message_id_0, message_id_1, predict_data_0, predict_data_1) -> None:
        if not predict_data_0 or self.msg_queue.full():
            return False
        
        self.msg_queue.put([((slot_id, agent_id_0, message_id_0), predict_data_0),((slot_id, agent_id_1, message_id_1), predict_data_1)])
        return True
    
    def get_predict_data(self, slot_id):

        input_pipe = self.slots.get_input_pipe(self.slot_group_name, slot_id)

        # 设置queue的超时时间
        if input_pipe.poll(CONFIG.queue_wait_timeout):
            return input_pipe.recv()
        
        return None
    
    async def recv_data_from_actor_by_coroutine(self):
        self.recv_data_from_actor_detail()


    def recv_data_from_actor_by_direct(self):
        self.recv_data_from_actor_detail()

    '''
    actor --> aisrv 接收预测响应
    '''
    def recv_data_from_actor_detail(self):
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.aisrv_send_heartbeat_to_actor()
        
        pred_data = self.zmq_client.recv(binary=True)

        try:

            # 增加解压缩
            with TimeIt() as ti:
                pred_data = decompress_data(pred_data)

            # 业务数据反序列化
            pred_data = self.deserialize_buff_data(pred_data)
            
            if self.max_decompress_time < ti.interval:
                self.max_decompress_time = ti.interval

            self.recv_from_actor_succ_cnt += 1
            
            # 处理响应回包
            result_map = {}
            for compose_id, pred_result in pred_data:
                slot_id, agent_id, message_id, model_version = compose_id

                # 增加model_version值
                pred_result['model_version'] = model_version
                result_map.setdefault(slot_id, {})[agent_id] = pred_result

                if CONFIG.distributed_tracing:
                    self.logger.info(f'actor_proxy distributed_tracing compose_id {compose_id} from zmq server {self.get_zmq_server_ip()} success', g_not_server_label)

                '''
                处理响应回包里超时情况, 处理步骤如下:
                1. 按照message_id计算出耗时, 放入time_cost_map
                2. 删除timeout_map对应的message_id项

                主要去掉第一帧耗时大的
                ''' 
                index = (slot_id, agent_id, message_id)
                if index in self.timeout_map:
                    now = int(round(time.time() * 1000))
                    cost_time = now - self.timeout_map.get(index)
                    self.time_cost_map[index] = cost_time

                    del self.timeout_map[index]

            for slot_id, client_results in result_map.items():
                output_pipe = self.slots.get_output_pipe(self.slot_group_name, slot_id)
                output_pipe.send(client_results)
                # self.logger.debug(f'actor_proxy slot_id {slot_id} output_pipe send success', g_not_server_label)

        except Exception as e:
            self.process_run_idle_count += 1

    '''
    包括的操作:
    1. aisrv --> actor发送预测请求
    2. actor --> aisrv接收预测响应
    '''
    def flush_buffer_data(self):
        if self.cur_buf_size <= 0:
            self.logger.info(f'actor_proxy current cur_buf_size is 0, then return', g_not_server_label)
            return

        with TimeIt() as ti:

            # 协程模式
            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_COROUTINE:
                self.aisrv_send_recv_actor_by_direct()

            else:
                self.aisrv_send_recv_actor_by_direct()

        self.cur_buf_size = 0

    '''
    aisrv --> actor 发送预测请求
    '''
    def send_data_to_actor_detail(self, data):

        if not data:
            return
        
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.zmq_ops_client.send(data)
        else:
            self.zmq_client.send(data, binary=True)
        
        self.send_to_actor_succ_cnt += 1
        # self.logger.debug(f'actor_proxy send to zmq_server {self.get_zmq_server_ip()} success, client_id is {self.client_id}', g_not_server_label)

    '''
    针对不同的数据类型, 进行反序列化
    消息的数据格式:
    (compose_id, pred_result)
    1. pickle, 直接返回
    2. protobuf, 从KaiwuServerResponse获取数据后组装返回
    '''
    def deserialize_buff_data(self, data):
        if not data:
            return data

        if CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PICKLE:
            return data
        
        elif CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PROTOBUF:
            kaiwu_server_response = AisrvActorResponse()
            kaiwu_server_response.ParseFromString(data)

            # 适配数据格式，引入数据后处理
            #[(tuple(compose_id), send_data)]
            #send_data = np.array(kaiwu_server_response.logits), np.array(kaiwu_server_response.value),\
            #            np.array(kaiwu_server_response.meta_msg), np.array(kaiwu_server_response.lstm_cell), \
            #            np.array(kaiwu_server_response.lstm_hidden)
                        
            np_output = kaiwu_server_response.output
            cell_hidden = np.array(np_output)[:ModelConfig.LSTM_UNIT_SIZE*2*ModelConfig.HERO_NUM].reshape(ModelConfig.HERO_NUM,-1)
            lstm_cell = cell_hidden[:,:ModelConfig.LSTM_UNIT_SIZE].reshape(-1)
            lstm_hidden = cell_hidden[:,ModelConfig.LSTM_UNIT_SIZE:].reshape(-1)
            ell = np.array(np_output)[ModelConfig.LSTM_UNIT_SIZE*2*ModelConfig.HERO_NUM:].reshape(ModelConfig.HERO_NUM,-1)
            
            label_size = sum(ModelConfig.HERO_LABEL_SIZE_LIST[0])
            logits = ell[:,:label_size].reshape(-1)
            value = ell[:,label_size:label_size+ModelConfig.VALUE_HEAD_NUM].reshape(-1)
            meta_msg = ell[:,label_size+ModelConfig.VALUE_HEAD_NUM:].reshape(-1)
            send_data = logits, value, meta_msg, lstm_cell, lstm_hidden            
            
            return [(tuple(kaiwu_server_response.compose_id), send_data)]
    
    '''
    针对不同的数据类型, 进行序列化
    消息的数据格式:
    (actor_id, slot_id, agent_id) | data
    1. pickle, 可以直接组装msg
    2. protobuf, 需要对KaiwuServerRequest类赋值
    '''
    def serialize_buffer_data(self):
        # 序列化 pre_req(业务State定义的类里的key的顺序) + client_id + compose_id(agent_id, slot_id)
        
        assert self.cur_buf_size==1
        # self.logger.info(f'actor_proxy cur_buf_size is:{self.cur_buf_size}', g_not_server_label)

        # 采用pickle序列化
        if CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PICKLE:
            pred_req = [self.buffer_data[key][:self.cur_buf_size] for key in self.state_keys]

            msg = pred_req + [self.client_id_buf[:self.cur_buf_size], self.compose_id_buf[:self.cur_buf_size]]

        # 采用protobuf序列化
        elif CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PROTOBUF:
            
            kaiwu_server_request = AisrvActorRequest()
            kaiwu_server_request.client_id = self.client_id_buf[:self.cur_buf_size][0]
            kaiwu_server_request.sample_size = 1
            # 形如[[0, 1, 2]]
            kaiwu_server_request.compose_id.extend(self.compose_id_buf[:self.cur_buf_size].flatten().tolist())
            
            # 接入数据前处理过程
            input_array = np.concatenate([self.buffer_data['observation'][0],self.buffer_data['lstm_cell'][0],self.buffer_data['lstm_hidden'][0]],axis=1).reshape(-1)
            kaiwu_server_request.feature.extend(input_array.tolist())
            #kaiwu_server_request.observation.extend(self.buffer_data['observation'][:self.cur_buf_size].flatten().tolist())
            #kaiwu_server_request.lstm_hidden.extend(self.buffer_data['lstm_hidden'][:self.cur_buf_size].flatten().tolist())
            #kaiwu_server_request.lstm_cell.extend(self.buffer_data['lstm_cell'][:self.cur_buf_size].flatten().tolist())

            msg = kaiwu_server_request

        # 增加lz4的压缩
        with TimeIt() as ti:
            compress_msg = compress_data(msg)

        # 压缩耗时和压缩包大小
        if self.max_compress_time < ti.interval:
            self.max_compress_time = ti.interval
        
        compress_msg_len = len(compress_msg)
        if self.max_compress_size < compress_msg_len:
            self.max_compress_size = compress_msg_len

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return dump_arrays(compress_msg)
        
        return compress_msg
    
    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('actor_proxy ActorProxy stop success', g_not_server_label)

    '''
    返回zmq server的IP和端口
    '''
    def get_zmq_server_ip(self):
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            return f'{self.actor_address}:{CONFIG.zmq_server_port}'
        elif CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return f'{self.actor_address}:{CONFIG.zmq_server_op_port}'
        else:
            return None
    
    '''
    每隔多少s朝actor发送心跳请求
    '''
    def aisrv_send_heartbeat_to_actor(self):
        heartbeat = 'heartbeat'
        self.zmq_client.send(heartbeat, binary=True)
        
        #self.logger.debug(f'actor_proxy heartbeat from {self.client_id} to zmq_server({self.actor_address}:{CONFIG.zmq_server_port})', g_not_server_label)

    def before_run(self) -> None:
        
        # 日志处理, 需要放在before run里进行初始化, 否则容易出现卡死
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/actor_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'actor_proxy')
        self.logger.info(f'actor_proxy start at pid {pid}', g_not_server_label)

        # zmq相关处理
        self.zmq_client.connect()
        self.logger.info(f'actor_proxy zmq client connect at {self.actor_address} : {CONFIG.zmq_server_port} with client_id {self.client_id}', g_not_server_label)

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.zmq_ops_client.connect()
            self.logger.info(f'actor_proxy zmq ops client connect at {self.actor_address} : {CONFIG.zmq_server_op_port} with client_id {self.client_id}', g_not_server_label)

        # 填充client_id
        self.client_id_buf = np.empty((CONFIG.proxy_batch_size * 2), np.int32)
        self.client_id_buf.fill(self.client_id)

        # buff 的处理, 填充了COMPOSE_ID(agent_id, slot_id, message_id, model_version)
        self.compose_id_buf = np.empty((CONFIG.proxy_batch_size * 2, KaiwuDRLDefine.COMPOSE_ID_SIZE),
                                       np.int32)

        self.state_space = AppConf[CONFIG.app].policies[self.policy_name].state.state_space()
        self.state_keys = self.state_space.keys()
        self.buffer_data = {key: np.zeros((CONFIG.proxy_batch_size * 2,) +
                                          self.state_space[key].shape, self.state_space[key].dtype)
                            for key in self.state_keys}
        
        self.cur_buf_size = 0

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        '''
        用于做超时控制的, key为aisrv --> actor的message_id, value为发送时间
        1. 发送时, 将message_id和发送时间放在map里
        2. 当响应包回来, 则当前时间 - 发送时间, 即耗时
        3. 如果在一定时间里没有响应包回来, 则开始删除map里的key, 并且记录ERROR日志
        '''
        self.timeout_map = {}
        self.time_cost_map = {}
    
    def prometheus_stat_reset(self):
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0
        self.recv_from_actor_succ_cnt = 0
        self.recv_from_actor_error_cnt = 0

        self.max_compress_size = 0
        self.max_compress_time = 0
        self.max_decompress_time = 0

    '''
    普罗米修斯相关数据上报, 不能阻塞核心流程, 故采用间隔prometheus_stat_per_minutes进行上报处理
    需要考虑该QPS里的占用的map大小, 以防被OOM掉
    '''
    def prometheus_stat(self):

        monitor_data = {}

        # 针对zmq_client的统计
        if int(CONFIG.use_prometheus):

            # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except Exception as e:
                msg_queue_size = 0

            monitor_data = {
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT : self.send_to_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT : self.send_to_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT : self.recv_from_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT : self.recv_from_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_ACTOR_PROXY_QUEUE_LEN : msg_queue_size,
                KaiwuDRLDefine.MONITOR_AISRV_MAX_COMPRESS_TIME : self.max_compress_time,
                KaiwuDRLDefine.MONITOR_AISRV_MAX_DECOMPRESS_TIME : self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_AISRV_MAX_COMPRESS_SIZE : self.max_compress_size

            }

        '''
        self.logger.debug(f'actor_proxy zmq stat, prometheus_stat, send_succ_cnt is {self.send_to_actor_succ_cnt}, \
                          send_error_cnt is {self.send_to_actor_error_cnt} \
                          recv_succ_cnt is {self.recv_from_actor_succ_cnt} \
                          recv_error_cnt is {self.recv_from_actor_error_cnt}', g_not_server_label)
        '''
        
        # 针对aisrv发出去的请求, 有响应包的场景, 只是计算最大值和平均值时延
        if int(CONFIG.use_prometheus):
            mean_value, max_value = get_mean_and_max(self.time_cost_map.values())

            monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MEAN_TIME_COST] = mean_value
            monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MAX_TIME_COST] = max_value
        
        self.time_cost_map.clear()

        # 针对aisrv发出去的请求, 没有响应包的场景
        timeout_cnt = 0
        for key in list(self.timeout_map.keys()):
            value = self.timeout_map.get(key)

            # 计算下来是s为单位
            time_dela = (int(round(time.time() * 1000)) - value) / 1000

            if  time_dela > CONFIG.aisrv_actor_timeout_second_threshold:
                timeout_cnt += 1
                self.logger.error(f'actor_proxy message id {key} timeout after {time_dela} seconds', g_not_server_label)
                
                del self.timeout_map[key]
                continue

        if int(CONFIG.use_prometheus):
            monitor_data[f'{KaiwuDRLDefine.MONITOR_AISRV_ACTOR_TIMEOUT_GT}{CONFIG.aisrv_actor_timeout_second_threshold}'] = timeout_cnt
        
        if monitor_data:
            self.monitor_proxy.put_data(monitor_data)

        self.prometheus_stat_reset()

        # self.logger.info('actor_proxy prometheus_stat success', g_not_server_label)

    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):
        
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.prometheus_stat)

    async def send_data_to_actor_by_coroutine(self):
        msg = self.get_data_from_predict_data_queue()
        if msg:
            self.send_data_to_actor_detail(msg)
            self.cur_buf_size = 0

    '''
    aisrv朝actor的发送逻辑
    1. 组装数据
    2. 发送
    '''
    def send_data_to_actor_by_direct(self):
        msg = self.get_data_from_predict_data_queue()
        if msg:
            self.send_data_to_actor_detail(msg)
            self.cur_buf_size = 0

    def get_data_from_predict_data_queue(self):
        tmp_data = self.msg_queue.get()
        if not tmp_data:
            return None

        # 单帧单agent处理逻辑
        if isinstance(tmp_data, tuple):
            
            compose_id, data = tmp_data

            self.compose_id_buf[self.cur_buf_size] = np.asarray(compose_id).astype(np.int32)

            for key in self.state_keys:
                self.buffer_data[key][self.cur_buf_size] = data[key]
            
            self.cur_buf_size += 1

            '''
            注意:
            1. 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
            2. slot_id, agent_id, message_id作为存放时延的key, 不能加入model_version, 该值可能被actor返回的值修改掉
            '''
            # 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
            (slot_id, agent_id, message_id, model_version) = compose_id
            if message_id != 1:
                self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))
            
            if CONFIG.distributed_tracing:
                self.logger.info(f'actor_proxy distributed_tracing compose_id {compose_id} will send to actor {self.get_zmq_server_ip()}', g_not_server_label)

            msg = self.serialize_buffer_data()
        
        else:
            compose_id_0, data_0 = tmp_data[0]
            compose_id_1, data_1 = tmp_data[1]
            
            (slot_id, agent_id, message_id, model_version) = compose_id_0
            if message_id != 1:
                self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))
                
            kaiwu_server_request = AisrvActorRequest()
            kaiwu_server_request.client_id = self.client_id_buf[0]
            kaiwu_server_request.sample_size = 2
            # 形如[[0, 1, 2]]
            
            kaiwu_server_request.compose_id.extend(np.asarray(compose_id_0).astype(np.int32).flatten().tolist()+np.asarray(compose_id_1).astype(np.int32).flatten().tolist())
            
            # 接入数据前处理过程
            inputs = []
            for data in (data_0, data_1):
                tmp = np.concatenate([np.array(data['observation']),np.array(data['lstm_cell']),np.array(data['lstm_hidden'])],axis=1).reshape(-1)
                inputs.append(tmp)
            input_array = np.concatenate(inputs)
            
            kaiwu_server_request.feature.extend(input_array.tolist())
            msg = kaiwu_server_request

        return msg

    '''
    单次run_once, 采用协程操作
    '''
    async def run_once_by_coroutine(self):

        # 进行预测请求/响应收发
        await asyncio.gather(self.send_data_to_actor_by_coroutine(), self.recv_data_from_actor_by_coroutine())
        
        # 记录发送给actor成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    '''
    单次run_once, 采用串行操作
    '''
    def run_once_by_direct(self) -> None:

        # 发送请求
        self.send_data_to_actor_by_direct()

        # 接收请求
        self.recv_data_from_actor_by_direct()
        
        # 记录发送给actor成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    def run(self):
        self.before_run()

        if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_COROUTINE:
            loop = asyncio.get_event_loop()

        while not self.exit_flag.value:
            try:
                if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_COROUTINE:
                    loop.run_until_complete(self.run_once_by_coroutine())
                else:
                    self.run_once_by_direct()

                '''

                由于zmq的操作耗时, 这不需要进行sleep操作, 这样不会导致CPU 100%的

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0

                '''

            except Exception as e:
                self.logger.error(f'actor_proxy actor_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
