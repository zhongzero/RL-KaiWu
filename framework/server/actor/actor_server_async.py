#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
actor采用python进程时, actor_server单独进程进行预测请求/响应收发
'''

from framework.server.actor.actor_server_sync_sender import ActorServerSyncSender
from framework.server.actor.actor_server_sync_receiver import ActorServerSyncReceiver
from framework.common.ipc.zmq_util import ZmqServer
from framework.common.config.config_control import CONFIG

'''
该类主要用于actor <--> aisrv之间的消息处理:
1. aisrv <-- actor方向, 采用self.zmq_client
2. aisrv --> actor方向
2.1 如果是框架定义session, 采用self.zmq_ops_client
2.2 如果是业务定义session, 采用self.zmq_client
'''
class ActorServerASync(object):
    def __init__(self) -> None:
        # actor建立zmq, 从aisrv收到请求, 并且处理后返回给aisrv, 注意其端口必须和aisrv启动的zmq端口一致
        self.zmq_server = ZmqServer(CONFIG.ip_address, CONFIG.zmq_server_port)
        self.zmq_server.bind()

        self.actor_send_server = ActorServerSyncSender(self.zmq_server)
        self.actor_recv_server = ActorServerSyncReceiver(self.zmq_server)
    
    def get_actor_send_server(self):
        return self.actor_send_server
    
    def get_actor_recv_server(self):
        return self.actor_recv_server
