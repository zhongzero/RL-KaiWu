#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from time import sleep
import unittest
import flatbuffers
import socket

from framework.server.aisrv.flatbuffer.kaiwu_msg import *
from framework.common.ipc.connection import Connection
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper

'''
该类主要是模拟客户端进行处理, 与aisrv_socketserver.py配合
'''
class AisrvSocketServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect to aisrv
        try:
            self.new_sock.connect(("127.0.0.1", 8000))
        except socket.error as e:
            print("socke errr: %s" % e)
        
        self.send_header_buff = bytearray(Connection.HEADER_TOTAL_LEN)
        # magic
        self.send_header_buff[0:Connection.MAGIC_LEN] = Connection.MAGIC_NUM.to_bytes(Connection.MAGIC_LEN, byteorder='big')
    
    def add_header(self, msg_len):
        # msg_len
        self.send_header_buff[Connection.MAGIC_LEN:Connection.HEADER_TOTAL_LEN] = msg_len.to_bytes(Connection.DATA_LEN, byteorder='big')
    
    def test_main(self):

        # 模拟gamecore操作   
        
        # test on_init
        print("on_init\n")
        self.on_init()
            
        data = self.new_sock.recv(1024)
        print("data " + str(data))
         # 等待1s, 打印回包

        # test on_ep_start
        print("on_ep_start\n")
        self.on_ep_start()
        data = self.new_sock.recv(1024)
        print("on_ep_start data " + str(data))

        # test on_agent_start
        print("on_agent_start\n")
        self.on_agent_start()
        # 为了收取到响应包
        data = self.new_sock.recv(1024)
        print("data " + str(data))

        self.on_agent_start(1)
        print("on_agent_start2\n")
        data = self.new_sock.recv(1024)
        print("data " + str(data))
        # test on update
        print("on update\n")
        self.on_update()
        # 为了收取到响应包
        data = self.new_sock.recv(1024)
        print("data " + str(data))

        self.on_update_fun()

        sleep(1)
    
        # 释放连接
        self.finsh()

    def on_init(self):
        # 设置请求
        builder = flatbuffers.Builder(0)

        init_req = KaiwuMsgHelper.encode_init_req(builder, 'client-1', 'client_version', b'Hello World!')

        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.init_req, init_req)

        builder.Finish(req)

        buf = builder.Output()

        ''' 解析请求包
        req = Request.Request.GetRootAsRequest(buf, 0)

        seq_no, msg_type, msg = KaiwuMsgHelper.decode_request(req)

        print(seq_no, msg_type, msg)
        '''

        data_len = len(buf)
        
        self.add_header(data_len)

        # 发送头部header
        self.new_sock.send(self.send_header_buff, Connection.HEADER_TOTAL_LEN)

        # 发送数据Data
        self.new_sock.send(buf, data_len)
    
    def on_update(self):
        # 设置请求
        builder = flatbuffers.Builder(0)

        update_req = KaiwuMsgHelper.encode_update_req(builder, 'client-1', 2, {
            0: [b'Hello 1'],
            1: [b'Hello 2']
        })

        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.update_req, update_req)

        builder.Finish(req)

        buf = builder.Output()

        data_len = len(buf)
        
        self.add_header(data_len)

        # 发送头部header
        self.new_sock.send(self.send_header_buff, Connection.HEADER_TOTAL_LEN)

        # 发送数据Data
        self.new_sock.send(buf, data_len)
    
    # 直接调用函数来执行解析
    def on_update_fun(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_update_req(builder, 'client_id', 0, {
            1: [b'Hello 1'],
            2: [b'Hello 2']
        })

        builder.Finish(req)

        buf = builder.Output()

        req = UpdateReq.UpdateReq.GetRootAsUpdateReq(buf, 0)

        client_id, ep_id, data = KaiwuMsgHelper.decode_update_req(req)
        print(client_id)

        for i, agent_id in enumerate(sorted(data)):
            self.assertEqual(agent_id, i + 1)
            frames = data[agent_id]
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0], bytes(f"Hello {i + 1}", encoding="utf8"))

    def on_ep_start(self):
        # 设置请求
        builder = flatbuffers.Builder(0)

        ep_start_req = KaiwuMsgHelper.encode_ep_start_req(builder, "client-1", 2, b'Hello World!')

        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_start_req, ep_start_req)

        builder.Finish(req)

        buf = builder.Output()

        data_len = len(buf)
        
        self.add_header(data_len)

        # 发送头部header
        self.new_sock.send(self.send_header_buff, Connection.HEADER_TOTAL_LEN)

        # 发送数据Data
        self.new_sock.send(buf, data_len)
    
    def on_agent_start(self,agent_id = 0):
        # 设置请求
        builder = flatbuffers.Builder(0)

        agent_start_req = KaiwuMsgHelper.encode_agent_start_req(builder, "client-1", 2, agent_id, b'Hello World!')

        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.agent_start_req, agent_start_req)
        builder.Finish(req)

        buf = builder.Output()

        data_len = len(buf)
        
        self.add_header(data_len)

        # 发送头部header
        self.new_sock.send(self.send_header_buff, Connection.HEADER_TOTAL_LEN)

        # 发送数据Data
        self.new_sock.send(buf, data_len)


    def finsh(self):
        self.new_sock.close()

if __name__ == '__main__':
    unittest.main()
