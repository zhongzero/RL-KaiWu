#!/usr/bin/env python
# -*- coding: utf-8 -*-


import queue

from framework.common.config.config_control import CONFIG

class MsgBuff:
    __slots__ = ("exit_flag", "client_address", "input_q", "output_q")
    
    def __init__(self, context):
        self.exit_flag = False

        self.client_address = context.client_address

        # input_q是收队列(gamecore --> aisrv)，output_q是发队列(aisrv --> gamecore)
        self.input_q, self.output_q = queue.Queue(CONFIG.queue_size), queue.Queue(CONFIG.queue_size)

    # 放入从gamecore --> aisrv的消息, 如果有需要aisrv --> gamecore的消息则返回
    def update(self, recv_msg):
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times and not self.exit_flag:
            try:
                self.input_q.put(recv_msg,)
            except queue.Full:
                pass
            else:
                break
            retry_num += 1
        if retry_num >= CONFIG.socket_retry_times:
            raise RuntimeError("gamecore %s failed to put msg into input queue" %
                               (self.client_address))
        if self.exit_flag:
            raise RuntimeError("gamecore %s exit..." %
                               (self.client_address))

        send_msg = None
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times and not self.exit_flag:
            try:
                send_msg = self.output_q.get(block=True,timeout=None)
            except queue.Empty:
                pass
            else:
                break
            retry_num += 1
        if retry_num >= CONFIG.socket_retry_times:
            raise RuntimeError("gamecore %s failed to get msg from output queue" %
                               (self.client_address))
        if self.exit_flag:
            raise RuntimeError("gamecore %s exit..." %
                               (self.client_address))

        return send_msg

    # gamecore --> aisrv的消息放入了input_q
    def recv_msg(self):
        """
        从网络中接收一个字符串消息, 默认是json字符串, 这里不对格式做解析
        :return: 字符串，接收的消息, 当gamecore退出时会返回None
        """
        json_str = None
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times and not self.exit_flag and json_str is None:
            try:
                retry_num += 1
                json_str = self.input_q.get()
            except queue.Empty:
                continue

        if json_str is None:
            raise RuntimeError("gamecore %s failed to get msg from input queue" %
                               (self.client_address))
        return json_str

    # aisrv --> gamecore的消息放入了output_q
    def send_msg(self, json_str):
        """
        发送一个消息给客户端(gamecore)
        :param json_str: 编码后的消息体, 默认是json字符串
        """
        retry_num = 0
        while retry_num < CONFIG.socket_retry_times and not self.exit_flag and json_str is not None:
            try:
                retry_num += 1
                self.output_q.put(json_str, )
            except queue.Full:
                continue
            json_str = None

        if json_str is not None:
            raise RuntimeError("gamecore %s failed to put msg into output queue" %
                               (self.client_address))

    def get(self):
        return self.recv_msg()

    def put(self, json_str):
        self.send_msg(json_str)

    def qsize(self):
        return self.output_q.qsize()