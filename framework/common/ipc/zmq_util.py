#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import queue
import socket
import threading
import time

try:
    import _pickle as pickle
except:
    import pickle

# need pip install pyzmq
import zmq

from framework.common.config.config_control import CONFIG

def pick_unused_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    addr, port = s.getsockname()
    s.close()
    return port

'''
适配zmq-ops的ZmqReader
'''
class ZmqReader:
    def __init__(self, port, bind=False, hwm=50):
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self._reader = self._context.socket(zmq.PULL)
        self._reader.setsockopt(zmq.LINGER, 0)
        self._reader.setsockopt(zmq.RCVHWM, hwm)
        self._reader.setsockopt(zmq.RCVBUF, 4 * 1024 * 1024)
        if bind:
            self._reader.bind("tcp://127.0.0.1:" + str(port))
        else:
            self._reader.connect("tcp://127.0.0.1:" + str(port))

    def get(self, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                return self._reader.recv() if binary else self._reader.recv_pyobj()
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Empty
            else:
                if not self._lock.acquire(False):
                    raise queue.Empty
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._reader.poll(timeout * 1000, flags=zmq.POLLIN):
                        raise queue.Empty
                elif not self.readable():
                    raise queue.Empty
                return self._reader.recv() if binary else self._reader.recv_pyobj()
            finally:
                self._lock.release()

    def get_nowait(self):
        return self.get(block=False)

    def readable(self):
        return self._reader.poll(0, flags=zmq.POLLIN) == zmq.POLLIN

'''
适配zmq-ops的ZmqWriter
'''
class ZmqWriter:
    def __init__(self, port, bind=False, hwm=50):
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self._writer = self._context.socket(zmq.PUSH)
        self._writer.setsockopt(zmq.LINGER, 0)
        self._writer.setsockopt(zmq.SNDHWM, hwm)
        self._writer.setsockopt(zmq.SNDBUF, 4 * 1024 * 1024)
        if bind:
            self._writer.bind("tcp://127.0.0.1:" + str(port))
        else:
            self._writer.connect("tcp://127.0.0.1:" + str(port))

    def put_nowait(self, obj):
        return self.put(obj, block=False)

    def put(self, obj, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                self._writer.send(obj, copy=False) if binary else self._writer.send_pyobj(obj)
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Full
            else:
                if not self._lock.acquire(False):
                    raise queue.Full
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._writer.poll(timeout * 1000, flags=zmq.POLLOUT):
                        raise queue.Full
                elif not self.writable():
                    raise queue.Full
                self._writer.send(obj) if binary else self._writer.send_pyobj(obj)
            finally:
                self._lock.release()

    def writable(self):
        return self._writer.poll(0, flags=zmq.POLLOUT) == zmq.POLLOUT
    
'''
zmq_server:
    svr = ZmqServer('127.0.0.1', 9999)
    while True:
        client_id, data = svr.recv()
        print("server receive data: " + str(data))

        svr.send(client_id, data)
        print("server send data: " + str(data))

zmq_client:
    svr = ZmqClient('client-id', '127.0.0.1', 9999)
    while True:
        data = "hello world"
        svr.send(data)
        print("client send data: " + str(data))

        data = svr.recv()
        print("client receive data: " + str(data))

先启动zmq_server

注意下面情况：
1. 多进程使用时, 在init函数里调用init, 在run函数里调用bind
2. 单进程使用时, 连续调用init和bind即可

如果不按照上述方法调用, zmq在多进程环境里调用会出现收发包异常的情况
'''

class ZmqServer:
    def __init__(self, ip, port):
        self._context = zmq.Context()

        '''
        推荐的值取决于应用程序的需求和机器的硬件资源
        '''
        self._context.set(zmq.IO_THREADS, CONFIG.zmq_io_threads_server)
        self._lock = threading.Lock()

        self.ip = ip
        self.port = port
    
    def bind(self):
        # zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)

        # 设置下网络参数
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, CONFIG.tcp_keep_alive)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, CONFIG.tcp_keep_alive_idle)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, CONFIG.tcp_keep_alive_intvl)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, CONFIG.tcp_keep_alive_cnt)
        self._socket.setsockopt(zmq.SNDBUF, CONFIG.sock_buff_size)
        self._socket.setsockopt(zmq.RCVBUF, CONFIG.sock_buff_size)
        self._socket.setsockopt(zmq.BACKLOG, CONFIG.backlog_size)
        self._socket.setsockopt(zmq.IMMEDIATE, CONFIG.tcp_immediate)

        '''
        zmq的发送和接收缓冲区大小对性能影响很大。如果缓冲区大小太小, 会导致消息堆积和阻塞,从而降低整体性能
        如果缓冲区大小太大, 会导致内存占用过多,从而影响系统的稳定性和可靠性
        目前KaiwuDRL使用到zmq的场景有:
        1. aisrv <--> actor, 单个包比较小, 设置发送接收缓冲区10MB合理
        2. aisrv <--> learner, 单个包比较大, 设置发送接收缓冲区30MB合理
        综合1和2的情况, 故设置发送接收缓冲区30MB合理
        '''

        self._socket.setsockopt(zmq.SNDHWM, CONFIG.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, CONFIG.zmq_ops_recvhwm)
        
        # self._socket.set_hwm(CONFIG.zmq_ops_hwm)
        self._socket.bind("tcp://" + self.ip + ":" + str(self.port))

    def readable(self):
        return self._socket.poll(0, flags=zmq.POLLIN) == zmq.POLLIN

    def recv_nowait(self):
        return self.recv(block=False)

    def recv(self, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                # 注意zmq不同命令字返回的参数个数不一样
                if not CONFIG.cpp_daemon_send_recv_zmq_data:
                    client_id, _, data = self._socket.recv_multipart()
                else:
                    [client_id, data] = self._socket.recv_multipart()
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Empty
            else:
                if not self._lock.acquire(False):
                    raise queue.Empty
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLIN):
                        raise queue.Empty
                elif not self.readable():
                    raise queue.Empty
                # 注意zmq不同命令字返回的参数个数不一样
                if not CONFIG.cpp_daemon_send_recv_zmq_data:
                    client_id, _, data = self._socket.recv_multipart()
                else:
                    [client_id, data] = self._socket.recv_multipart()
            finally:
                self._lock.release()

        client_id = str(client_id, 'utf-8')
        if not binary:
            data = pickle.loads(data)

        return client_id, data

    def writable(self):
        return self._socket.poll(0, flags=zmq.POLLOUT) == zmq.POLLOUT

    '''
    获取本地缓存的消息大小
    '''
    def get_cache_message_count(self):
        return self._socket.getsockopt(zmq.RCVBUF)

    def send_nowait(self, client_id, data):
        self.send(client_id, data, block=False)

    def send(self, client_id, data, block=True, timeout=-1, binary=False):
        client_id = bytes(client_id, 'utf-8')
        if not binary:
            data = pickle.dumps(data)

        if block and timeout == -1:
            with self._lock:
                self._socket.send_multipart([
                    client_id, b'', data
                ])
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Full
            else:
                if not self._lock.acquire(False):
                    raise queue.Full
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLOUT):
                        raise queue.Full
                elif not self.writable():
                    raise queue.Full
                self._socket.send_multipart([
                    client_id, b'', data
                ])
            finally:
                self._lock.release()


'''
注意下面情况：
1. 多进程使用时, 在init函数里调用init, 在run函数里调用connect
2. 单进程使用时, 连续调用init和connect即可

如果不按照上述方法调用, zmq在多进程环境里调用会出现收发包异常的情况
'''
class ZmqClient:
    def __init__(self, client_id, ip, port):
        self._context = zmq.Context()
        self._context.set(zmq.IO_THREADS, CONFIG.zmq_io_threads_client)

        self._lock = threading.Lock()
        self.client_id = client_id
        self.ip = ip
        self.port = port
    
    def connect(self):
        ''' 

        zmq支持的默认: zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER
        C++/python版本zmq: DEALER/ROUTER
        '''
        zmq_type = zmq.DEALER
        if not CONFIG.cpp_daemon_send_recv_zmq_data:
            zmq_type = zmq.REQ

        self._socket = self._context.socket(zmq_type)
        self._socket.setsockopt(zmq.LINGER, 0)

        # 设置下网络参数
        self._socket.setsockopt(zmq.SNDBUF, CONFIG.sock_buff_size)
        self._socket.setsockopt(zmq.RCVBUF, CONFIG.sock_buff_size)

        # 增加重连机制
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, CONFIG.tcp_keep_alive)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, CONFIG.tcp_keep_alive_idle)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, CONFIG.tcp_keep_alive_intvl)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, CONFIG.tcp_keep_alive_cnt)
        self._socket.setsockopt(zmq.SNDHWM, CONFIG.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, CONFIG.zmq_ops_recvhwm)
        self._socket.setsockopt(zmq.IMMEDIATE, CONFIG.tcp_immediate)

        # self._socket.set_hwm(CONFIG.zmq_ops_hwm)
        self._socket.setsockopt(zmq.IDENTITY, bytes(self.client_id, 'utf-8'))
        self._socket.connect("tcp://" + self.ip + ":" + str(self.port))

    def readable(self):
        return self._socket.poll(0, flags=zmq.POLLIN) == zmq.POLLIN

    def recv_nowait(self):
        return self.recv(block=False)

    def recv(self, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                data = self._socket.recv()
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Empty
            else:
                if not self._lock.acquire(False):
                    raise queue.Empty

            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLIN):
                        raise queue.Empty
                elif not self.readable():
                    raise queue.Empty
                data = self._socket.recv()
            finally:
                self._lock.release()

        if not binary:
            data = pickle.loads(data)
        return data

    def writable(self):
        return self._socket.poll(0, flags=zmq.POLLOUT) == zmq.POLLOUT

    def send_nowait(self, data):
        self.send(data, block=False)

    def send(self, data, block=True, timeout=-1, binary=False):
        if not binary:
            data = pickle.dumps(data)

        if block and timeout == -1:
            with self._lock:
                self._socket.send(data)
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Full
            else:
                if not self._lock.acquire(False):
                    raise queue.Full
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLOUT):
                        raise queue.Full
                elif not self.writable():
                    raise queue.Full
                self._socket.send(data)
            finally:
                self._lock.release()

'''
aisrv <--> actror上通信方法
1. aisrv --> actor, aisrv采用ZmqOpsClient, actor上采用ZMQPullSocket(类似Server)
2. actor --> aisrv, aisrv采用ZmqClient, actor上采用ZmqServer
'''
class ZmqOpsClient:
    def __init__(self, client_id, ip, port):
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self.client_id = client_id
        self.ip = ip
        self.port = port
    
    def connect(self):
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, CONFIG.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, CONFIG.zmq_ops_recvhwm)
        
        # self._socket.set_hwm(CONFIG.zmq_ops_hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, bytes(self.client_id, 'utf-8'))
        self._socket.connect("tcp://" + self.ip + ":" + str(self.port))
    
    def send(self, data):
        self._socket.send(data, copy=False)


'''
zmq Poller
'''

class ZmqPoller:
    def __init__(self) -> None:
        self.poller = zmq.Poller()

    def get_poller(self):
        return self.poller
    
    def register(self, socket):
        self.poller.register(socket, zmq.POLLIN)
    
    # 消息达到的标志位
    def get_zmq_pollin_state(self):
        return zmq.POLLIN
    