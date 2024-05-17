#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import socket
import datetime

from framework.common.config.config_control import CONFIG
from framework.interface.exception import ClientQuitException
from framework.common.logging.kaiwu_logger import KaiwuLogger

class Connection(object):
    #
    # Message Format is : Magic | Data Length | Data
    #

    # 64MB
    DEF_BUF_SIZE = 64 * 1024 * 1024
    # Magic Length
    MAGIC_LEN = 4
    # Data Length
    DATA_LEN = 4
    # Header Total Length = Data Length + Magic Length
    HEADER_TOTAL_LEN = DATA_LEN + MAGIC_LEN
    # Magic Num
    MAGIC_NUM = 0x12345678
    # Max Msg Size
    MAX_MSG_SIZE = 128 * 1024 * 1024
    
    def __init__(self, sock) -> None:
        self.sock = sock
        
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CONFIG.sock_buff_size)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CONFIG.sock_buff_size)
        self.sock.settimeout(CONFIG.socket_timeout)
        # self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

        # 设置为阻塞模式, 遇到错误并不会阻止操作
        self.sock.setblocking(1)
        # 端口复用
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.recv_buff = bytearray(Connection.DEF_BUF_SIZE)
        self.send_header_buff = bytearray(Connection.HEADER_TOTAL_LEN)
        self.send_header_buff[0:Connection.MAGIC_LEN] = Connection.MAGIC_NUM.to_bytes(Connection.MAGIC_LEN, byteorder='big')

        
    
    def send_msg(self, send_msg):
        def send_msg_impl(send_buff, total_bytes):
            retry = 0
            left_bytes = total_bytes
            while retry < CONFIG.socket_retry_times and left_bytes > 0:
                bytes_written = self.sock.send(send_buff[total_bytes - left_bytes:])
                left_bytes -= bytes_written
                retry += 1
            if left_bytes > 0:
                raise RuntimeError("failed to send message: msg len %d send len %d, retry %d" % (
                    total_bytes, total_bytes - left_bytes, retry))
        
        msg_len = len(send_msg)
        self.send_header_buff[Connection.MAGIC_LEN:Connection.HEADER_TOTAL_LEN] = msg_len.to_bytes(Connection.DATA_LEN, byteorder='big')

        # send magic + header length
        send_msg_impl(self.send_header_buff + send_msg, Connection.HEADER_TOTAL_LEN+msg_len)
        # send data
        # send_msg_impl(send_msg, msg_len)

    def recv_msg(self):
        magic_number = self.sock.recv(Connection.MAGIC_LEN, socket.MSG_WAITALL)
        if not magic_number:
            raise ClientQuitException(client_id = str(self.sock.getpeername()),quit_code=0, message="magic_number is {}, peer {} close connection.".format(magic_number, str(self.sock.getpeername())))
        magic_number = int.from_bytes(magic_number, byteorder="big")
        if magic_number != Connection.MAGIC_NUM:
            raise RuntimeError("magic number %x is error, right is %x, peer ip: %s" % (magic_number, Connection.MAGIC_NUM, str(self.sock.getpeername())))

        msg_len = self.sock.recv(Connection.DATA_LEN)
        msg_len = int.from_bytes(msg_len, byteorder="big")
        if msg_len <= 0 or msg_len > Connection.MAX_MSG_SIZE:
            raise RuntimeError("invalid msg len: %d" % msg_len)

        retry, pos = 0, 0
        if msg_len > len(self.recv_buff):
            self.recv_buff = bytearray(msg_len)
        raw_msg = memoryview(self.recv_buff)[:msg_len]
        while retry < CONFIG.socket_retry_times and pos < msg_len:
            pos += self.sock.recv_into(raw_msg[pos:], msg_len - pos)
            retry += 1

        if pos != msg_len:
            raise RuntimeError("failed to recv message: msg len %d recv len %d, retry %d" % (
                msg_len, pos, retry))

        return raw_msg




    

        


    
