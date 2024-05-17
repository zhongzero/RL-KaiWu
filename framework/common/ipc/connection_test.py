#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
import socket
import unittest
from unittest.mock import patch
from unittest import mock
from framework.common.ipc.connection import Connection
from framework.common.config.config_control import CONFIG

class ConnectionTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # set configure file 
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/configure.toml')
        CONFIG.parse_main_configure()

        self.sock = mock.create_autospec(socket.socket, instance=True)
        self.conn = Connection(self.sock)
        self.msg = b'a' * 1000

    def test_send_msg(self):
        self.sock.configure_mock(**{'send.return_value': 1012})
        self.conn.send_msg(self.msg)
        self.sock.send.assert_called()

    def test_send_receive_msg(self):
        self.new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect
        try:
            self.new_sock.connect(("9.134.253.89", 8000))
        except socket.error as e:
            print("socke errr: %s" % e)
        
        msg_len = 1000
        self.send_header_buff = bytearray(Connection.HEADER_TOTAL_LEN)
        self.send_header_buff[0:Connection.MAGIC_LEN] = Connection.MAGIC_NUM.to_bytes(Connection.MAGIC_LEN, byteorder='big')
        self.send_header_buff[Connection.MAGIC_LEN:Connection.HEADER_TOTAL_LEN] = msg_len.to_bytes(Connection.DATA_LEN, byteorder='big')

        magic_num = int.from_bytes(self.send_header_buff[0:Connection.MAGIC_LEN], byteorder='big')
        print("magic_num is " + str(magic_num))

        msg_len = int.from_bytes(self.send_header_buff[Connection.MAGIC_LEN:Connection.HEADER_TOTAL_LEN], byteorder='big')
        print("msg_len is " + str(msg_len))
        
        for i in range(1):
            # send
            print(self.new_sock.send(self.send_header_buff, Connection.HEADER_TOTAL_LEN))
            print(self.new_sock.send(self.msg, msg_len))

            #recev
            rec_msg = self.new_sock.recv(1024)
            print("rec_msg " + str(rec_msg))



        # close
        self.new_sock.close()

if __name__ == '__main__':
    unittest.main()
