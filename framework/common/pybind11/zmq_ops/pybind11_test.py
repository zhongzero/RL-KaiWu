#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import unittest
import struct
from framework.common.utils.common_func import TimeIt
import sys
import numpy as np

# use different dir
sys.path.append("../pybind11/")

# 需要配置加载pybind11
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir + '/../../common/pybind11/zmq_ops')


import argparse
import time
import numpy as np
import sys
import tqdm
import os

import zmq
import tensorflow as tf
from framework.common.utils.tf_utils import *
from framework.common.pybind11.zmq_ops.zmq_ops import *

#send samples
g_send_samples = {}

class Pybind11Test():
    def setUp(self) -> None:
        pass

    '''
    验证是否能正常引入libzmqop
    '''
    def test_import_libzmqop(self):
        import libzmqop

    def test_zmq_ops(self):
        self.send()
        self.recv()

    def send(self):
        """ We use float32 data to pressure the system. In practice you'd use uint8 images."""
        data = [
            np.random.rand(64, 224, 224, 3).astype('float32'),
            (np.random.rand(64)*100).astype('int32')
        ]   # 37MB per data
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUSH)
        socket.set_hwm(50)
        socket.connect(f'tcp://127.0.0.1:8888')

        try:
            while True:
                socket.send(dump_arrays(data), copy=False)
                print("data is send ")
        finally:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            if not ctx.closed:
                ctx.destroy(0)
            sys.exit()

    def recv(self):
        sock = ZMQPullSocket(f'tcp://127.0.0.1:8888', [tf.float32, tf.int32], 50)

        fetches = []
        for k in range(8):  # 8 GPUs pulling together in one sess.run call
            fetches.extend(sock.pull())
        fetch_op = tf.group(*fetches)

        with tf.compat.v1.Session() as sess:
            while True:
                sess.run(fetch_op)
                print("data is recive ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['send', 'recv'])
    parser.add_argument('--hwm', type=int, default=200)
    args = parser.parse_args()

    Pybind11Test = Pybind11Test()

    if args.task == 'send':
        Pybind11Test.send()
    elif args.task == 'recv':
        Pybind11Test.recv()
