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
import kaiwu

#from framework.common.pybind11.zmq_ops.zmq_ops import dump_arrays, ZMQPullSocket
import argparse
import time
import numpy as np
import sys
import tqdm

import zmq
import tensorflow as tf


#send samples
g_send_samples = {}

PIPE = 'ipc://@testpipe'

class Pybind11Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    #struct pack
    def test_struct_pack(self):

        #time.sleep(1)
        # Magic + DataLen + Data
        magic = 0x12345678
        data_length = 8192
        data = 'b' * 8192
        with TimeIt() as ti:
            for i in range(0, 10):
                send_sample = struct.pack("<I", i) + struct.pack("<I", magic) + struct.pack("<I", data_length) + struct.pack("<8192s", data.encode("utf-8"))
                g_send_samples[i] = send_sample
        
        print("struct python pack time cost: " + str((ti.interval * 1000)) + "s")
    
    #struct unpack
    def test_struct_unpack(self):

        with TimeIt() as ti:
            for i in range(0, 10):
                send_sample = g_send_samples.get(i)
                idx = struct.unpack("I", memoryview(send_sample[0:4]))
                magic = struct.unpack("I", memoryview(send_sample[4:8]))
                data_length = struct.unpack("I", memoryview(send_sample[8:12]))
                data = struct.unpack("8192s", memoryview(send_sample[12:]))

        print("struct python unpack time cost: " + str((ti.interval * 1000)) + "s")


    # C++ + python pack
    def test_cpp_pack(self):

        magic = 0x12345678
        data_length = 8192
        data = 'b' * 8192
        with TimeIt() as ti:
            for i in range(0, 10):
                kaiwu.Pack(i, magic, data_length, data)
        print("struct C++ pack time cost: " + str((ti.interval * 1000)) + "s")


    # C++ + python unpack
    def test_cpp_unpack(self):

        with TimeIt() as ti:
            for i in range(0, 10):
                kaiwu.UnPack()
        print("struct C++ unpack time cost: " + str((ti.interval * 1000)) + "s")


    def test_pybind11(self):
        # func
        print("kaiwu.Add: %d"%(kaiwu.Add(1, 2)))

        # class
        x = kaiwu.KaiWu("kaiwu")
        print("name: " + str(x.getName()))

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
        socket.connect(PIPE)

        try:
            while True:
                socket.send(dump_arrays(data), copy=False)
        finally:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            if not ctx.closed:
                ctx.destroy(0)
            sys.exit()

    def recv(self):
        sock = ZMQPullSocket(PIPE, [tf.float32, tf.int32], 50)
        fetches = []
        for k in range(8):  # 8 GPUs pulling together in one sess.run call
            fetches.extend(sock.pull())
        fetch_op = tf.group(*fetches)

        with tf.Session() as sess:
            while True:
                sess.run(fetch_op)
    
    # 测试耗时
    def test_list2Vector(self):

        bs = 128
        b = np.random.rand(bs*5*(12667+2048))
        list = b.tolist()

        with TimeIt() as ti:
            kaiwu.list2Vector(list, bs)
        
        print(f'list2Vector cos {ti.interval} ms')

    # 测试耗时
    def test_numpy2Array(self):

        bs = 128
        b = np.random.rand(bs*5*(12667+2048))

        with TimeIt() as ti:
            kaiwu.numpy2Array(b, bs)
        
        print(f'numpy2Array cos {ti.interval} ms')

if __name__ == '__main__':
    unittest.main()

