#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import queue
import time
import unittest
import argparse
from framework.common.ipc.zmq_util import ZmqClient
from framework.common.ipc.zmq_util import ZmqReader
from framework.common.ipc.zmq_util import ZmqServer
from framework.common.ipc.zmq_util import ZmqWriter
from framework.common.config.config_control import CONFIG

class TestZmqRW(unittest.TestCase):
    def test_readable(self):
        q = ZmqReader(31000, bind=True)
        p = ZmqWriter(31000)
        self.assertFalse(q.readable())

        p.put('x'*1024)
        time.sleep(0.1)
        self.assertTrue(q.readable())

    def test_writable(self):
        q = ZmqReader(31000, bind=True)
        p = ZmqWriter(31000)

        with self.assertRaises(queue.Full):
            for i in range(3000):
                p.put_nowait("x" * 4096)

    def test_get(self):
        q = ZmqReader(31000)

        with self.assertRaises(queue.Empty):
            q.get_nowait()

        start = time.time()
        with self.assertRaises(queue.Empty):
            q.get(timeout=1)
        end = time.time()
        self.assertAlmostEqual(end - start, 1.0, delta=0.01)

        p = ZmqWriter(31000, bind=True)
        p.put('x'*1024)
        self.assertEqual(q.get(), 'x'*1024)

        p.put(b'x' * 1024, binary=True)
        self.assertEqual(q.get(binary=True), b'x' * 1024)

    def test_put(self):
        q = ZmqWriter(31000, bind=True)

        with self.assertRaises(queue.Full):
            for _ in range(1024):
                q.put_nowait('x'*1024)

        start = time.time()
        with self.assertRaises(queue.Full):
            q.put('x'*1024, timeout=1)
        end = time.time()
        self.assertAlmostEqual(end - start, 1.0, delta=0.01)

        p = ZmqReader(31000)
        q.put('x'*1024)
        self.assertEqual(p.get(), 'x'*1024)

        q.put(b'x' * 1024, binary=True)
        self.assertEqual(p.get(binary=True), b'x' * 1024)


class TestZmqCS(unittest.TestCase):
    def test_readable_and_writable(self):
        svr = ZmqServer('127.0.0.1', 31000)
        svr.bind()
        cli = ZmqClient('client', '127.0.0.1',31000)
        cli.connect()

        self.assertTrue(cli.writable())
        cli.send("hello")
        self.assertFalse(cli.writable())

        time.sleep(0.1)

        self.assertTrue(svr.readable())
        client_id, _ = svr.recv()
        self.assertFalse(svr.readable())

        self.assertTrue(svr.writable())
        svr.send(client_id, "echo")
        self.assertTrue(svr.writable())

        time.sleep(0.1)

        self.assertTrue(cli.readable())
        cli.recv()
        self.assertFalse(cli.readable())

    def test_send_recv(self):
        svr = ZmqServer('127.0.0.1', 31000)
        svr.bind()

        cli_0 = ZmqClient('client_0', '127.0.0.1', 31000)
        cli_0.connect()
        cli_1 = ZmqClient('client_1', '127.0.0.1', 31000)
        cli_1.connect()

        cli_0.send("greeting from client_0")
        cli_1.send("greeting from client_1")

        client_id, data = svr.recv()
        self.assertEqual(data, f"greeting from {client_id}")
        svr.send(client_id, f"reply to {client_id}")

        client_id, data = svr.recv()
        self.assertEqual(data, f"greeting from {client_id}")
        svr.send(client_id, f"reply to {client_id}")

        data = cli_0.recv()
        self.assertEqual(data, f"reply to client_0")

        data = cli_1.recv()
        self.assertEqual(data, f"reply to client_1")

    def test_binary_send_and_recv(self):
        svr = ZmqServer('127.0.0.1', 31000)
        svr.bind()

        cli_0 = ZmqClient('client_0', '127.0.0.1', 31000)
        cli_0.connect()
        cli_1 = ZmqClient('client_1', '127.0.0.1', 31000)
        cli_1.connect()

        cli_0.send(b"greeting from client_0", binary=True)
        cli_1.send(b"greeting from client_1", binary=True)

        client_id, data = svr.recv(binary=True)
        self.assertEqual(data, bytes(f"greeting from {client_id}", encoding='utf8'))
        svr.send(client_id, bytes(f"reply to {client_id}", encoding='utf8'), binary=True)

        client_id, data = svr.recv(binary=True)
        self.assertEqual(data, bytes(f"greeting from {client_id}", encoding='utf8'))
        svr.send(client_id, bytes(f"reply to {client_id}", encoding='utf8'), binary=True)

        data = cli_0.recv(binary=True)
        self.assertEqual(data, bytes("reply to client_0", encoding='utf8'))

        data = cli_1.recv(binary=True)
        self.assertEqual(data, bytes("reply to client_1", encoding='utf8'))

    def test_svr_nowait(self):
        svr = ZmqServer('127.0.0.1', 31000)
        svr.bind()

        with self.assertRaises(queue.Empty):
            svr.recv_nowait()

    def test_cli_nowait(self):
        cli = ZmqClient("client_id", '127.0.0.1', 31000)
        cli.connect()

        with self.assertRaises(queue.Empty):
            cli.recv_nowait()

        cli.send_nowait("x"*1024)

        with self.assertRaises(queue.Full):
            cli.send_nowait("x" * 1024)

    def test_svr_timeout(self):
        svr = ZmqServer('127.0.0.1', 31000)
        svr.bind()

        start = time.monotonic()
        with self.assertRaises(queue.Empty):
            svr.recv(timeout=1)
        end = time.monotonic()

        self.assertAlmostEqual(1.0, end - start, delta=0.01)

    def test_cli_timeout(self):
        cli = ZmqClient("client_id", '127.0.0.1', 31000)
        cli.connect()

        start = time.monotonic()
        with self.assertRaises(queue.Empty):
            cli.recv(timeout=1)
        end = time.monotonic()

        self.assertAlmostEqual(1.0, end - start, delta=0.01)

        cli.send_nowait("x" * 1024)

        start = time.monotonic()
        with self.assertRaises(queue.Full):
            cli.send("x" * 1024, timeout=1)
        end = time.monotonic()

        self.assertAlmostEqual(1.0, end - start, delta=0.01)
    
    def test_client(self):
        self.zmq_client = ZmqClient("client_train_one0", "127.0.0.1", 8888)
        self.zmq_client.connect()
        print('zmq_client')

        if self.zmq_client.writable():
            print("self.zmq_client.writable() is yes")
    
    def test_server(self):
        self.zmq_server = ZmqServer('127.0.0.1', 8888)
        self.zmq_server.bind()
        print('zmq_server')

        while True:
            if self.zmq_server.readable():
                print("self.zmq_server.readable() is yes")

class TestZmqServerClient():
    def __init__(self) -> None:
        self.host = '127.0.0.1'
        self.port = '8888'
        self.client_id = 'client_id'

    def server(self):
        zmq_server = ZmqServer(self.host, self.port)
        zmq_server.bind()
        while True:
            try:
                client_id, data = zmq_server.recv()
                if data:
                    print('server recv data success')
                    zmq_server.send(client_id, data)
                    print('server send data success')
            except Exception as e:
                pass
        
    def client(self):
        zmq_client = ZmqClient(self.client_id, self.host, self.port)
        zmq_client.connect()
        while True:
            try:
                data = {'message' : 'hello'}
                zmq_client.send(data)
                print(f'client send success')
                
                data = zmq_client.recv(block=False, binary=True)
                print(f'client recv success')
            except Exception as e:
                pass


'''
注意unittest和下面的测试是冲突的, 故按照需要注释
'''
if __name__ == '__main__':
    unittest.main()

    test_zmq_server_client = TestZmqServerClient()
    configure_file = '/data/projects/gorge_walk_v1/conf/framework/actor.toml'
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_actor_configure()

    parser = argparse.ArgumentParser(description='Example script with command line arguments')
    parser.add_argument('--model', type=str, help='model, server or client')
    args = parser.parse_args()
    if args.model == 'server':
        test_zmq_server_client.server()
    elif args.model == 'client':
        test_zmq_server_client.client()
    else:
        print(f'not supported model {args.model}')