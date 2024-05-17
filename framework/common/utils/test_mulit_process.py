#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import sys
from loguru import logger


'''
本例子主要是用于测试多进程之间的使用队列Queue方式
'''
class Client(multiprocessing.Process):
    def __init__(self):
        super(Client, self).__init__()

        self.count = 1

        self.message_queue = multiprocessing.Queue(2048)

        logger.remove(handler_id=None)
        logger.add(sys.stderr, level='DEBUG')

    def get_data(self):
        return self.message_queue.get()

    def put_data(self, data):
        self.message_queue.put(data)
    
    def run(self):
        # 获取数据
        while True:
            data = self.get_data()
            logger.debug(f'client get data {data}, count {self.count}')
            self.count += 1

if __name__ == '__main__':
    client = Client()
    client.start()

    count = 1

    logger.remove(handler_id=None)
    logger.add(sys.stderr, level='DEBUG')

    while True:
        data = 'hello'
        client.put_data(data)
        logger.debug(f"main put data, count is {count}")
        count += 1
