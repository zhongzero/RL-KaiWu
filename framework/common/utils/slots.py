#!/usr/bin/env python
# encoding: utf-8


from multiprocessing import Array
from multiprocessing import Lock
from multiprocessing import Queue
from multiprocessing import Pipe
import sys


class _Queue():
    __slots__=("input_pipe", "output_pipe")

    def __init__(self, maxsize):
        #self.queue = Queue(maxsize)
        self.input_pipe, self.output_pipe=Pipe(duplex=False)

    def recv(self):
        return self.input_pipe.recv()
        #return self.queue.get()

    def send(self, obj):
        return self.output_pipe.send(obj)
        #return self.queue.put(obj)

    def poll(self, timeout=0):
        if timeout == 0:
            return self.input_pipe.poll()
            #return self.queue._poll()
        else:
            return self.input_pipe.poll(timeout)
            #return self.queue._poll(timeout)


'''
因为会使用到multiprocessing.SemLock, 故需要修改ulimit -n 10000, 其配置值是CONFIG.max_tcp_count, 其配置的值是多少, 则需要ulimit -n修改成该值
'''
class Slots:
    __slots__=("lock", "slot_num", "slots", "pipes", "max_buf")
    
    def __init__(self, slot_num, max_buf=0):
        self.lock = Lock()
        self.slot_num = slot_num
        self.slots = Array('i', self.slot_num, lock=False)
        self.pipes = {}
        self.max_buf = max_buf

    def register_group(self, group_name):
        self.pipes[group_name] = [_Queue(self.max_buf) for _ in range(self.slot_num)]

    def get_slot(self):
        with self.lock:
            for i in range(len(self.slots)):
                if self.slots[i] == 0:
                    # skip expired data
                    for group_name in self.pipes:
                        input_pipe = self.get_input_pipe(group_name, i)
                        while input_pipe.poll():
                            input_pipe.recv()
                    self.slots[i] = 1
                    return i
            raise RuntimeError("can't find empty slots")

    def used_slot(self):
        cnt = 0
        with self.lock:
            for i in range(len(self.slots)):
                if self.slots[i] == 1:
                    cnt += 1
            return cnt

    def put_slot(self, i):
        with self.lock:
            self.slots[i] = 0

    def get_input_pipe(self, group_name, i):
        return self.pipes[group_name][i]

    def get_output_pipe(self, group_name, i):
        return self.pipes[group_name][i]
    
    '''
    获取最小/最大slot_id
    '''
    def get_min_max_slot_id(self):
        min_slot_id = sys.maxsize
        max_slot_id = -sys.maxsize

        for i in range(len(self.slots)):
            if self.slots[i] == 1:
                if i < min_slot_id:
                    min_slot_id = i
                if i > max_slot_id:
                    max_slot_id = i
        
        return min_slot_id, max_slot_id