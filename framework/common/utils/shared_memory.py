#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import ctypes
from multiprocessing import sharedctypes
import os


'''
共享内存库
'''
class SharedMemory(object):
    def __init__(self, type, size) -> None:

        self.size = size
        self.array =  sharedctypes.RawArray(type, self.size)

    def check_idx_valid(self, idx):
        if 0 < idx or idx > self.size:
            return False
        
        return True

    def put(self, idx, data):
        if not self.check_idx_valid(idx) or not data:
            return

        self.array[idx] = data

    def get(self, idx):
        if not self.check_idx_valid(idx) :
            return
        
        return self.array[idx]

    def size(self):
        return self.size