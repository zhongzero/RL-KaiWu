#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from abc import ABCMeta

class ReplayBufferBase(metaclass=ABCMeta):

        def __init__(self, data_spec, capacity):
            self._data_spec = data_spec
            self._capacity = capacity
        
        def init(self):
            raise NotImplementedError
        
        @property
        def data_spec(self):
            return self._data_spec
        
        @property
        def capacity(self):
            return self._capacity
        
        def total_size(self, *args, **kwargs) -> any:
            raise NotImplementedError

        def add_batch(self, *args, **kwargs):
            raise NotImplementedError

        def as_dataset(self, *args, **kwargs):
            raise NotImplementedError

        def gather_all(self, *args, **kwargs):
            raise NotImplementedError

        def clear(self, *args, **kwargs) -> any:
            raise NotImplementedError