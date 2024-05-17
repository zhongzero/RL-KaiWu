#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import json
import unittest
import struct
from framework.common.utils.common_func import *
import sys
from framework.common.config.config_control import CONFIG
from framework.common.utils.singleton import Singleton

#send samples
g_send_samples = {}

@Singleton
class A(object):
    x = 1

    def __init__(self) -> None:
        pass

class CommonFuncTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_timeit(self):
        with TimeIt() as it:
            time.sleep(1)
        
        print(it.interval)
        self.assertAlmostEqual(it.interval, 1.0, places=1)

        start_time = time.time()
        with TimeIt() as it:
            for i in range(10000):
                print(i)
                print(f'cost1: {time.time() - start_time}')

        start_time = time.time()
        for i in range(10000):
            print(i)
        print(f'cost2: {time.time() - start_time}')
        
    def test_context(self):
        self.context = Context()
        self.context.a = 1
        self.context.b = 2

        print(str(self.context.a) + ' ' + str(self.context.b))

        self.context.a = 2
        self.context.b = 1

        print(str(self.context.a) + ' ' + str(self.context.b))
    
    def test_Singleton(self):
        x1 = A()
        x2 = A()

        print("x1 " + str(x1) + " x2 " + str(x2))
    
    def test_hashlib(self):
        data = "agentid-1"
        print(hashlib_md5(data))
    
    def test_get_local_rank(self):
        print(f"get_local_rank: {get_local_rank()}")
    
    def test_uuid(self):
        print(get_uuid())

        print(get_uuid())
    
    def test_schedule(self):
        
        def run():
            print('hello word')

        set_schedule_event(1, run)
    
    '''
    采用time.sleep(0)让出CPU
    '''
    def test_sleep(self):
        return
        while True:
            print("check point")
            time.sleep(0)
    
    '''
    测试根据函数名字返回函数内容
    '''
    def test_get_content_by_name(self):
        print(get_fun_content_by_name(get_fun_content_by_name))

        print(get_fun_content_by_name('get_fun_content_by_name'))

    def test_insert_any_string(self):
        checkpoint_path = '/data/ckpt/sgame_ppo/models/checkpoint'
        file_data = ''
        to_insert_str = 'models/'
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                line  = insert_any_string(line, to_insert_str, 'model.ckpt', 'before')
                file_data += line

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
                f.write(file_data)
    
    def test_list_eq(self):
        listA = []
        listB = [1, 2]
        print(is_list_eq(listA, listB))

        listA = [1, 2, 3]
        listB = [1, 2, 3]
        print(is_list_eq(listA, listB))

        listA = [1, 2]
        listB = [1, 2, 3]
        print(is_list_eq(listA, listB))
    
    def test_get_any_time_by_now(self):
        print(get_any_time_from_now(1, ways='day'))
        print(get_any_time_from_now(1, ways='hour'))
        print(get_any_time_from_now(1, ways='min'))
        print(get_any_time_from_now(10, ways='second'))

        print(get_any_time_from_now(10, ways='seconds'))
    
    def test_men_and_get(self):
        a = [1, 2, 3]
        mean_value, max_value = get_mean_and_max(a)
        print(f'mean is {mean_value}, max is {max_value}')
    
    def test_md5sum(self):
        file_name = '/data/ckpt/sgame_5v5_ppo/convert_models_actor/trt_weights.wts2'
        start = time.time()
        print(f'start at {start}')
        md5sum(file_name)
        print(f'cost time is {time.time() - start}')
    
    def test_compress_decompress(self):
        import lz4.frame
        import lz4.block
        import lz4.stream
        import numpy as np

        input_data =  20 * 128 * np.random.rand(5 * 32 * (12677 + 2048 + 6 * 17 * 17))

        # 采用lz4.frame
        print(f'lz4.frame compress start at {time.time()}')
        compress_msg = lz4.frame.compress(input_data)
        print(f'lz4.frame compress end at {time.time()}')
        print(f'lz4.frame compress_msg size is {len(compress_msg)}')

        print(f'lz4.frame decompress start at {time.time()}')
        decompressed = lz4.frame.decompress(compress_msg)
        print(f'lz4.frame decompress start at {time.time()}')
        print(f'lz4.frame decompressed size is {len(decompressed)}')

        # 采用lz4.block
        print(f'lz4.block compress start at {time.time()}')
        compress_msg = lz4.block.compress(input_data, mode='fast', store_size=False)
        print(f'lz4.block compress end at {time.time()}')
        print(f'lz4.block compress_msg size is {len(compress_msg)}')

        print(f'lz4.block decompress start at {time.time()}')
        decompressed = lz4.block.decompress(compress_msg, return_bytearray=False, uncompressed_size=314572800)
        print(f'lz4.block decompress start at {time.time()}')
        print(f'lz4.block decompressed size is {len(decompressed)}')

        # 采用lz4.stream

        # 采用pyzstd
        import zstd
        print(f'pyzstd compress start at {time.time()}')
        compress_msg = zstd.compress(input_data, 1)
        print(f'pyzstd compress end at {time.time()}')
        print(f'pyzstd compress_msg size is {len(compress_msg)}')

        print(f'pyzstd decompress start at {time.time()}')
        decompressed =  zstd.decompress(compress_msg)
        print(f'pyzstddecompress start at {time.time()}')
        print(f'pyzstd decompressed size is {len(decompressed)}')



if __name__ == '__main__':
    unittest.main()