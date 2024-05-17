#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
import unittest
from framework.common.utils.torch_utils import is_gpu_available, compiled_func_name

class TFUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    # 测试torch compile func
    def test_torch_compile_func(self):

        @torch.jit.script
        def add(a, b):
            return a + b
        
        compiled_func_name = compiled_func_name(add)
        print(compiled_func_name(1, 2))

if __name__ == '__main__':
    unittest.main()