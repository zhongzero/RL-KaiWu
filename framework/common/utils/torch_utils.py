#!/usr/bin/env python3
# -*- coding:utf-8 -*-



'''
与PyTorch相关的公共函数
'''

from framework.common.config.config_control import  CONFIG
import torch

'''
判断机器上GPU是否安装成功
'''
def is_gpu_available():
    return torch.cuda.is_available()

'''
设置运行的GPU卡
'''
def set_runtime_gpu():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

'''
获取GPU的型号
'''
def get_gpu_info():
    return torch.cuda.get_device_name()

'''
释放pytorch占用的显存
'''
def release_cache():
    torch.cuda.empty_cache()

'''
判断某个model文件是合理的
'''
def pytorch_model_file_valid(model_path):
    if not model_path:
        return False
    try:
        # 加载模型
        model = torch.load(model_path)
        return True
    except Exception as e:
        return False
    
'''
编译Torch脚本为本机代码
'''
def torch_compile_func(func_name):
    if not func_name:
        return None
    
    compiled_func_name = torch.jit.compile(func_name)
    return compiled_func_name