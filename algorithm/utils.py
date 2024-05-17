#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Module Introduction: This module mainly includes some utility functions.
# 模块介绍：本模块主要包含一些工具函数

'''
@Project :1v1
@File    :util.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

from algorithm.infer_input_output import InferInput, InferOutput

def cvt_tensor_to_infer_input(input_tensors):
    """
    Convert tensor list to infer input list.
    
    将张量列表转换为推理输入列表
    """
    infer_input_list = []
    for inp in input_tensors:
        name = inp.name
        shape = [-1 if x is None else x for x in inp.shape.as_list()]
        dtype = inp.dtype.as_numpy_dtype
        infer_input_list.append(InferInput(name, shape, dtype))

    return infer_input_list


def cvt_tensor_to_infer_output(output_tensors):
    """
    Convert tensor list to infer output list.

    将张量列表转换为输出列表
    """
    infer_output_list = []
    for out in output_tensors:
        name = out.name
        shape = [-1 if x is None else x for x in out.shape.as_list()]
        dtype = out.dtype.as_numpy_dtype
        infer_output_list.append(InferOutput(name, shape, dtype))

    return infer_output_list
