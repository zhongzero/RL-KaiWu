#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Module Introduction: This module includes the wrapping of training data and interface definitions.
# 模块介绍：本模块包含了训练数据的包装和接口定义

'''
@Project :1v1
@File    :infer_input_output.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

import logging


class InferData(object):
    """An object of InferData class is used to describe
    input or output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input/output whose data will be described by this object
    dims : list
        The shape of the associated input/output.
    data_type : str
        The datatype of the associated input/output.
    data: numpy array
        The data of the associated input/output.

    InferData类的对象用于描述推理请求的输入或输出张量
    参数：
        name: 描述其数据的输入/输出的名称
        list: 相关输入输出的维度
        data_type: 相关输入输出的数据类型
        data: 相关输入输出数据
    """
    def __init__(self, name, dims, data_type=None, data=None):
        self.name = name
        self.dims = dims
        self.data_type = data_type
        self.data = None
        if data is not None:
            self.set_data(data)

    def get(self):
        """Get all attributes of the tensor.

        Returns
        ------
        str
            The tensor name.
        list
            The tensor shape.
        str
            The tensor datatype.
        numpy array
            The tensor data in numpy array format.

        获取张量的所有属性
        返回值：张量的名字, 维度，数据类型以及数组格式的数据
        """
        return self.name, self.dims, self.data_type, self.data

    def get_name(self):
        """
        Get the name of the tensor.
        
        获取张量的名字
        """
        return self.name

    def set_data(self, data):
        """Set the tensor data from the specified numpy array for
        input/output associated with this object.

        Parameters
        ----------
        data : numpy array
            The tensor data in numpy array format

        Raises
        ------
        Exception
            If failed to reshape data with dims.

        使用指定的NumPy数组为与该对象关联的输入/输出设置张量数据。
        参数：
            data: numpy数组格式的张量数据
        
        异常：
            转换数据维度失败时会抛出异常
        """
        try:
            if self.data_type is not None:
                self.data = data.reshape(self.dims).astype(self.data_type)
            else:
                self.data = data.reshape(self.dims)
        except Exception:
            logging.error("can not convert data shape from {} to {}".format(
                str(data.shape), str(self.dims)))
            raise

    def get_data(self):
        """Get the tensor data in numpy array format.

        获取numpy数组格式的张量数据
        """
        return self.data


class InferInput(InferData):
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    dims : list
        The shape of the associated input.
    data_type : str
        The datatype of the associated input.
    data: numpy array
        The data of the associated input.
    
    InferInput类用于描述推理请求的输入张量。
    参数：
        name: 输入名
        dims: 相关输入维度
        data_type: 相关输入数据类型
        data: 相关输入数据
    """
    def __init__(self, name, dims, data_type=None, data=None):
        super(InferInput, self).__init__(name, dims, data_type, data)


class InferOutput(InferData):
    """An object of InferOutput class is used to describe
    output tensor for an inference request.

    InferOutput类用于描述推理请求的输出张量。
    """
    def __init__(self, name, dims, data_type=None, data=None):
        super(InferOutput, self).__init__(name, dims, data_type, data)
