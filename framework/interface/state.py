#!/usr/bin/env python3
# -*- coding:utf-8 -*-


class State:
    """
    状态对象，主要用于业务相关的协议解析和预处理接口，主要包括：
    """

    def get_state(self):
        """
        从原始数据包中获取模型需要的状态数据，并做必要的状态处理
        该操作可能会被调用多次，如果开销比较大，建议缓存结果
        :return 返回一个字典类型，
        key是字符串, 标识输入字段
        value是numpy.array类型
        """
        raise NotImplementedError

    @staticmethod
    def state_space():
        """
        返回状态空间
        :return 返回一个字典类型，
        key是字符串, 标识输入字段
        value是一个ArraySpec对象, 标识tensor的数据
        一般来说如果是DNN的话, 这里返回一个单元素的list
        对于CNN来说这里返回一个3个元素的list
        """
        raise NotImplementedError