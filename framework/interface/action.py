#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ActionSpec:
    """
    动作标准的描述, 为action_space函数的返回值, 主要描述了action的以下几个特征:
    1. shape: 动作值的shape
    2. dtype: 动作值的dtype
    3. pdclass: 动作值的概率密度分布, 是framework.common.algo.distribution.Pd的一个子类
    比如CategoricalDist和GaussianDist
    """

    def __init__(self, array_spec, pdclass):
        """
        :param array_spec: ArraySpec对象
        :param pdclass: framework.common.algo.distribution.Pd的一个子类
        """
        self._array_spec = array_spec
        self._pdclass = pdclass

    @property
    def shape(self):
        return self._array_spec.shape

    @property
    def dtype(self):
        return self._array_spec.dtype

    @property
    def pdclass(self):
        return self._pdclass


class Action:
    """
    动作对象，主要用于业务相关的协议解析和预处理接口，主要包括：
    """

    def get_action(self):
        """
        获得Action对象对应的结果
        该操作可能会被调用多次，如果开销比较大，建议缓存结果
        :return 返回一个字典类型，
        key是字符串, 标识输入字段
        value是一个标识具体动作的对象, 一般来说是一个int
        """
        raise NotImplementedError

    @staticmethod
    def action_space():
        """
        返回动作空间:
        :return 返回一个字典类型，
        key是字符串, 标识输入字段
        value是一个ActionSpec对象, 标识tensor的数据
        对于离散动作：
            可以是一个类别, 比如atari上的19个动作, 那么动作空间就是{'a': (19,)}
            可以是多个类别, 比如方向有8个, 技能有4个, 那么动作空间就是{'move': (8,), 'skill': (4,)}
        对于连续动作
            从网络输出的一般为正态分布的均值与方差, 但动作是通过使用均值与方差抽样得到的一个具体的值。所以shape为(1,)
            可以是一个类别, 比如moutain car 是连续控制，那么动作空间就是{'a': (1,)},
            可以是多个类别，比如有前后控制 和 左右控制两个维度， 那么动作空间就是{'front': (1,), 'left', (1,)}
        """
        raise NotImplementedError
