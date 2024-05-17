#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Network:
    def __init__(self):
        self._output_tensors = None

    def __getattr__(self, item):
        """
        在通过build_network创建tensorflow网络图结构, key是字符串, value是返回的Tensor, 而使用kears模型reload时, 不需要调
        用build_network, 因此无法访问到由build_network构造的tensor成员变量, 但是需要使用到自定的tensor的名字key, 此时通
        过函数返回值形式为(dict{key: None})
        """
        if item in self.__dict__:
            return self.__dict__[item]
        return None

    def build_network(self, input_tensors):
        """
        创建tensorflow网络图结构
        :return 返回一个dict, key是字符串, value是返回的Tensor
        """
        raise NotImplementedError

    def extra_tensors(self):
        """
        可通过继承实现该接口返回其他需要的tensors, 例如, LSTM的隐状态
        该接口默认返回空字典, 即代表没有需要额外返回的tensors
        Args:
        Returns (dict{key: tensor}): key是用户自定义的tensor的名字, tensor是Tensorflow Tensor类型
        """
        return {}


class RLNetwork(Network):
    def __init__(self, state_space, action_space):
        """
        强化学习网络
        :param state_space: 返回一个字典类型, key是字符串, 标识输入字段, value是一个ArraySpec对象, 标识tensor的数据
        :param action_space: 返回一个字典类型, key是字符串, 标识动作字段, value是一个ArraySpec对象, 标识tensor的数据
        """
        self._state_space = state_space
        self._action_space = action_space
        super().__init__()

    def as_p(self):
        """
        policy对应的输出
        :return: list of Tensor, 每项对应一个head输出
        离散动作： 每项对应一个head输出, 每个head输出代表一个action对应的概率的Tensor
        连续动作： 每项的shape为(?, 2). 对应一个head输出, 每个head输出代表每一维action对应的正态分布的mean, logstd
        """
        raise NotImplementedError

    def as_v(self):
        """
        value对应的输出
        :return: float Tensor
        """
        raise NotImplementedError

    def as_q(self):
        """
        action value(Q function)对应的输出
        :return: list of Tensor, 每项对应一个head输出, 每个head输出代表一个action对应的q值的Tensor
        """
        raise NotImplementedError

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space
