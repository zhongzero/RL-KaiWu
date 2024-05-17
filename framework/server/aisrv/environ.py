#!/usr/bin/env python
# -*- coding: utf-8 -*-


import abc

__all__ = [
    'Environ'
]


class Environ:
    def __init__(self, simu_ctx):
        """
        :param simu_ctx 包含kaiwu_rl_helper生命周期有效的context信息
        """
        self._simu_ctx = simu_ctx

    @abc.abstractmethod
    def init(self,**kwargs):
        """
        完成Envrion的初始化工作, 并将环境的静态信息返回
        :return: 字典类型, 解析后的json对象
        """

    @abc.abstractmethod
    def reset(self):
        """
        处理episode开始时的逻辑, 返回episode的第一个state对象返回
        :return: new_states: dict类型, key是准备好的agent的agent_id, value是proto.State的子类, 下一个状态s_t+1
        :return: ex_rewards: dict类型, key是准备好的agent的agent_id, value是float数组, 包含了从环境中返回的各种reward信号, r_t
        :return: dones: dict类型, key是准备好的agent的agent_id, value是bool类型, terminal_t+1
                           "__all__"这个特殊的key是必须的, 用于指示整个环境是否结束
        """

    @abc.abstractmethod
    def step(self, actions, extra_info=None):
        """
        把action发给envrion, 然后从envrion中获取新的信息,
        注意: step返回的状态必须是有效帧, 无效帧可以在environ中直接过滤掉
        也可以在environ中把多帧数据组装为一帧的状态数据返回

        :param actions: dict类型, key是准备好的agent的agent_id, value是proto.Action子类
        :param extra_info: dict类型, key是准备好的agent的agent_id, value是network.extra_tensors()
            返回的dict{key: tensor}结构
        :return: new_states: dict类型, key是准备好的agent的agent_id, value是proto.State的子类, 下一个状态s_t+1
        :return: ex_rewards: dict类型, key是准备好的agent的agent_id, value是float数组, 包含了从环境中返回的各种reward信号, r_t
        :return: dones: dict类型, key是准备好的agent的agent_id, value是bool类型, terminal_t+1
                           "__all__"这个特殊的key是必须的, 用于指示整个环境是否结束
        :raise ClientQuitException标识客户端主动关闭连接
        """

    @abc.abstractmethod
    def reject(self, e):
        """
        拒绝env的请求
        :param e: Exception子类, 框架中发生的异常情况
        """

    @abc.abstractmethod
    def finsh(self):
        """
        完成Envrion的销毁工作
        """

    @property
    @abc.abstractmethod
    def client_id(self):
        """
        返回标识客户端信息的信息, 主要用来定位客户端实际的IP地址, 方便调试
        :return: 字符串
        """

    @property
    @abc.abstractmethod
    def client_version(self):
        """
        返回标识客户端版本的信息
        :return: 字符串
        """

    @property
    @abc.abstractmethod
    def ep_id(self):
        """
        返回当前ep_id, 用于唯一标识当前局
        :return:
        """

    @abc.abstractmethod
    def policy_mapping_fn(self, agent_id):
        """
        将agent映射到对应的policy名字
        :param agent_id: int类型, 标识唯一的agent
        :return:
        """
