#!/usr/bin/env python3
# -*- coding:utf-8 -*-


class Policy:
    def send_pred_data(self, client_conn_id, pred_data, agent_ctx):
        """
        发送预测数据
        :param client_conn_id: string类型, 用于标识客户端
        :param pred_data: 字典类型，包含所有预测的输入数据
        :param agent_ctx: Context对象, 包含aisrv相关的上下文
        """
        raise NotImplementedError

    def get_pred_result(self, client_conn_id, agent_ctx):
        """
        获取预测结果
        :param client_conn_id: string类型, 用于标识客户端
        :param agent_ctx: Context对象, 包含aisrv相关的上下文
        :return: pred_result: dict对象, 包含预测的结果
        """
        raise NotImplementedError

    def need_train(self):
        """
        :return: bool值, 返回当前policy是否需要训练
        """
        raise NotImplementedError

    def send_train_data(self, client_conn_id, train_data, agent_ctx):
        """
        发送训练数据, 数据流为kaiwu_rl_helper->learner_proxy->learner
        :param client_conn_id: string类型, 用于标识客户端
        :param train_data: 字典类型，包含所有训练的输入数据
        :param agent_ctx: Context对象, 包含aisrv相关的上下文
        """
        raise NotImplementedError
    
    def send_train_data_to_sample_server(self, client_conn_id, train_data, agent_ctx):
        """
        发送训练数据, 数据流为kaiwu_rl_helper->sample_server->learner_proxy->learner
        :param client_conn_id: string类型, 用于标识客户端
        :param train_data: 字典类型，包含所有训练的输入数据
        :param agent_ctx: Context对象, 包含aisrv相关的上下文
        """
        raise NotImplementedError

    def stop(self):
        """
        停止当前Policy, 当policy内部开启了线程/进程时, kaiwu_rl_helper可以通过这个函数停止policy。
        """


class PolicyBuilder:
    def __init__(self, policy_name, simu_ctx):
        """
        :param policy_name: policy的名称
        """
        self._policy_name = policy_name
        self._simu_ctx = simu_ctx

    def build(self):
        """
        创建一个policy
        :return policy: Policy实例
        """
        raise NotImplementedError
