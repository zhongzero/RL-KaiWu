#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :1v1
@File    :actor.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 
'''

from framework.server.actor.on_policy_predictor import OnPolicyPredictor
from framework.server.actor.on_policy_predictor_pipeline import OnPolicyPredictor_Pipeline
from framework.common.config.config_control import CONFIG


class PPOPredictor(OnPolicyPredictor):
    """
    Defines the structure of input_tensors for the prediction network.

    定义了预测网络的input_tensors的结构。
    """

    def __init__(self, send_server, recv_server):
        super().__init__(send_server, recv_server, 'ppo')
