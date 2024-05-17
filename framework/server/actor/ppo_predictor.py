#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.server.actor.on_policy_predictor import OnPolicyPredictor
from framework.server.actor.on_policy_predictor_pipeline import OnPolicyPredictor_Pipeline
from framework.common.config.config_control import CONFIG
'''
定义了预测网络的input_tensors的结构
'''

class PPOPredictor(OnPolicyPredictor):
    def __init__(self, send_server, recv_server):
        super().__init__(send_server, recv_server, 'ppo')
