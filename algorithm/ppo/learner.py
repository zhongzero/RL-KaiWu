#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :1v1
@File    :learner.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

from framework.common.utils.tf_utils import *
from framework.server.learner.on_policy_trainer import OnPolicyTrainer
from framework.common.config.config_control import CONFIG


class Sgame1V1PPOTrainer(OnPolicyTrainer):
    """
    ppo trainer

    PPO算法训练器
    """
    def __init__(self):
        super(Sgame1V1PPOTrainer, self).__init__(name='ppo')
    
    def init(self):
        super().init()

    @property
    def tensor_names(self):
        """
        Sgame has an attribute called input_datas which sets the key for data in the reverb table. 
        When sending samples, each sample is a dictionary with keys as tensor_names.

        sgame的设置中有个属性为input_datas,主要是设置reverb表中数据的key
        发送样本时, 每条样本为一个dict, key为tensor_names
        Returns:
            _type_: _description_
        """

        names = []
        names.append('input_datas')
        return names

    @property
    def tensor_dtypes(self):
        """
        Specify the sample type.

        设置样本的类型
        """
        dtypes = []
        dtypes.append(tf.float16)
        return dtypes

    @property
    def tensor_shapes(self):
        """
        Specify the sample shape.

        设置样本的shape
        """
        shapes = []
        shapes.append(tf.TensorShape((64, 15552)))

        return shapes
