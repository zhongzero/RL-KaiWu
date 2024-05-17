#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.common.utils.tf_utils import *
from framework.server.learner.on_policy_trainer import OnPolicyTrainer
from framework.common.config.config_control import CONFIG

'''
PPO Trainer
'''
class PPOTrainer(OnPolicyTrainer):

    def __init__(self):
        super(PPOTrainer, self).__init__(name='ppo')
    
    def init(self):
        super().init()

    @property
    def tensor_names(self):
        '''
        PPO算法需要的参数:
        old_neg_logp
        y_r
        old_vpred
        m

        下面是x和a对应的:
        x, 对应的state_spec
        a, 对应的action_space
        '''
        names = []
        names.extend(self.policy_conf.state.state_space().keys())
        names.extend(self.policy_conf.action.action_space().keys())
        names.extend([f'old_neg_logp_{name}' for name in self.policy_conf.action.action_space()])
        names.append('y_r')
        names.append('old_vpred')
        names.append('m')
        return names

    @property
    def tensor_dtypes(self):
        dtypes = []
        dtypes.extend([tf.as_dtype(array_spec.dtype)
                       for _, array_spec in self.policy_conf.state.state_space().items()])
        dtypes.extend([tf.as_dtype(array_spec.dtype)
                       for _, array_spec in self.policy_conf.action.action_space().items()])
        dtypes.extend([tf.float32 for _, array_spec in self.policy_conf.action.action_space().items()])
        dtypes.append(tf.float32)
        dtypes.append(tf.float32)
        dtypes.append(tf.float32)
        return dtypes

    @property
    def tensor_shapes(self):
        shapes = []
        shapes.extend([tf.TensorShape((None,) + array_spec.shape) if not CONFIG.use_rnn else
                    tf.TensorShape((None, None) + array_spec.shape)
                    for _, array_spec in self.policy_conf.state.state_space().items()])
        shapes.extend([tf.TensorShape((None,) + array_spec.shape[:-1]) if not CONFIG.use_rnn else
                       tf.TensorShape((None, None) + array_spec.shape[:-1])
                       for _, array_spec in self.policy_conf.action.action_space().items()])
        shapes.extend([tf.TensorShape((None,) + array_spec.shape[:-1]) if not CONFIG.use_rnn else
                       tf.TensorShape((None, None) + array_spec.shape[:-1])
                       for _, array_spec in self.policy_conf.action.action_space().items()])
        shapes.append(tf.TensorShape((None,)) if not CONFIG.use_rnn else
                      tf.TensorShape((None, None)))
        shapes.append(tf.TensorShape((None,)) if not CONFIG.use_rnn else
                      tf.TensorShape((None, None)))
        shapes.append(tf.TensorShape((None,)) if not CONFIG.use_rnn else
                      tf.TensorShape((None, None)))

        return shapes
