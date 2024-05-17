#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
from framework.common.utils.tf_utils import *

'''
目前只是train和predict
'''
class ModeKeys:
    TRAIN = 'train'
    PREDICT = 'predict'

class Model(object):
    '''
    所有算法模型的基类, 通过调用build_model构建模型, 并返回ModelSpec
    '''
    def __init__(self, network, name, *args):
        self.network = network
        self.name = name

    def build_model(self, mode, input_tensors):
        """
        构建模型结构
        :param mode: ModeKeys类型, 表示是构建训练模型还是预测模型
        :param input_tensors: Tensor数组, 表示模型的输入
        :return: 返回ModelSpec
        """
        raise NotImplementedError

    def ckpt_saver_hook(self, var_list=None):
        """
        返回保存模型的session_run_hook, 采用的是hook方式
        :return: tf.estimator.CheckpointSaverHook的一个子类
        """
        raise NotImplementedError

    def inference(self, feature):
        raise NotImplementedError("build model: not implemented!")
    
    @staticmethod
    def current_step(checkpoint_dir, relative_step=False):
        """
        在很多算法中learning rate或者其他一些需要按照global step做衰减的计算,
        :param checkpoint_dir: checkpoint所在目录
        :param relative_step: False表示使用绝对偏移计算decay, True表示使用相对偏移计算decay
        """
        global_step = tf.compat.v1.train.get_or_create_global_step()
        if not relative_step:
            return global_step

        ckpt_step = 0
        lastest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if lastest_ckpt:
            ckpt_step = int(os.path.basename(lastest_ckpt).split('-')[1])
        return global_step - ckpt_step

class ModelSpec:
    def __init__(self,
                 predict_input=None,
                 predict_output=None,
                 train_input=None,
                 loss=None,
                 optimizer=None):
        """
        仿照EstimatorSpec的定义, ModelSpec是Model的build_model函数的返回值
        :param predict_input: dict of Tensor.
        :param predict_output: dict of Tensor.
        :param train_input: dict of Tensor.
        :param loss: Training loss Tensor. Must be either scalar, or with shape [1]
        :param optimizer: Optimizer used to optimize loss function
        """
        if predict_input is not None:
            self._check_is_tensor_dict(predict_input, "predict_input")
        self.predict_input = predict_input

        if predict_output is not None:
            self._check_is_tensor_dict(predict_output, "predict_output")
        self.predict_output = predict_output

        if train_input is not None:
            self._check_is_tensor_dict(train_input, "train_input")
        self.train_input = train_input
        if loss is not None:
            self._check_is_tensor(loss, 'loss')
        self.loss = loss

        if optimizer is not None:
            self._check_is_optimizer(optimizer, 'optimizer')
        self.optimizer = optimizer

    def _check_is_tensor(self, x, tensor_name):
        if not isinstance(x, tf.Tensor):
            raise TypeError('{} must be Tensor, given: {}'.format(tensor_name, x))

    def _check_is_optimizer(self, x, name):
        if not isinstance(x, tf.compat.v1.train.Optimizer):
            raise TypeError('{} must be Optimizer, given: {}'.format(name, x))

    def _check_is_tensor_dict(self, d, name):
        if not isinstance(d, dict):
            raise TypeError('{} must be dict, given: {}'.format(name, d))

        for k, v in d.items():
            self._check_is_tensor(v, "%s[%s]" % (name, k))