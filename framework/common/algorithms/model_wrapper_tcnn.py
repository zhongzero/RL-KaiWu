#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# rerf: https://github.com/horovod/horovod, need pip install horovod, gcc 7.3.1
import horovod.tensorflow as hvd

from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG
from framework.common.config.cluster_conf import ClusterConf
from framework.common.algorithms.model import ModeKeys
from framework.common.utils.common_func import get_local_rank, get_local_size

'''
ModelWrapperTcnn类, actor和learner都会使用, 主要用于预测, 训练等
'''
class ModelWrapperTcnn:

    def __init__(self, model, logger, server = None) -> None:
        
        self.model = model
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.predict_count = 0

    def should_stop(self):
        return self.sess.should_stop()

    def close(self):
        return self.sess.close()
    
    def before_train(self):
        pass

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False

        self.train_count += 1

        return has_model_file_changed, -1

    def before_predict(self):
        pass

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    '''
    train 函数
    '''
    def train(self, extra_tensors=None):
        self.before_train()

        # 具体的训练流程
        values = None

        has_model_file_changed, model_file_id = self.after_train()

        return values, has_model_file_changed, model_file_id

    '''
    predict函数
    '''
    def predict(self, extra_tensors, batch_size):
        self.before_predict()
        
        # 部分场景需要更新predict_count
        if hasattr(self.model, 'update_predict_count'):
            self.model.update_predict_count(self.predict_count)

        # 具体的预测流程
        values = None

        self.after_predict(batch_size)

        return values
        
    @property
    def train_stat(self):
        return self.train_count
    
    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return 'ModelWrapperTcnn'