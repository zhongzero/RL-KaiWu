#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# rerf: https://github.com/horovod/horovod, need pip install horovod, gcc 7.3.1
import horovod.tensorflow as hvd

from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG
from framework.common.config.cluster_conf import ClusterConf
from framework.common.algorithms.model import ModeKeys
from framework.common.utils.common_func import get_local_rank, get_local_size
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
import numpy as np
import glob
import os

'''
ModelWrapperPytorch类, actor和learner都会使用, 主要用于预测, 训练等
'''


class ModelWrapperPytorch:

    def __init__(self, model, logger, server=None) -> None:

        self.model = model
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.predict_count = 0
        self.save_model_count = 0

        # 主learner
        self.is_chief = (CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER)

    def should_stop(self):
        return self.model.should_stop()

    def set_logger(self):
        self.model.set_logger(self.logger)

    def close(self):
        return self.model.stop()

    def before_train(self):
        pass

    def after_train(self):

        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False
        self.train_count += 1
        if self.train_count % CONFIG.dump_model_freq == 0:
            self.save_param()
            has_model_file_changed = True
        
        return has_model_file_changed, (self.save_model_count-1)*CONFIG.dump_model_freq

    def save_param(self):
        self.model.save_param(id=self.save_model_count*CONFIG.dump_model_freq)
        self.save_model_count += 1

    def before_predict(self, predict_data):
        return isinstance(predict_data, dict)

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    '''
    train 函数
    '''

    def train(self, extra_tensors=None):
        self.before_train()

        # 具体的训练流程
        data = self.sess.run(self.next_tensors)
        values = self.model.learn(data)

        # 返回是否更新了model文件, 更新的model文件的ID
        has_model_file_changed, model_file_id = self.after_train()

        return values, has_model_file_changed, model_file_id

    '''
    predict函数
    '''

    def predict(self, predict_data, batch_size):

        # 具体的预测流程
        values = None
        if self.before_predict(predict_data):
            # 部分场景需要更新predict_count
            if hasattr(self.model, 'update_predict_count'):
                self.model.update_predict_count(self.predict_count)

            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                values = self.model.predict(predict_data, types="prob")
            elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                types = "max" if np.random.rand(1) >= 0.1 else "prob" 
                values = self.model.predict(predict_data, types=types)   
            else:
                raise ValueError

            self.after_predict(batch_size)

        return values

    def get_global_step(self):
        return self.train_count

    @property
    def train_stat(self):
        return self.train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return 'ModelWrapperPytorch'

    @property
    def tf_sess(self):
        return self.sess

    '''
    直接调用业务类的load_last_new_model
    '''
    def load_last_new_model(self, models_path):
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            return self.model.load_last_new_model(models_path)
        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
            self.model.load_specific_model(CONFIG.eval_model_dir)
            self.logger.info(
                f"eval mode predict load_specific_model from {CONFIG.eval_model_dir} success")
        else:
            pass

    '''
    预加载模型文件, 直接调用业务类, 步骤如下:
    1. 删除引擎文件目录下的类似/data/ckpt/gorge_walk_v2_dqn/下的model.ckpt开头的文件
    2. 修改checkpoint文件内容类似/data/ckpt/gorge_walk_v2_dqn/checkpoint
    3. 保存最新的引擎文件
    4. 修改以后计数保存的变量值self.save_model_count
    '''
    def preload_model_file(self, preload_model_file, id):
        if not preload_model_file:
            return
        
        model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'
        file_pattern = "model.ckpt*"
        file_paths = glob.glob(os.path.join(model_path, file_pattern))
        for file_path in file_paths:
            os.remove(file_path)

        checkpoint_file = f'{model_path}/checkpoint'
        with open(checkpoint_file, "w") as f:
            # 将文件截断为0字节
            f.truncate(0)
            # 写入checkpoints list\n字符串
            f.writelines([
                    f"checkpoints list\n"
                ])

        self.model.load_specific_model(preload_model_file)

        self.model.save_param(id=id)

        # 按照整数来计数
        self.save_model_count = int(id / CONFIG.dump_model_freq) + 1
        
    def set_dataset(self, replay_buffer_wrapper):
        # self.model.set_dataset(dataset)

        self.sess = tf.Session()
        self.next_tensors = replay_buffer_wrapper.dataset_from_generator()
        self.sess.run(replay_buffer_wrapper.extra_initializer_ops())