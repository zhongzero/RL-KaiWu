#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import re
from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG

class CheckpointSaverHook(tf.compat.v1.train.CheckpointSaverHook):
    def __init__(self, 
                checkpoint_dir, 
                summary_dir, 
                save_secs=None, 
                save_steps=None, 
                saver=None, 
                checkpoint_basename="model.ckpt", 
                scaffold=None, 
                listeners=None) -> None:

        super(CheckpointSaverHook, self).__init__(
            checkpoint_dir, 
            save_secs, 
            save_steps, 
            saver, 
            checkpoint_basename, 
            scaffold, 
            listeners)
    
        self._summary_dir = summary_dir

    def begin(self):
        old_dir = self._checkpoint_dir
        self._checkpoint_dir = self._summary_dir
        super(CheckpointSaverHook, self).begin()
        self._checkpoint_dir = old_dir

class CkptSaverListener(tf.compat.v1.train.CheckpointSaverListener):
    def __init__(self, inputs, outputs, extra_info=None):
        super(CkptSaverListener, self).__init__()
        self._inputs = inputs
        self._outputs = outputs
        self._dir_name = None
        self._config = tf.compat.v1.ConfigProto()
        self._config.gpu_options.visible_device_list = '0'
        self._extra_info = extra_info
    
    def after_save(self, session, global_step_value):
        self.save_model(session, global_step_value)
    
    def save_model(self, session, step):
        self.setup_output_dir(step)
        input_signature = {k: tensor.name for k, tensor in self._inputs.items()}
        output_signature = {k: tensor.name for k, tensor in self._outputs.items()}
        save_frozen_model(session, self._dir_name, input_signature, output_signature, config=self._config)
        self.remove_expired_models()

    '''
    保存格式: 形如/data/pb_model/00001168/
    '''
    def setup_output_dir(self, step):
        self._dir_name = '%s/%08d' % (CONFIG.pb_model_dir, step)
        if tf.io.gfile.exists(self._dir_name):
            tf.io.gfile.rmtree(self._dir_name)
        
    '''
    删除本地保存的过期的model文件
    '''
    def remove_expired_models(self):
        saved_models = []
        for home, dirs, files in tf.io.gfile.walk(CONFIG.pb_model_dir):
            if home != CONFIG.pb_model_dir:
                continue
            dirs.sort()

            for filename in dirs:
                if re.match("[0-9]{8}", filename):
                    saved_models.append(os.path.join(home, filename))
        while len(saved_models) > CONFIG.save_pb_num:
            del_pb_dir = saved_models.pop(0)
            if tf.io.gfile.exists(del_pb_dir):
                tf.io.gfile.rmtree(del_pb_dir)