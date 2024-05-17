#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import datetime
from abc import ABCMeta
from framework.common.logging.kaiwu_logger import KaiwuLogger
from framework.common.config.config_control import CONFIG
from framework.common.config.app_conf import AppConf

class Trainer(metaclass=ABCMeta):
    def __init__(self, name) -> None:
        super().__init__()

        self.name = name

        self.policy_conf = AppConf[CONFIG.app].policies[CONFIG.policy_name]

        self.logger = KaiwuLogger()
        self.pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/learner_train_pid{self.pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
        self.logger.info('train process start at pid is {}', self.pid)

    def init(self):
        raise NotImplementedError

    @property
    def tensor_names(self):
        raise NotImplementedError

    @property
    def tensor_dtypes(self):
        raise NotImplementedError

    @property
    def tensor_shapes(self):
        raise NotImplementedError

    def input_ready(self):
        raise NotImplementedError

    def loop(self):
        raise NotImplementedError
    
    def train_hooks(self):
        raise NotImplementedError
    
    def chief_only_hooks(self):
        raise NotImplementedError
    
    def input_tensors(self):
        raise NotImplementedError
