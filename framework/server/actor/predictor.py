#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import datetime
from abc import ABCMeta
import psutil
from framework.common.config.app_conf import AppConf
from framework.common.config.config_control import CONFIG

class Predictor(metaclass=ABCMeta):
    def __init__(self, send_server, recv_server, name):
        super().__init__()

        '''
        receive_server, actor上负责从aisrv获取预测请求的类
        send_server, actor上负责朝aisrv发送预测响应的类
        使用场景:
        1. receive_server和send_server是同一个, 比如actor采用同步批处理方式
        2. receive_server和send_server是不同类, 比如actor采用异步批处理方式

        为了兼顾旧的代码, self.server = receive_server, 即server默认指receive_server
        '''
        self.send_server = send_server
        self.recv_server = recv_server

        self.name = name

        # policy_name 主要是和conf/app_conf.json设置一致
        self.policy_conf = AppConf[CONFIG.app].policies[CONFIG.policy_name]

    def input_tensors(self):
        raise NotImplementedError
    
    def predict_hooks(self):
        raise NotImplementedError
    
    def loop(self):
        raise NotImplementedError
