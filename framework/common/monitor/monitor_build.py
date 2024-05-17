#!/usr/bin/env python3
# -*- coding:utf-8 -*-



from framework.common.config.config_control import CONFIG
from framework.common.monitor.monitor_proxy import MonitorProxy

'''
主要是考虑到有多个进程调用, 但是只是需要初始化monitor_proxy一次
'''

class MonitorBuilder:
    def __init__(self) -> None:
        self.monitor_proxy = None

    
    def build(self):
        pass