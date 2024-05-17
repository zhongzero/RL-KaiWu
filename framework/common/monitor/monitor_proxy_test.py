#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import json
from framework.common.config.config_control import CONFIG
from framework.common.monitor.monitor_proxy import MonitorProxy

class MonitorProxyTest(unittest.TestCase):
    def test_all(self):
        pass
    
    # 测试修改conf配置文件里的值
    def test_learner_actor_address(self):
        monitor_proxy = MonitorProxy(None)

if __name__ == '__main__':
    unittest.main()

