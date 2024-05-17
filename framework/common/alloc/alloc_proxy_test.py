#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import json

from framework.common.alloc.alloc_proxy import AllocProxy
from framework.common.config.config_control import CONFIG

class AllocProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()
    
    def test_all(self):
        pass
    
    # 测试修改conf配置文件里的值
    def test_learner_actor_address(self):

        learner_addrs = '127.0.0.1:8000'

        if learner_addrs:
            learner_addrs = learner_addrs.split(',')
            CONFIG.learner_proxy_num = len(learner_addrs)

            # 实例: learner_addrs = {"train_one": ["127.0.0.1:9000"], "train_two": ["127.0.0.1:9001"]}
            learner_address_str = None
            for idx in range(len(learner_addrs)):
                if 0 == idx:
                    learner_address_str = learner_addrs[idx]
                else:
                    learner_address_str = ',' + learner_addrs[idx]
            
            print(learner_address_str)
            
            CONFIG.learner_addrs = '{' + f'\"train\": ["{learner_address_str}"]' + '}'
            print(f'AiSrv from alloc learner IP is {learner_address_str}, CONFIG.learner_addrs is {CONFIG.learner_addrs}')

            learner_addrs = json.loads(CONFIG.learner_addrs)['train']
            print(f'learner_addrs json load is {learner_addrs}')
    
    # 测试AllocProxy的逻辑
    def test_alloc(self):
        alloc_proxy = AllocProxy()
        alloc_proxy.start()

if __name__ == '__main__':
    unittest.main()

