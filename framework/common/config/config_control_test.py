#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.common.config.config_control import CONFIG
import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

class ConfigControlTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_load_configure(self):
        print("run_mode" + str(CONFIG.run_mode))
    
    def test_aisrv_configure(self):
        CONFIG.set_configure_file("../../../conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        print(CONFIG.aisrv_ip_address)

    def test_learner_configure(self):
        CONFIG.set_configure_file("../../../conf/framework/learner.toml")
        CONFIG.parse_learner_configure()

        print(CONFIG.train_batch_size)

    def test_client_configure(self):
        CONFIG.set_configure_file("../../../conf/framework/client.toml")
        CONFIG.parse_client_configure()

    def test_actor_configure(self):
        CONFIG.set_configure_file("../../../conf/framework/actor.toml")
        CONFIG.parse_actor_configure()

        print(CONFIG.actor_server_port)
    
    def test_actor_ip_addrs(self):
        CONFIG.set_configure_file("../../../conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        print(CONFIG.actor_addrs)
        print(CONFIG.learner_addrs)

        address = json.loads(CONFIG.actor_addrs, strict=False)
        print(f'address is {address}, type is {type(address)}')

        actor_address = address.get('train_one')
        print(f'actor address is {actor_address}, type is {type(actor_address)}')

    
    # 测试配置文件回写功能
    def test_save_to_file(self):
        print('test_save_to_file')

        CONFIG.save_to_file('aisrv')

if __name__ == '__main__':
    unittest.main()
