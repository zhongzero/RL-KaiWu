#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
import unittest
import os
import yaml
from framework.common.config.config_control import CONFIG
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.rainbow_utils import RainbowUtils

class RainbowUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()

        self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, CONFIG.rainbow_secret_key, CONFIG.rainbow_group, CONFIG.rainbow_env_name, None)
    
    def test_read_from_rainbow(self):
        print('rainbow_utils.read_from_rainbow')
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow()
        if result_code:
            print(f'read_from_rainbow failed, err msg is {result_msg}')
            return
        
        # to_change_key_values = {}
        # for key,value in data.items():
        #     print(f'key {key}, value {value}')

        #     # 更新CONFIG的值
        #     CONFIG.__dict__[key] = value
        #     to_change_key_values[key] = value
        #     print(f'key {key} is {CONFIG.__dict__[key]}')
        to_change_key_values = yaml.load(data[CONFIG.svr_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(CONFIG.svr_name, to_change_key_values)

    def test_write_to_rainbow(self):
        print('rainbow_utils.write_to_rainbow')
        configure_data = {}
        self.rainbow_utils.write_to_rainbow(configure_data)


if __name__ == '__main__':
    unittest.main()
