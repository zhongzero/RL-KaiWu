#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
import unittest
import os
from framework.common.config.config_control import CONFIG
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.redis_utils import redis_client

class RainbowUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()

        self.redis_client = redis_client

    def test_read_from_redis(self):
        print('redis_client.test_read_from_redis')
        self.redis_client.set("key", "value")


if __name__ == '__main__':
    unittest.main()
