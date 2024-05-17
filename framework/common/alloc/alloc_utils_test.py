#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import os
from framework.common.config.config_control import CONFIG
from framework.common.alloc.alloc_utils import SERVER_ROLE_CONFIGURE, AllocUtils

class AllocUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()

    def test_registry(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.registry())

    def test_get(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.get(SERVER_ROLE_CONFIGURE['actor']))
        print(allocUtils.get(SERVER_ROLE_CONFIGURE['learner']))
    
    def test_get_self_play(self):
        allocUtils = AllocUtils(None)
        print(allocUtils.get(SERVER_ROLE_CONFIGURE['actor']), 'set100')
        print(allocUtils.get(SERVER_ROLE_CONFIGURE['learner']), 'set100')

if __name__ == '__main__':
    unittest.main()
