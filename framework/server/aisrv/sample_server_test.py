#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from framework.common.config.config_control import CONFIG

class MsgEngineTest(unittest.TestCase):
    def test_all(self):

        # 解析配置
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()


if __name__ == '__main__':
    unittest.main()
