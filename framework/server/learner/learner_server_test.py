#!/usr/bin/env python3
# -*- coding:utf-8 -*-



import unittest
from framework.common.config.config_control import CONFIG
from framework.common.config.algo_conf import AlgoConf
from framework.common.config.app_conf import AppConf
from framework.common.utils.cmd_argparser import cmd_args_parse
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.server.learner.learner_server import LearnerServer

class LearnerServerTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def proc_flags(configure_file):
        CONFIG.set_configure_file(configure_file)
        CONFIG.parse_learner_configure()

        # 加载配置文件conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)

    def test_all(self):
        # 步骤1, 按照命令行来解析参数
        args = cmd_args_parse(KaiwuDRLDefine.SERVER_LEARNER)
        # 步骤2, 解析参数, 包括业务级别和算法级别
        self.proc_flags(args.conf)
        a=LearnerServer()
        a.start()
        while True:pass

if __name__ == '__main__':
    unittest.main()