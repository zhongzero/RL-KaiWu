#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.common.config.config_control import CONFIG
from framework.server.aisrv.aisrv_arena_server import AiServer

class AiServerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_all(self):
        server = AiServer()
        server.run()

if __name__ == '__main__':
    unittest.main()