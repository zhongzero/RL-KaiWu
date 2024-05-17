#!/usr/bin/env python
# -*- coding: utf-8 -*-


import threading
import unittest

from framework.common.utils.common_func import Context
from framework.server.aisrv.msg_buff import MsgBuff
from framework.common.config.config_control import CONFIG

def consumer(msg_buff):
    msg = msg_buff.recv_msg()
    msg_buff.send_msg(msg)


class MsgEngineTest(unittest.TestCase):
    def test_all(self):

        # 解析配置
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        context = Context()
        context.slot_id = 1
        context.client_address = "127.0.0.1:8080"

        msg_engine = MsgBuff(context)
        t = threading.Thread(target=consumer, args=(msg_engine,))
        t.start()

        msg = msg_engine.update("Hello Kaiwu!")
        self.assertEqual(msg, "Hello Kaiwu!")

        t.join()

if __name__ == '__main__':
    unittest.main()
