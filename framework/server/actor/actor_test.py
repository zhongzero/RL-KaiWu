#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest

try:
    import _pickle as pickle
except:
    import pickle

from framework.server.actor.actor_server_sync import ActorServerSync
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import TimeIt

class ActorServerTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/actor.toml")
        CONFIG.parse_actor_configure()

    def test_pickle_data(self):
        data = [0 for i in range(10000)]
        with TimeIt() as it:
            data = pickle.dumps(data)
        print(it.interval)

        with TimeIt() as it:
            data = pickle.loads(data, encoding='bytes')
        print(it.interval)



if __name__ == '__main__':
    unittest.main()