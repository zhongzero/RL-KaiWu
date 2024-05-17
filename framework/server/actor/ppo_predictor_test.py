#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.server.actor.ppo_predictor import PPOPredictor
from framework.common.config.config_control import CONFIG

class PPOPredictorTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/actor.toml")
        CONFIG.parse_actor_configure()

if __name__ == '__main__':
    unittest.main()