#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.server.learner.ppo_trainer import PPOTrainer
from framework.common.config.config_control import CONFIG

class PPOTrainerTest(unittest.TestCase):
    def setUp(self):
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/learner.toml")
        CONFIG.parse_learner_configure()

if __name__ == '__main__':
    unittest.main()