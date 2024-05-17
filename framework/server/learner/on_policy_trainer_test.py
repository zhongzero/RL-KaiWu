#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.server.learner.on_policy_trainer import OnPolicyTrainer
from framework.common.config.config_control import CONFIG
from framework.common.config.app_conf import AppConf

class OnPolicyTrainerTest(unittest.TestCase):
    def test_run(self):
        CONFIG.app = 'gym'
        CONFIG.policy_name = 'train'
        
        AppConf._load_conf("""
                       {
                         "gym":{
                           "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                           "policies": {
                             "train": {
                               "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                               "state": "app.gym.gym_proto.GymState",
                               "action": "app.gym.gym_proto.GymAction",
                               "reward": "app.gym.gym_proto.GymReward",
                               "actor_network": "app.gym.gym_network.GymDeepNetwork",
                               "learner_network": "app.gym.gym_network.GymDeepNetwork",
                               "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                             }
                           }
                         }
                       }
                       """)



if __name__ == '__main__':
    unittest.main()