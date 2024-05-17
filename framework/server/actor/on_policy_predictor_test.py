#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import lz4.block
import unittest

from framework.common.config.app_conf import AppConf
from framework.common.config.config_control import CONFIG


class TestOnPolicyPredictor(unittest.TestCase):
    def setUp(self):
        # pylint: disable=protected-access
        AppConf._load_conf("""
                {
                  "hero":{
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

        CONFIG.app = 'gym'
        CONFIG.policy_name = 'train'

    def test_lz4(self):
      data = [1, 2, 3, 4, 5]
      compress_train_data = lz4.block.compress(bytes(data), store_size=False)
      print(data, compress_train_data)

if __name__ == '__main__':
    unittest.main()