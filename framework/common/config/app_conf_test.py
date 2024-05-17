#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest

from framework.common.config.app_conf import AppConf

from framework.common.config.config_control import CONFIG
from framework.interface.action import Action
from framework.interface.network import Network
from framework.interface.reward import Reward
from framework.interface.reward_shaper import RewardShaper
from framework.interface.run_handler import RunHandler
from framework.interface.state import State

class MyState(State):
    def get_state(self):
        pass

    @staticmethod
    def state_space():
        pass


class MyAction(Action):
    def get_action(self):
        pass

    @staticmethod
    def action_space():
        pass


class MyReward(Reward):
    pass


class MyNetwork(Network):
    def build_network(self, input_tensors):
        pass

    def as_p(self):
        pass

    def as_v(self):
        pass

    def as_q(self):
        pass


class MyRewardShaper(RewardShaper):
    def should_train(self, exprs):
        pass

    def assign_rewards(self, exprs):
        pass


class MyRunHandler(RunHandler):
    def on_update_req(self, client_id, ep_id, req_data):
        pass

    def on_update_rsp(self, actions, extra_info=None):
        pass

    def policy_mapping_fn(self, agent_id):
        pass


class AppConfTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_all(self):
        AppConf._load_conf(
            '''
        {"hero": 
            {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "policies": {
                "train_one": {
                    "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                    "algo": "ppo", 
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
            '''
        )

        self.assertEqual(AppConf['hero'].run_handler.__name__, 'GymRunHandler')

        # 下面的配置需要和上面的load_conf里的配置一致
        policy = AppConf['hero'].policies['train_one']
        self.assertEqual(policy.policy_builder.__name__, 'AsyncBuilder')
        self.assertEqual(policy.state.__name__, 'GymState')
        self.assertEqual(policy.action.__name__, 'GymAction')
        self.assertEqual(policy.reward.__name__, 'GymReward')
        self.assertEqual(policy.network.__name__, 'GymDeepNetwork')
        self.assertEqual(policy.reward_shaper.__name__, 'GymRewardShaper')

    def test_AppConf(self):
        AppConf.load_conf("/data/projects/kaiwu-fwk/conf/app_conf.json")
        print("policy_conf1 " + str(AppConf[CONFIG.app]))
    
    def test_AppConf2(self):
        print("policy_conf2 " + str(AppConf[CONFIG.app]))
    
    def test_AppConf3(self):
        app_conf1 = AppConf
        
        app_conf2 = AppConf

        print("app_conf1 " + str(app_conf1) + " app_conf2 " + str(app_conf2))

if __name__ == '__main__':
    unittest.main()
