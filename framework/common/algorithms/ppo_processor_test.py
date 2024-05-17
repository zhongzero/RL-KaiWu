#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import numpy as np
import unittest
from unittest import mock

from framework.common.algorithms.distribution import GaussianDist
from framework.common.algorithms.ppo_processor import PPOProcessor
from framework.common.utils.common_func import Context
from framework.interface.agent_context import AgentContext
from framework.interface.action import ActionSpec
from framework.interface.array_spec import ArraySpec
from framework.common.config.config_control import CONFIG


class MockFlags:
    def __init__(self, args=None):
        self._dict = args or dict()

    def __getattr__(self, item):
        if item in self._dict:
            return self._dict.get(item)

        return mock.Mock()

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]

        return mock.Mock()

    def __setitem__(self, key, value):
        self._dict[key] = value


class MockLogger:
    @staticmethod
    def get_logger():
        logger = logging.getLogger("ut")
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter(fmt="%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] - %(message)s")
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)

            console.setFormatter(formatter)
            logger.addHandler(console)

        return logger


class MockReward:
    def __init__(self):
        self._ex_rewards = []  # 外部环境赋予的reward
        self._in_rewards = []  # 内部模型计算获得reward

    def get_reward(self):
        """
        :return: float类型, 按照一定的规则形成最终的reward值
        """
        return self.get_in_reward() + self.get_ex_reward()

    def get_in_reward(self):
        return sum(self._in_rewards)

    def get_ex_reward(self):
        return sum(self._ex_rewards)

    def extend_ex_reward(self, ex_rewards):
        self._ex_rewards.extend(ex_rewards)

    def add_in_reward(self, in_reward):
        self._in_rewards.append(in_reward)


class MockState:
    def __init__(self, value):
        self.value = value  # [1, 2]

    def get_state(self):
        array = np.array(self.value, dtype=np.float32)
        return {'x': array}

    @staticmethod
    def state_space():
        return {'x': ArraySpec((2, 3, 4, ), np.float32)}

    def __str__(self):
        return str(self.value)


class MockAction:
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {'a': self.a}

    @staticmethod
    def action_space():
        return {'a': ActionSpec(ArraySpec((1,), np.float32), GaussianDist)}

    def __str__(self):
        return str(self.a)


class MockRewardShaper:
    def __init__(self, simu_ctx, agent_id):
        pass

    def should_train(self, exprs):
        if len(exprs) > 0:
            done = exprs[-1].done
        else:
            done = False

        return done

    def assign_rewards(self, exprs):
        for expr in exprs:
            expr.reward.add_in_reward(0.0)

    def initialize(self):
        pass

    def finalize(self):
        pass


class PPOProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        class MockPolicyConf:
            def __init__(self):
                self.state = MockState
                self.action = MockAction
                self.reward_shaper = MockRewardShaper
        
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/learner.toml")
        CONFIG.parse_learner_configure()

        self._simu_ctx = Context()

        self._agent_ctx = AgentContext()
        self._agent_ctx.state = {
            'train': MockState(value=[1.0, 2.0])
        }
        self._agent_ctx.action = MockAction(a=[1.0])
        self._agent_ctx.reward = MockReward()
        self._agent_ctx.done = True

        self._agent_ctx.pred_output = {
            'train': {
                "neg_logprob_a": 0.999,  # a对应的neg_logp值
                "v": 0.1,  # float, vpred
                "s": 1,  # int, step
            }
        }
        self._agent_ctx.policy_conf = {
            'train': MockPolicyConf()
        }
        self._agent_ctx.main_id = 'train'

        args = {
            'expr_skip_rate': 0.0,
            'ppo_gamma': 0.99,
            'ppo_lam': 0.95,
            'ppo_ent_coef': 0.01,
            'ppo_vf_coef': 0.5,
            'ppo_pg_coef': 1,
            'ppo_mini_batch_count': 96,
            'ppo_clip_range': 0.2,
            'ppo_end_clip_range': 0.05,

            'use_rnn': False,
            'rnn_states': ['lstm_cell', 'lstm_hidden'],

            'expr_overlap': False
        }

        self._ppo_processor = PPOProcessor(simu_ctx=self._simu_ctx,
                                           agent_ctx=self._agent_ctx,
                                           policy_id='train')

    def test_exprs_procedure(self):
        self._ppo_processor.initialize()

        
        self._ppo_processor.gen_expr()


 
        #self.assertTrue(self._ppo_processor.should_train())

        train_data, valid_frame_cnt, skip_frame_cnt = self._ppo_processor.proc_exprs()
        print(train_data, valid_frame_cnt, skip_frame_cnt)

        #self.assertEqual(self._ppo_processor.reward_sum, 0)
        #self.assertEqual(self._ppo_processor.in_reward_sum, 0)
        #self.assertEqual(self._ppo_processor.reward_sum, 0)

        self._ppo_processor.finalize()

if __name__ == '__main__':
    unittest.main()
