#!/usr/bin/env python
# -*- coding: utf-8 -*-


import multiprocessing
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from framework.common.config.app_conf import AppConf
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import Context

# 单侧时加上的, 因为在调用ActorProxy需要对接口里的函数进行处理, 正式上线的不需要
CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
CONFIG.parse_aisrv_configure()

from framework.server.aisrv.actor_proxy import ActorProxy

class ActorProxyTest(unittest.TestCase):
    def test_put_and_get(self):

        CONFIG.app = "gym"
        AppConf._load_conf(
            """
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
            """
        )

        context = Context()
        context.slots = MagicMock()

        actor_proxy = ActorProxy('train', 0, '127.0.0.1:8080', context)
        actor_proxy.before_run()

        actor_proxy.put_data(1, 2, {'x': np.ones((4,))})
        actor_proxy.proc_one_msg()

        self.assertEqual(actor_proxy.compose_id_buf[0].tolist(), [1, 2])

        actor_proxy.actor_proxy.configure_mock(**{
            'recv.return_value': [((1, 2), {'x': np.ones((4,))})]})

        pipe = multiprocessing.Pipe(duplex=False)
        actor_proxy.slots.configure_mock(**{
            'get_input_pipe.return_value': pipe[0],
            'get_output_pipe.return_value': pipe[1]
        })

        actor_proxy.flush_buffer_data()
        actor_proxy.actor_proxy.send.assert_called_once()
        self.assertEqual(actor_proxy.cur_buf_size, 0)

        data = actor_proxy.get_data(1)
        self.assertTrue(2 in data)


if __name__ == '__main__':
    unittest.main()
