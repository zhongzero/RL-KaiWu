#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import flatbuffers
import unittest
from unittest import mock
import json

from framework.common.utils.common_func import Context

# 配置文件提前加载
from framework.common.config.config_control import CONFIG

CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
CONFIG.parse_aisrv_configure()

from framework.server.aisrv.async_policy import AsyncBuilder
from framework.common.config.app_conf import AppConf
from framework.common.utils.slots import Slots


class AsyncBuilderTest(unittest.TestCase):
    def setUp(self) -> None:

        # 注意和配置文件里配置的项目名字一致
        print(json.loads(CONFIG.actor_addrs)["train_one"])

        AppConf.load_conf("/data/projects/kaiwu-fwk/conf/sgame_app.json")

        fake_simu_ctx = Context()
        policy_name = "train_one"
        fake_simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))

        self.async_builder = AsyncBuilder(policy_name=policy_name, simu_ctx=fake_simu_ctx)

        self.simu_ctx = Context()
        # 从配置文件加载
        policies_builder = {}
        policies_conf = AppConf[CONFIG.app].policies
        for policy_name, policy_conf in policies_conf.items():
            print("policy_name " + str(policy_name))
            policies_builder[policy_name] = policy_conf.policy_builder(policy_name, self.simu_ctx)
        self.simu_ctx.policies_builder = policies_builder

    def test_build(self):
        self.async_builder.build()
    
    '''
    下面测试actor/learner扩缩容实例
    '''
    def test_add_actor(self):
        actor_ip = '127.0.0.1'
        self.async_builder.add_actor_proxy_list(actor_ip)
    
    def test_reduce_actor(self):
        actor_ip = '127.0.0.1'
        self.async_builder.reduce_actor_proxy_list(actor_ip)
    
    def test_add_learner(self):
        learner_ip = '127.0.0.1'
        self.async_builder.add_learner_proxy_list(learner_ip)
    
    def test_reduce_learner(self):
        learner_ip = '127.0.0.1'
        self.async_builder.reduce_learner_proxy_list(learner_ip)

if __name__ == '__main__':
    unittest.main()
