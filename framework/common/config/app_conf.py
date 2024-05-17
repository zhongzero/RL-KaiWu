#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pydoc import locate

import rapidjson

from framework.common.config.config_control import CONFIG
from framework.common.utils.singleton import Singleton
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine


'''业务配置类, 实例如下
{
  "gym": 业务名
  {
    "run_handler": "app.gym.gym_run_handler.GymRunHandler",
    "rl_helper": "environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
    "policies": {
      "train": {
        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
        "algo": "ppo", 算法
        "state": "app.gym.gym_proto.GymState", State
        "action": "app.gym.gym_proto.GymAction", Action
        "reward": "app.gym.gym_proto.GymReward", Reward
        "actor_network": "app.gym.gym_network.GymDeepNetwork", NetWork
        "learner_network": "app.gym.gym_network.GymDeepNetwork", NetWork
        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper" Reward Shaper,
        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
      }
    }
  }
}
'''

class _PolicyConf:
    def __init__(self, policy_builder, **kwargs):
        """
        一般需要定义state、action、reward以及reward_shaper等
        """
        self.policy_builder = locate(policy_builder)
        algo = kwargs.pop('algo', None)
        if algo is not None:
            self.algo = algo

        # 如果是aisrv, 不需要加载actor, learner相关的
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            if 'learner_network' in kwargs:
                del kwargs['learner_network']
            if 'actor_network' in kwargs:
                del kwargs['actor_network']
        else:
            # 如果是learner, 不需要加载actor相关的
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
                if 'learner_network' in kwargs:
                    del kwargs['learner_network']
            
            # 如果是actor, 不需要加载learner相关的
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                if 'actor_network' in kwargs:
                    del kwargs['actor_network']
            else:
                pass

        attrs = {k: locate(clazz) for k, clazz in kwargs.items()}
        self.__dict__.update(**attrs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


@Singleton
class _AppConf:
    class TupleType:
        def __init__(self,
                     run_handler,
                     rl_helper,
                     policies,
                     builder=None, **kwargs
                     ):
            if 'server_name' in kwargs and kwargs['server_name'] == 'aisrv':
                self.run_handler = locate(run_handler)
                self.rl_helper = locate(rl_helper)

            if "environment" in kwargs:
                self.environment = locate(kwargs["environment"])
            else:
                self.environment = None

            self.policies = {name: _PolicyConf(**policy) for name, policy in policies.items()}
            if "actor" in kwargs:
                self.actor = locate(kwargs["actor"])
            else:
                self.actor = None
            
            if builder is None:
                builder = "framework.interface.builder.Builder"
            self.builder = locate(builder)

    _instance = None

    def __init__(self):
        self.config_map = {}

    def __getitem__(self, key):
        return self.config_map[key]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def load_conf(self, file_name, server_name=None):

        with open(file_name, "r") as file_obj:
            self._load_conf(file_obj.read(), server_name)

    def _load_conf(self, json_str, server_name=None):
        json_obj = rapidjson.loads(json_str)

        if server_name:
            json_obj[CONFIG.app]['server_name'] = server_name

        assert CONFIG.app in json_obj, f"failed to find {CONFIG.app} app conf"
        self.config_map[CONFIG.app] = self.TupleType(**json_obj[CONFIG.app])


AppConf = _AppConf()
