#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pydoc import locate
import rapidjson
from framework.common.utils.singleton import Singleton

''' 算法配置类, 实例如下:
{
  "ppo": {
    "actor_model": "framework.common.algorithms.ppo.PPO",
    "learner_model": "framework.common.algorithms.ppo.PPO",
    "trainer": "framework.server.learner.ppo_trainer.PPOTrainer",
    "predictor": "framework.server.actor.ppo_predictor.PPOPredictor",
    "expr_processor": "framework.common.algorithms.ppo_processor.PPOProcessor",
    "default_config": "framework.common.algorithms.ppo.PPODefaultConfig"
  }
}
'''

@Singleton
class _AlgoConf:
    class TupleType:
        def __init__(self, actor_model, learner_model, trainer, predictor, expr_processor, default_config):
            self._actor_model = actor_model
            self._learner_model = learner_model
            self._trainer = trainer
            self._predictor = predictor
            # experience processor
            self._expr_processor = expr_processor
            self._default_config = default_config

        def __getattr__(self, key):
            value = getattr(self, '_' + key)
            value = locate(value)
            return value

    _instance = None

    def __init__(self):
        self.config_map = {
            # 主网络算法，ppo的相关配置
            "ppo": self.TupleType(
                "framework.common.algorithms.ppo.PPO",
                "framework.common.algorithms.ppo.PPO",
                "framework.server.learner.ppo_trainer.PPOTrainer",
                "framework.server.actor.ppo_predictor.PPOPredictor",
                "framework.common.algorithms.ppo_processor.PPOProcessor",
                "framework.common.algorithms.ppo.PPODefaultConfig"
            ),
        }

    def __getitem__(self, key):
        return self.config_map[key]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def load_conf(self, file_name):
        with open(file_name, "r") as file_obj:
            self._load_conf(file_obj.read())

    def _load_conf(self, json_str):
        json_obj = rapidjson.loads(json_str)
        for algo in json_obj:
            self.config_map[algo] = self.TupleType(**json_obj[algo])


AlgoConf = _AlgoConf()