#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest

from framework.common.config.algo_conf import AlgoConf
from framework.server.actor.predictor import Predictor
from framework.server.learner.trainer import Trainer
from framework.common.algorithms.model import Model
from framework.common.algorithms.expr_processor import ExprProcessor
from framework.common.config.config_control import CONFIG

class AlgoConfTest(unittest.TestCase):
    def test_get_conf(self):

        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        AlgoConf.load_conf("/data/projects/kaiwu-fwk/conf/algo_conf.json")

        print(AlgoConf['ppo'].expr_processor)

        conf = AlgoConf['ppo']
        print(conf.model, conf.predictor, conf.trainer, conf.expr_processor, conf.default_config)

        self.assertTrue(issubclass(conf.model, Model))
        self.assertTrue(issubclass(conf.expr_processor, ExprProcessor))
        self.assertTrue(issubclass(conf.predictor, Predictor))
        self.assertTrue(issubclass(conf.trainer, Trainer))
        self.assertTrue(isinstance(conf.default_config, dict))


if __name__ == '__main__':
    unittest.main()
