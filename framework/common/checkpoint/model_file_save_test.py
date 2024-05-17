#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import tempfile

from framework.common.config.config_control import CONFIG
from framework.common.checkpoint.model_file_save import ModelFileSave
from distutils.dir_util import copy_tree, remove_tree
from framework.common.config.algo_conf import AlgoConf
from framework.common.config.app_conf import AppConf

class ModelFileSaveTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()

        # 加载配置文件conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)
    
    def test_model_file_save(self):
        pass

        # model_file_save = ModelFileSave()
        # model_file_save.start()

    def test_file(self):
        local_and_remote_dirs = {
            'ckpt_dir' : '/data/ckpt/',
            'summary_dir' : '/data/summary/',
            'pb_model' : '/data/pb_model/',
        }

        temp_remote_dirs = {}
        for _, local_and_remote_dir in local_and_remote_dirs.items():
            print(local_and_remote_dir)
            target_dir = tempfile.mkdtemp()
            print(target_dir)
            copy_tree(local_and_remote_dir, target_dir)
            temp_remote_dirs[target_dir] =  target_dir
        
        print(temp_remote_dirs)
    
    def test_save_model_file_to_cos(self):
        model_file_save = ModelFileSave()
        model_file_save.save_model_file_to_cos()
    
    def test_clear_dir(self):
        model_file_save = ModelFileSave()
        model_file_save.clearn_dir()

if __name__ == '__main__':
    unittest.main()