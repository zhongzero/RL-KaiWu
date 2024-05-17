#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import hashlib
import os
import re
import shutil
import tempfile
import unittest
import random
from distutils.dir_util import copy_tree, remove_tree
from framework.common.checkpoint.model_file_sync import ModelFileSync
from framework.common.utils.common_func import get_first_line_and_last_line_from_file, make_tar_file, tar_flie_extract
from framework.common.checkpoint.model_pool_apis import ModelPoolAPIs
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.config.config_control import CONFIG

class ModelFileSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()
    
    def test_model_file_wrapper(self):
        model_file_wrapper = ModelFileSync()
        model_file_wrapper.start()
    
    def test_model_pool_ip(self):

        model_pool_addrs = "127.0.0.1:10013:10014"
        model_pool_addrs =  model_pool_addrs.split(',')
        print(model_pool_addrs)
        rand_int = random.randint(0, len(model_pool_addrs))
        print(rand_int)
    
    '''
    流程如下:
    1. 根据形如/data/ckpt/hero_ppo/checkpoint找到最新的step生成的checkpoint文件, 形如:
    all_model_checkpoint_paths: "/data/ckpt//hero_ppo/model.ckpt-3247"
    2. 在/data/ckpt/hero_ppo/ | grep 3247找出满足需求的model-3247.data-00000-of-00001, model-3247.index, model-3247.meta, checkpoint
    3. 对2制作成tar文件, 生成tar文件路径
    '''
    def test_push_to_modelpool(self):
        model_path = f'/data/ckpt/sgame_ppo/'
        checkpoint_file = f'{model_path}/checkpoint'
        _, last_line = get_first_line_and_last_line_from_file(checkpoint_file)

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841"
        checkpoint_id = re.findall(r'\d+\.?\d*', last_line)[0]
        checkpoint_id = last_line.split(f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1]

        target_dir = tempfile.mkdtemp()
   
        # 寻找包含checkpoint_id的meta, data, index
        for root, dirs, file_list in os.walk(model_path):
            for file_name in file_list:
                if f'model.ckpt-{checkpoint_id}' in file_name:
                    shutil.copy(os.path.join(root, file_name), target_dir)
        
        shutil.copy(checkpoint_file, target_dir)
                    
        # 放在/tmp目录下生成tar文件
        output_file_name = f'{model_path}/kaiwu_checkpoint_sgame_ppo_{checkpoint_id}.tar.gz'
        make_tar_file(output_file_name, target_dir)

        print(target_dir)

        # 删除/tmp的临时文件
        #remove_tree(target_dir)

        model_pool_apis = ModelPoolAPIs('127.0.0.1:10014')
        model_pool_apis.check_server_set_up()

        with open(output_file_name, "rb") as fin:
            model = fin.read()
            local_md5 = hashlib.md5(model).hexdigest()

            print(f'local_md5 is {local_md5}')

            print(f'output_file_name is {output_file_name}')
            save_file_name = output_file_name.split('/')[-1]
            print(f'save_file_name is {save_file_name}')

            model_pool_apis.push_model(model=model, hyperparam=None, key="model.ckpt_sgame_ppo",\
                        md5sum=local_md5, save_file_name=save_file_name)
    
    def test_pull_from_modelpool(self):
        model_pool_apis = ModelPoolAPIs('127.0.0.1:10014')
        model_pool_apis.check_server_set_up()
        model_name_list = model_pool_apis.pull_keys()
        for model_name in model_name_list:

            # 获取model文件名字
            model_info = model_pool_apis.pull_model_info(model_name)
            if not model_info:
                continue
            model_file_name = model_info._file_name

            # 获取model文件内容
            model = model_pool_apis.pull_model(model_name)
            if not model:
                continue

            with open(f'/data/ckpt/sgame_ppo/plugins/{model_file_name}', 'wb+') as file:
                file.write(model)
            
            print(f'success to /data/ckpt/sgame_ppo/plugins/{model_file_name}')

            tar_flie_extract(f'/data/ckpt/sgame_ppo/plugins/{model_file_name}', '/data/ckpt/sgame_ppo/plugins/')

            # 遍历文件夹拷贝文件到models下去
            for root, dirs, file_list in os.walk('/data/ckpt/sgame_ppo/plugins/'):
                for file_name in file_list:
                    if f'model.ckpt' in file_name or 'checkpoint' == file_name:
                        shutil.copy(os.path.join(root, file_name), '/data/ckpt/sgame_ppo/models/')

            # 删除plugins下的文件, 采用删除原文件夹, 新增文件夹方式
            shutil.rmtree('/data/ckpt/sgame_ppo/plugins/')
            os.mkdir('/data/ckpt/sgame_ppo/plugins/')

        
    def test_push_pull_model_file(self):
        model_file_wrapper = ModelFileSync()

        # push model文件
        model_file_wrapper.push_checkpoint_to_model_pool(None)

        # pull model文件
        model_file_wrapper.pull_checkpoint_from_model_pool(None)


if __name__ == '__main__':
    unittest.main()
