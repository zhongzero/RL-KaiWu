#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.config.config_control import CONFIG

'''

由于目前保存的模型路径比较多, 故定义个类进行整理
1. ckpt_dir, 形如/data/ckpt/sgame_ppo
2. summary_dir, 形如/data/summary/sgame_ppo
3. pb_model_dir, 形如/data/pb_model/
4. restore_dir, 形如/data/ckpt/sgame_ppo

'''
class ModelPathManager:
    def __init__(self) -> None:

        # 这里按照实际需要进行设置
        self.need_to_sync_map = {
            KaiwuDRLDefine.CKPT_DIR : True,
            KaiwuDRLDefine.RESTOR_DIR : False,
            KaiwuDRLDefine.SUMMARY_DIR : False,
            KaiwuDRLDefine.PB_MODEL_DIR : False
        }

    @property
    def ckpt_dir(self):
        return CONFIG.ckpt_dir
    
    @property
    def restore_dir(self):
        return CONFIG.restore_dir

    @property
    def summary_dir(self):
        return CONFIG.summary_dir

    @property
    def pb_model_dir(self):
        return CONFIG.pb_model_dir
    
    def get_local_and_remote_dirs(self):
        return {
            KaiwuDRLDefine.CKPT_DIR : f'{self.ckpt_dir}/{CONFIG.app}_{CONFIG.algo}',
            KaiwuDRLDefine.RESTOR_DIR : self.restore_dir,
            KaiwuDRLDefine.SUMMARY_DIR : f'{self.summary_dir}/{CONFIG.app}_{CONFIG.algo}',
            KaiwuDRLDefine.PB_MODEL_DIR : self.pb_model_dir
        }
    
    '''
    需要按照实际情况设置是否需要上传COS
    '''
    def need_to_sync(self, local_and_remote_dir):
        return self.need_to_sync_map.get(local_and_remote_dir, False)
    
    '''
    在对CONFIG.ckpt_dir/CONFIG.app_CONFIG.algo下的遍历操作时, 需要排除下面的目录
    1. convert_models_actor
    2. convert_models_learner
    3. plugins
    4. models
    '''
    def exclude_directories(self):
        return ['plugins', 'models', 'convert_models_actor', 'convert_models_learner']

MODEL_PATH_MANGER = ModelPathManager()