#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.common.config.config_control import CONFIG
from framework.common.checkpoint.model_file_sync import ModelFileSync

class ModelFileSyncWrapper(object):
    def __init__(self) -> None:
        
        self.ckpt_sync_warper = None
    
    def init(self):
        # 如果配置项中启用了模型池（modelpool），则实例化ModelFileSync对象并启动同步
        if CONFIG.use_modelpool():
            self.ckpt_sync_warper = ModelFileSync()
            self.ckpt_sync_warper.start()
        # 如果未启用模型池，则将ckpt_sync_warper设置为NotImplemented
        else:
            self.ckpt_sync_warper = NotImplemented
    
    def stop(self):
        # 如果配置项中启用了模型池（modelpool），则停止模型文件同步
        if CONFIG.use_modelpool():
            self.ckpt_sync_warper.stop()
        else:
            self.ckpt_sync_warper = NotImplemented