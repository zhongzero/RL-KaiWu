#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
import unittest
import os
from framework.common.config.config_control import CONFIG
from framework.common.utils.cos_utils import COSSave
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
import warnings
warnings.simplefilter('ignore', ResourceWarning)

class CosUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/learner.toml')
        CONFIG.parse_learner_configure()

        self.cos_save = COSSave(None, CONFIG.cos_secret_id, CONFIG.cos_secret_key, CONFIG.cos_region, CONFIG.cos_token, scheme='https')

    def test_cos_connect(self):
        self.cos_save.connect_to_cos()
    
    def test_cos_bucket(self):

        self.cos_save.connect_to_cos()

        # create bucket
        self.cos_save.create_bucket('examplebucket-1250000000')

        # query bucket
        self.cos_save.query_bucket_list()

        # delete bucket
        self.cos_save.delete_bucket('examplebucket-1250000000')

    def test_cos_push_pull(self):

        self.cos_save.connect_to_cos()
        now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H')

        cos_bucket_key = f'{KaiwuDRLDefine.COS_BUCKET_KEY}{CONFIG.app}'

        key= f'{cos_bucket_key}{now_str}{KaiwuDRLDefine.KAIWU_MODEL_CKPT}.100'

        file_name = '/data/projects/kaiwu-fwk/framework/common/utils/cos_utils_test.py'
        file_name_new = '/data/projects/kaiwu-fwk/framework/common/utils/cos_utils_test_new.py'

        # push
        self.cos_save.push_to_cos(file_name, CONFIG.cos_bucket, key)

        # query
        query_list = self.cos_save.query_object_list(CONFIG.cos_bucket, cos_bucket_key)
        print(f'query_list is {query_list}')

        # pull
        self.cos_save.get_from_cos(CONFIG.cos_bucket, key, file_name_new)     

        # query
        query_list = self.cos_save.query_object_list(CONFIG.cos_bucket, cos_bucket_key)
        print(f'query_list is {query_list}')

if __name__ == '__main__':
    unittest.main()
