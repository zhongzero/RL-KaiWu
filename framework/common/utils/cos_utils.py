#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos import CosServiceError
from framework.common.config.config_control import CONFIG
import warnings
warnings.simplefilter('ignore', ResourceWarning)

# ref:https://cloud.tencent.com/document/product/436/12269
# need pip install -U cos-python-sdk-v5

class COSSave(object):
    '''
    设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在CosConfig中移除, 请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
    secret_id, 请登录访问管理控制台进行查看和管理, https://console.cloud.tencent.com/cam/capi
    secret_key, 请登录访问管理控制台进行查看和管理, https://console.cloud.tencent.com/cam/capi
    region, 已创建桶归属的region可以在控制台查看, https://console.cloud.tencent.com/cos5/bucket, COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
    token, 如果使用永久密钥不需要填入token, 如果使用临时密钥需要填入, 临时密钥生成和使用指引参见https://cloud.tencent.com/document/product/436/14048
    scheme, 指定使用 http/https 协议来访问 COS, 默认为 https, 可不填
    bucket_name, 在申请时即需要指定, 一般不做修改, 单个业务即1个bucket_name即可
    key, 分层目录, 目前的格式为/kaiwu_drl_models/日期/model.ckpt.step
    '''
    def __init__(self, logger, secret_id, secret_key, region, token, scheme='https') -> None:
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.token = token
        self.scheme = scheme

        # 需要增加内部域名的指定
        self.endpoint  = f'cos-internal.{self.region}.tencentcos.cn'
        #self.service_domain  = 'service.cos.tencentcos.cn'

        self.client = None

        # 日志由使用者传递日志句柄
        self.logger = logger

    '''
    返回本次实例字符串
    '''
    @property
    def identity(self):
        return f'region is {self.region}, SecretId is {self.secret_id}, SecretKey is {self.secret_key}, Token is {self.token}, Scheme is {self.scheme}'

    def connect_to_cos(self):
        config = CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key, Token=self.token, Scheme=self.scheme, Endpoint=self.endpoint)
        self.client = CosS3Client(config)

        if self.logger:
            self.logger.debug(f'cos connect success, {self.identity}')
    
    def create_bucket(self, bucket_name):
        if not self.client or not bucket_name:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or bucket_name is None')
            return

        try:
            response = self.client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            if self.logger:
                self.logger.error(f'cos error to create bucket_name {bucket_name} error is {str(e)}')
    
    def delete_bucket(self, bucket_name):
        if not self.client or not bucket_name:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or bucket_name is None')
            return
        
        try:
            response = self.client.delete_bucket(Bucket=bucket_name)
        except Exception as e:
            if self.logger:
                self.logger.error(f'cos error to delete bucket_name {bucket_name} error is {str(e)}')

    def query_bucket_list(self):
        if not self.client:
            if self.logger:
                self.logger.error(f'cos error, self.client is None')
            return None

        try:
            response = self.client.list_buckets()

            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"cos error to query bucket list, error is {str(e)}")
            
            return None
    
    def push_to_cos(self, file_name, bucket_name, key):
        if not self.client or not file_name or not bucket_name or not key:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or file_name is None or bucket_name is None or key is None')
            return False

        with open(file_name, 'rb') as fp:
            try:
                response  = self.client.put_object(Bucket=bucket_name, Body=fp, Key=key, StorageClass='STANDARD', EnableMD5=False)
                
                return True
            except Exception as e:
                if self.logger:
                    self.logger.error(f"cos error to push to cos file_name {file_name} bucket_name {bucket_name}  key {key} error is {str(e)}")
                
                return False
    
    def query_object_list(self, bucket_name, prefix):
        if not self.client or not bucket_name or not prefix:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or bucket_name is None or prefix is None')
            return None

        try:
            response = self.client.list_objects(Bucket=bucket_name, Prefix=prefix)
            
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"cos error to query object list, bucket_nbame {bucket_name} prefix {prefix}, error is {str(e)}")
            
            return None

    def get_from_cos(self, bucket_name, key, output_file_name):
        if not self.client or not bucket_name or not key or not output_file_name:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or bucket_name is None or key is None or output_file_name is None')
            return False

        try:
            response = self.client.get_object(Bucket=bucket_name, Key=key)
            response['Body'].get_stream_to_file(output_file_name)

            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"cos error to get from cos, bucket_nbame {bucket_name} key {key} output_file_name {output_file_name} error is {str(e)}")
            
            return False

    def delete_from_cos(self, bucket_name, key):
        if not self.client or not bucket_name or not key:
            if self.logger:
                self.logger.error(f'cos error, self.client is None or bucket_name is None or key is None')
            return

        try:
            response = self.client.delete_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            if self.logger:
                self.logger.error(f"cos error to delete from cos, bucket_nbame {bucket_name} key {key} error is {str(e)}")