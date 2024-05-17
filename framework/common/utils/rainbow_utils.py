#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# ref:https://git.woa.com/rainbow/python-sdk/tree/master
# need pip install rainbow-sdk -U -i https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

from rainbow_sdk.rainbow_client import RainbowClient

class RainbowUtils(object):

    def __init__(self, rainbow_url, app_id, user_id, secret_key, env_name, logger):
        self.rainbow_url = rainbow_url
        self.app_id = app_id
        self.user_id = user_id
        self.secret_key = secret_key
        self.env_name = env_name
        self.logger = logger

        # 可能存在多个进程的场景
        self.rainbow_group = []

        # 注意参数设置正确
        self.init_param = {
            "connectStr": self.rainbow_url,
            "isUsingFileCache" : True,
            "tokenConfig": {
                "app_id": self.app_id,
                "user_id": self.user_id,
                "secret_key": self.secret_key,
            },
        }
    
    '''
    返回本次实例字符串
    '''
    @property
    def identity(self):
        return  f'rainbow_url is {self.rainbow_url}, rainbow_app_id is {self.app_id}, \
         rainbow_user_id is {self.user_id}, rainbow_secret_key is {self.secret_key}, \
         rainbow_group is {self.rainbow_group}, rainbow_env_name is {self.env_name}'

    '''
    进程从七彩石读取配置条目
    grop 是进程名字

    返回的格式形如:
    {'code': 0, 'data': 'main:send_sample_size: 300', 'message': 'OK'}
    返回的数据格式: result_code, data, result_msg
    '''
    def read_from_rainbow(self, rainbow_group):
        rc = RainbowClient(self.init_param)
        self.rainbow_group.append(rainbow_group)

        res = rc.get_configs_v3(rainbow_group, env_name=self.env_name, key=rainbow_group)

        if res:
            return res['code'], res['data'], res['message']

        # 如果res没有, 则返回错误字符串
        return 1, None, 'rc.get_configs_v3 failed'

    '''
    进程将配置条目提交到七彩石

    需要采用http形式, 故暂时不考虑使用, 其内容如下:
    https://iwiki.woa.com/pages/viewpage.action?pageId=98145038
    对应的sdk是这个
    https://git.woa.com/rainbow/python-admin

    '''
    def write_to_rainbow(self, configure_data):
        if not configure_data:
            return