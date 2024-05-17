#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import json
import warnings
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.config.config_control import CONFIG
from framework.common.utils.http_utils import http_utils_post, http_utils_request
from framework.common.utils.common_func import get_host_ip

'''
aisrv、actor、learner的进程与alloc进程交互的类, 主要包括：
1. 注册：/api/registry
2. 请求实例：/api/get
'''
class AllocUtils(object):

    def __init__(self, logger) -> None:
        super().__init__()

        # 舍弃ResourceWarning的警告信息
        warnings.simplefilter('ignore', ResourceWarning)

        # alloc里关于KaiwuDRL进程的配置, map形式, 一般不做修改
        self.SERVER_ROLE_CONFIGURE = {
            KaiwuDRLDefine.SERVER_AISRV : int(CONFIG.alloc_process_role_aisrv),
            KaiwuDRLDefine.SERVER_ACTOR : int(CONFIG.alloc_process_role_actor),
            KaiwuDRLDefine.SERVER_LEARNER : int(CONFIG.alloc_process_role_learner),
            KaiwuDRLDefine.SERVER_CLIENT: int(CONFIG.alloc_process_role_client),
            KaiwuDRLDefine.SERVER_ARENA : int(CONFIG.alloc_process_role_arena)
        }

        # alloc里关于KaiwuDRL的进程的端口, map形式, 一般不做修改
        self.SERVER_PORT_CONFIGURE = {
            KaiwuDRLDefine.SERVER_AISRV : int(CONFIG.aisrv_server_port),
            KaiwuDRLDefine.SERVER_ACTOR : int(CONFIG.zmq_server_port),
            KaiwuDRLDefine.SERVER_LEARNER : int(CONFIG.reverb_svr_port),
            KaiwuDRLDefine.SERVER_CLIENT: int(CONFIG.client_svr_port),
            KaiwuDRLDefine.SERVER_ARENA : int(CONFIG.arena_svr_port)
        }

        # alloc里关于KaiwuDRL进程比例的配置, map形式, 一般不做修改
        # 比如aisrv选择的比例是1, 则意味着100台battlesrv与100台aisrv建立连接
        # 比如aisrv选择的比例是2, 则意味着100台battlesrv与50台aisrv建立连接, 每2台battlesrv连接1台aisrv, 有50台aisrv是空闲的
        self.SERVER_ASSIGN_LIMIT = {
            KaiwuDRLDefine.SERVER_AISRV : int(CONFIG.alloc_process_assign_limit_aisrv),
            KaiwuDRLDefine.SERVER_ACTOR : int(CONFIG.alloc_process_assign_limit_actor),
            KaiwuDRLDefine.SERVER_LEARNER : int(CONFIG.alloc_process_assign_limit_learner),
            KaiwuDRLDefine.SERVER_CLIENT: int(CONFIG.alloc_process_assign_limit_client),
            KaiwuDRLDefine.SERVER_ARENA: int(CONFIG.alloc_process_assign_limit_arena)
        }

        # 日志句柄
        self.logger = logger

        self.set_name = CONFIG.set_name
        self.role = self.SERVER_ROLE_CONFIGURE.get(CONFIG.svr_name)
        # IP:端口形式
        self.addr = f'{get_host_ip()}:{self.SERVER_PORT_CONFIGURE.get(CONFIG.svr_name)}'

        self.alloc_addr = f'http://{CONFIG.alloc_process_address}'

        # task_id
        self.task_id = CONFIG.task_id

        '''
        由于存在self_play模式和非self_play模式, 故这里的get参数由调用者设置
        1. 非self_play模式, 增加参数target_role, 形如:
        {
            "addr":"7.7.7.7:7777", // 字符串。自己的ip:port。或者一个唯一的id
            "target_role":2 // 整数。 请求哪种实例, 含义同上
        }
        2. self_play模式, 增加参数set_list
        {
            "addr":"7.7.7.7:7777", // 字符串。自己的ip:port。或者一个唯一的id
            "set_list": [
                {
                    "set": "set1",
                    "target_role": 3
                },
                {
                    "set": "set2",
                    "target_role": 4
                }
            ]
        }

        '''
        self.get_param = {
            "addr": self.addr,
            "task_id": self.task_id,
        }

    # 注册
    def registry(self):
        # 由于每次可能更新post参数, 故post参数需要放在这里组装
        self.assign_limit = self.SERVER_ASSIGN_LIMIT.get(CONFIG.svr_name)
        self.post_param = {
            "set": self.set_name,
            "role": self.role,
            "addr": self.addr,
            "assign_limit": self.assign_limit,
            "task_id": self.task_id,
        }
        url = f'{self.alloc_addr}/api/registry'
        resp = http_utils_post(url, self.post_param)
        if not resp:
            return False, f"http failed, alloc_url is {url}"

        if resp['code'] == 0:
            return True, None

        return resp['code'], resp['msg']

    '''
    每个具体的server获取到的进程数量
    '''
    def get_server_process_count(self, srv_name):
        count = 0
        if not srv_name:
            return count
        
        if srv_name == KaiwuDRLDefine.SERVER_ACTOR:
            count = int(CONFIG.aisrv_connect_to_actor_count)
        elif srv_name == KaiwuDRLDefine.SERVER_LEARNER:
            count = int(CONFIG.aisrv_connect_to_learner_count)
        elif srv_name == KaiwuDRLDefine.SERVER_AISRV:
            count = 1
        elif srv_name == KaiwuDRLDefine.SERVER_ARENA:
            count = int(CONFIG.aisrv_connect_to_arena_count)
        else:

            # 未来扩展
            pass

        return count
    
    def get(self, srv_name, set_name, self_play_set_name):
        '''
        获取实例, 需要填写目的端的role, set_name, self_play_set_name
        '''
        
        target_role = self.SERVER_ROLE_CONFIGURE[srv_name]

        url = f'{self.alloc_addr}/api/get'
        set_list = []

        count = self.get_server_process_count(srv_name)
        if not count:
            # 返回错误场景
            return True, None, None
        
        if not int(CONFIG.self_play):

            # 支持单个aisrv获取多个actor/learner的功能
            for i in range(int(count)):
                set_list.append(
                    {
                        'set': set_name,
                        'target_role': target_role
                    })

            self.get_param['set_list'] = set_list
        else:
            for i in range(int(count)):
                set_list.append(
                    {
                        'set': set_name,
                        'target_role': target_role
                    })

            # 获取actor时需要self_play_set_name, 获取learner时则不需要
            if set_name != self_play_set_name:
                if target_role == self.SERVER_ROLE_CONFIGURE.get(KaiwuDRLDefine.SERVER_ACTOR):
                    for i in range(int(count)):
                        set_list.append(
                            {
                                'set': self_play_set_name,
                                'target_role': target_role
                            })

            self.get_param['set_list'] = set_list

        # 目前需要采用post方式获取
        resp = http_utils_post(url, self.get_param)
        # resp = http_utils_request(url, self.get_param)
        if not resp:
            # 为了适配code为0时成功, 非0时失败
            return True, "http failed", None

        # 这里直接返回了code, msg, content, 需要和alloc约定
        return resp['code'], resp['msg'], resp['content']

    '''
    针对self_play模式, 获取actor和learner方法:
    {'set_list':
    [{'set': 'set2', 'role': 3, 'addr': '11.236.248.174:8888'},
    {'set': 'set1', 'role': 3, 'addr': '11.236.243.56:8888'},
    {'set': 'set2', 'role': 4, 'addr': '11.236.249.227:9999'},
    {'set': 'set1', 'role': 4, 'addr': '11.236.248.26:9999'}
    ]}
    '''

    def get_actor_learner_address(self, content, target_role):
        self_play_actor_learner_address_list = []
        self_play_old_actor_learner_address_list = []

        if not content:
            return self_play_actor_learner_address_list, self_play_old_actor_learner_address_list

        try:
            content = json.loads(content, strict=False)
            set_lists = content.get('set_list')
            if not set_lists:
                return self_play_actor_learner_address_list, self_play_old_actor_learner_address_list

            for adderss in set_lists:
                set_name = adderss.get('set', None)
                role = adderss.get('role', None)
                addr = adderss.get('addr', None)

                # 如果数据不完整, 直接continue
                if not set_name or not role or not addr:
                    continue

                # 区分self_play和old_self_play
                if set_name == CONFIG.set_name and role == target_role:
                    self_play_actor_learner_address_list += addr.split(',')

                if set_name == CONFIG.self_play_set_name and role == target_role:
                    self_play_old_actor_learner_address_list += addr.split(',')

        except Exception as e:
            self.logger.error(
                f"get learner IP fail, will retry next time, error_cod is {str(e)}")

        return self_play_actor_learner_address_list, self_play_old_actor_learner_address_list

    '''
    aisrv从alloc获取actor和learner地址
    下面是get接口返回情况:
    1. code为0时, content为IP列表(字符串以逗号分割), msg为success字符串
    2. code非0表示失败, msg为具体失败报错, content为空
    3. 具体的返回格式, 需要和alloc服务确定清楚

    返回格式举例:
    code 0
    msg success
    content {"set_list":[{"set":"sgame_5v5_set1","role":3,"addr":"172.17.0.3:8888"},{"set":"sgame_5v5_set1","role":3,"addr":"172.17.0.3:8888"},{"set":"sgame_5v5_set1","role":3,"addr":"172.17.0.3:8888"},{"set":"sgame_5v5_set1","role":3,"addr":"172.17.0.3:8888"}]}
    '''

    def get_actor_learner_ip(self, set_name, self_play_set_name):

        if not set_name:
            return None, None, None, None

        # 获取actor地址
        if not int(CONFIG.self_play):
            actor_address = []
            learner_address = []

            code, msg, content = self.get(KaiwuDRLDefine.SERVER_ACTOR, set_name, self_play_set_name)
            if code:
                self.logger.error(
                    f"get actor IP fail, will retry next time, error_cod is {msg}")
            else:
                # 注意需要和alloc约定格式, 返回json串
                content = json.loads(content, strict=False)
                for set_list in content['set_list']:
                    actor_address.append(set_list['addr'])
            
            code, msg, content = self.get(KaiwuDRLDefine.SERVER_LEARNER, set_name, self_play_set_name)
            if code:
                self.logger.error(
                    f"get learner IP fail, will retry next time, error_cod is {msg}")
            else:
                # 注意需要和alloc约定格式, 返回json串
                content = json.loads(content, strict=False)
                for set_list in content['set_list']:
                    learner_address.append(set_list['addr'])

            return actor_address, learner_address, None, None

        else:
            self_play_actor_address = None
            self_play_old_actor_address = None
            self_play_learner_address = None
            self_play_old_learner_address = None

            # 一次获取到set_name和self_play_set_name的actor和learner地址
            code, msg, content = self.get(
                KaiwuDRLDefine.SERVER_ACTOR, set_name, self_play_set_name)
            if code:
                self.logger.error(
                    f"get actor IP fail, will retry next time, error_cod is {msg}")
            else:
                # 注意需要和alloc约定格式, 目前是以,号分割的
                self_play_actor_address, self_play_old_actor_address = self.get_actor_learner_address(content, self.SERVER_ROLE_CONFIGURE[KaiwuDRLDefine.SERVER_ACTOR])
            
            code, msg, content = self.get(KaiwuDRLDefine.SERVER_LEARNER, set_name, self_play_set_name)
            if code:
                self.logger.error(
                    f"get learner IP fail, will retry next time, error_cod is {msg}")
            else:
                # 注意需要和alloc约定格式, 目前是以,号分割的
                self_play_learner_address, self_play_old_learner_address = self.get_actor_learner_address(
                    content, self.SERVER_ROLE_CONFIGURE[KaiwuDRLDefine.SERVER_LEARNER])

            return self_play_actor_address, self_play_old_actor_address, self_play_learner_address, self_play_old_learner_address

    # battlesrv获取aisrv地址, learner获取aisrv地址
    def get_aisrv_ip(self, set_name, self_play_set_name) -> list:

        if not set_name:
            return None

        # 获取aisrv地址
        aisrv_address = []
        code, msg, content = self.get(
            KaiwuDRLDefine.SERVER_AISRV, set_name, self_play_set_name)
        if code:
            self.logger.error(
                f"get aisrv IP fail, will retry next time, error_cod is {msg}")
            return None
        else:
            # 注意需要和alloc约定格式, 返回json串
            content = json.loads(content, strict=False)
            for set_list in content['set_list']:
                aisrv_address.append(set_list['addr'])

        return aisrv_address
    
    def get_arena_ip(self, set_name, self_play_set_name) -> list:

        if not set_name:
            return None
        
        # 获取arena地址
        arena_address = []
        code, msg, content = self.get(
        KaiwuDRLDefine.SERVER_ARENA, set_name, self_play_set_name)
        if code:
            self.logger.error(
                f"get aisrv IP fail, will retry next time, error_cod is {msg}")
            return None
        else:
            # 注意需要和alloc约定格式, 返回json串
            content = json.loads(content, strict=False)
            for set_list in content['set_list']:
                arena_address.append(set_list['addr'])

        return arena_address
    
    # 按照role来获取单个task_id下的role类型的进程
    def get_task_address_by_role(self, role):
        url = f'{self.alloc_addr}/api/getTaskAllAddrs'
        get_param = {
                "role" : role,
                "task_id" : self.task_id,
            }
        
        # 目前需要采用post方式获取
        resp = http_utils_post(url, get_param)
        # resp = http_utils_request(url, get_param)
        if not resp:
            # 为了适配code为0时成功, 非0时失败
            return True, "http failed", None
        
        # 这里直接返回了code, msg, content, 需要和alloc约定
        return resp['code'], resp['msg'], resp['content']
    
    # 按照srv_name来获取单个task_id或者set_name下的地址
    def get_all_address_by_srv_name(self, srv_name):
        all_address = []
        if not srv_name:
            return all_address

        code, msg, content = self.get_task_address_by_role(self.SERVER_ROLE_CONFIGURE[srv_name])
        if code:
            self.logger.error(
                f"get actor IP fail, will retry next time, error_cod is {msg}")
            return all_address
        else:
            # 注意需要和alloc约定格式, 返回json串
            content = json.loads(content, strict=False)
            for set_list in content['set_list']:
                all_address.append(set_list['addr'])
        
        return all_address
