#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import toml
from dynaconf import Dynaconf
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.singleton import Singleton


@Singleton
class ConfigControl(object):

    '''
    目录文件规则:
    1. 用户修改: app下的, 其中configure_app.toml的修改可以影响到KaiwuDRL
    2. KaiwuDRL修改: framework下的, 在对外时可以进行隐藏

    下面是目前配置文件列表:
    .
    ├── configure_app.toml
    ├── framework
    │   ├── actor.toml
    │   ├── aisrv.toml
    │   ├── client.toml
    │   ├── configure.toml
    │   └── learner.toml
    └── README.md
    '''

    def __init__(self) -> None:
        self.project_root = os.path.abspath(
            os.path.dirname(__file__) + ''.join([os.path.sep + ".."] * 3))
        # 每个进程各自的配置文件, 分为业务态和系统态配置文件
        self.config_file = None

        # 公共的配置文件
        self.config_system_file = None
        self.config = Dynaconf()

        # 支持业务自定义的配置文件
        self.app_configure_file = None

    def check_option_valid(self, option) -> bool:
        if option not in ['main', 'aisrv', 'actor', 'learner', 'client']:
            return False

        return True
    
    def read_from_env(self) -> None:
        '''
        支持从环境变量里读取值并且更新配置值, 主要用于k8s容器环境
        根据内存里的数据写回到配置文件
        1. option, 哪种进程
        2. 需要修改的key/value对
        '''
        env_list = os.environ
        if not env_list:
            return

        for key, value in env_list.items():
            key = key.lower()
            if key in CONFIG.__dict__:
                if key in ['actor_addrs', 'learner_addrs']:
                    CONFIG.__dict__[key] = eval(value)
                    continue
                # 获取配置字段的类型
                value_type = type(CONFIG.__dict__[key])
                if value_type is bool:
                    CONFIG.__dict__[key] = True if value == 'True' else False
                elif value_type is int:
                    CONFIG.__dict__[key] = int(value)
                elif value_type is float:
                    CONFIG.__dict__[key] = float(value)
                else:
                    CONFIG.__dict__[key] = value

    def write_to_config(self, key_values: dict) -> None:
        """_summary_
            将需要修改的内容写回内存
        Args:
            key_values: 需要写回内存的key/value对
        """
        for key, value in key_values.items():
            self.__dict__[key] = value

    def save_to_file_with_multi_table(self, config_file, to_change_key_values=None) ->None:
        if not config_file or not to_change_key_values:
            return
        
        '''
        需注意采用self.config.write写法会导致多个进程里的内容叠加
        这里如果文件不存在或者别的进程打开的, 需要加上try catch模式
        失败后下一次再更新文件即可
        '''
        try:
            with open(config_file, 'w+') as file:
                toml.dump(to_change_key_values, file)
        except Exception as e:
                    pass  

    def save_to_file(self, option, to_change_key_values=None) ->None:
        '''
        根据内存里的数据写回到配置文件
        1. option, 哪种进程
        2. 需要修改的key/value对
        '''
        if not self.check_option_valid(option):
            return

        # 回写配置到config
        for key, value in to_change_key_values.items():
            self.config.set(f'{option}.{key}', value)

        if option == 'main':
            config_file = self.config_system_file
        else:
            config_file = self.config_file

        '''
        需注意采用self.config.write写法会导致多个进程里的内容叠加
        这里如果文件不存在或者别的进程打开的, 需要加上try catch模式
        失败后下一次再更新文件即可
        '''
        try:
            with open(config_file, 'w+') as file:
                toml.dump({option: self.config[f'{option}']},file)
        except Exception as e:
                    pass  
    
    '''
    读取配置文件内容, 如果有具体的需要读取的值, 则返回
    '''

    def read_from_file(self, option, to_read_keys=None) -> dict:
        if not self.check_option_valid(option):
            return

        if 'main' == option:
            self.config.load_file(self.config_system_file)
        else:
            self.config.load_file(self.config_file)

        result = {}
        if to_read_keys:
            for key in to_read_keys:
                result[key] = self.config.get(f'{option}.{key}')

        return result

    def load_data_from_config(self, table, file_path) -> None:
        """load_data_from_config

        table: 配置文件的表名
        file_path: 配置文件路径
        通用逻辑， 从toml配置文件读取数据并赋给类
        """
        # 采用绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.project_root, file_path)
        
        # 路径检查
        if not file_path or not os.path.exists(file_path):
            print(f'the config path:{file_path} is not exists, please check!')
            return

        # 加载配置文件
        self.config.load_file(file_path)

        config_items_map = self.config.get(table, {})
        load_data_suc_count = 0

        # 读取各项配置
        for k, v in config_items_map.items():
            # 根据配置项类型进行检测
            if isinstance(v, int) and v > 0:
                setattr(self, k, v)
                load_data_suc_count += 1
            elif isinstance(v, str) and v != "":
                setattr(self, k, v)
                load_data_suc_count += 1
            elif isinstance(v, bool) and v in [True, False]:
                setattr(self, k, v)
                load_data_suc_count += 1
            elif isinstance(v, dict) and v != {}:
                setattr(self, k, v)
                load_data_suc_count += 1
            else:
                setattr(self, k, v)
                load_data_suc_count += 1
            
        if load_data_suc_count != len(config_items_map):
            print(f'Invalid value for {k}: {v}. Expected positive integer, non-empty string, or boolean, please check')

    def parse_main_configure(self) -> None:
        """parse_main_configure

        解析configure.toml配置文件
        """
        self.load_data_from_config('main', self.config_system_file)

        # 加载app配置
        self.parse_app_configure()
    
        self.read_from_env()

    def parse_app_configure(self) -> None:
        """parse_app_configure

        解析configure_app配置文件
        """
        if not self.app_configure_file:
            return
        
        # 拼接绝对路径
        self.app_configure_file = os.path.join(self.project_root, self.app_configure_file)
        self.load_data_from_config('app', self.app_configure_file)

        '''
        下面是set(逻辑上属于单个训练任务)的配置, 注意self_play模式时, 加载不同set下的actor和learner即可
        由于各个业务的set不一样, 这里的set_name默认就是app + set_name做下隔离, 故不同的app一定不能起的名字一样, 比如sgame_1v1, sgame_5v5
        '''
        self.set_name = f"{self.app}_{self.set_name}"
        self.self_play_set_name = f"{self.app}_{self.self_play_set_name}"

        # self_play配置项
        if self.self_play:
            self.app_conf = self.selfplay_app_conf
        else:
            self.app_conf = self.noselfplay_app_conf

    def parse_aisrv_configure(self) -> None:
        """parse_aisrv_configure

        解析aisrv配置文件
        """
        if not self.config_file:
            return

        self.load_data_from_config('aisrv', self.config_file)

        # 默认加载configure.toml文件
        self.parse_main_configure()

    def parse_actor_configure(self) -> None:
        """parse_actor_configure

        解析actor配置文件
        """
        if not self.config_file:
            return

        self.load_data_from_config('actor', self.config_file)

        # 默认加载configure.toml文件
        self.parse_main_configure()

    def parse_learner_configure(self) -> None:
        """parse_learner_configure

        解析learner配置文件
        """
        if not self.config_file:
            return

        self.load_data_from_config('learner', self.config_file)

        # 默认加载configure.toml文件
        self.parse_main_configure()

    # 解析client配置
    def parse_client_configure(self):
        if not self.config_file:
            return

        self.load_data_from_config('client', self.config_file)

        # 默认加载configure.toml文件
        self.parse_main_configure()

    def set_configure_file(self, config_file):
        if not config_file:
            return

        self.config_file = config_file

        dir_and_file_name = os.path.split(self.config_file)
        dir_name, file_name = dir_and_file_name[0], dir_and_file_name[1]

        config_system_file_name = f'{dir_name}/configure.toml'
        if os.path.exists(config_system_file_name):
            self.config_system_file = config_system_file_name
    
    '''
    重载__getattr__, 如果属性不存在返回默认值, 由于不知道该属性的类型, 只能设置为string
    '''
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return KaiwuDRLDefine.CONFIG_DEFAULT_STRING

    # 下面是控制replaybuff类型
    def use_reverb(self):
        return CONFIG.replay_buffer_type == 'reverb'

    def use_tf_uniform(self):
        return CONFIG.replay_buffer_type == 'tf_uniform'

    def use_mempool(self):
        return CONFIG.replay_buffer_type == 'mempool'

    # 控制采用的learn <--> actor之间的Model文件同步方式
    def use_modelpool(self):
        return CONFIG.ckpt_sync_way == 'modelpool'


CONFIG = ConfigControl()
