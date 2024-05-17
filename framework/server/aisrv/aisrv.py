#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file aisrv.py
# @brief
# @author kaiwu
# @date 2022-04-26

import os
import sys
import faulthandler
import signal
import io
import os
from framework.common.config.config_control import CONFIG
from framework.common.utils.cmd_argparser import cmd_args_parse
from framework.common.config.app_conf import AppConf
from framework.common.config.algo_conf import AlgoConf
from framework.common.utils.common_func import make_single_dir
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine

def proc_flags(configure_file):
    # 解析aisrv进程的配置
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_aisrv_configure()

    # aisrv上需要配置加载pybind11
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir + '/../../common/pybind11/zmq_ops')
    sys.path.append(base_dir + '/../../../')

    # 解析业务app的配置
    AppConf.load_conf(CONFIG.app_conf, CONFIG.svr_name)

    # 加载配置文件conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 确保框架需要的文件目录存在
    make_single_dir(CONFIG.log_dir)

def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass

'''
下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
sgame_1v1:
1. aisrv_actor_protocl = pickle

sgame_5v5:
1. aisrv_actor_protocl = protobuf

gym:
1. aisrv_actor_protocl = pickle
'''
def app_check_param():
    if CONFIG.app == KaiwuDRLDefine.APP_SGAME_1V1:
        if CONFIG.aisrv_actor_protocl != KaiwuDRLDefine.PROTOCL_PICKLE:
            CONFIG.aisrv_actor_protocl = KaiwuDRLDefine.PROTOCL_PICKLE

    elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
        if CONFIG.aisrv_actor_protocl != KaiwuDRLDefine.PROTOCL_PROTOBUF:
            CONFIG.aisrv_actor_protocl = KaiwuDRLDefine.PROTOCL_PROTOBUF

    elif CONFIG.app == KaiwuDRLDefine.APP_GYM:
        pass
    
    else:
        pass

'''
在进程启动前进行检测参数合理性
'''
def check_param():

    app_check_param()
    
    # 规则1, 如果是设置了self-play模式, 但是app文件里设置的policy是1个, 则报错
    # 规则2, 如果是设置了非self-play模式, 但是app文件里设置的policy是2个, 则报错
    # 规则3, 如果是设置了self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是1个, 则报错
    # 规则4, 如果是设置了非self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是2个, 则报错
    
    actor_addrs = CONFIG.actor_addrs
    learner_addrs = CONFIG.learner_addrs

    if int(CONFIG.self_play):
        if len(AppConf[CONFIG.app].policies) == 1:
            print(f'self-play模式, 但是配置的policy维度为1, 请修改配置后重启进程')
            return False

        if len(actor_addrs) == 1 or len(learner_addrs) == 1 :
            print(f'self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为1, 请修改配置后重启进程')
            return False
        
    else:
        if len(AppConf[CONFIG.app].policies) == 2:
            print(f'非self-play模式, 但是配置的policy维度为2, 请修改配置后重启进程')
            return False

        if len(actor_addrs) == 2 or len(learner_addrs) == 2 :
            print(f'非self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为2, 请修改配置后重启进程')
            return False

    return True

'''
启动命令样例: python3 aisrv.py --conf=/data/projects/gorge_walk_v2/conf/framework/aisrv.toml
'''

def main():

    os.chdir(CONFIG.project_root)

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_AISRV)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print('conf param error, please check')
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 启动进程
    if KaiwuDRLDefine.AISRV_FRAMEWORK_SOCKETSERVER == CONFIG.aisrv_framework:
        # python版本
        from framework.server.aisrv.aisrv_socketserver import AiSrv, AiSrvHandle

        server = AiSrv((CONFIG.aisrv_ip_address, CONFIG.aisrv_server_port), AiSrvHandle)
        server.serve_forever()
    
    elif KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
        # C++版本
        from framework.server.aisrv.aisrv_server import AiServer
        server = AiServer()
        server.run()

    elif KaiwuDRLDefine.AISRV_FRAMEWORK_ARENA == CONFIG.aisrv_framework:
        # python版本
        from framework.server.aisrv.aisrv_arena_server import AiServer
        server = AiServer()
        server.run()

    else:
        print(f"not support {CONFIG.aisrv_framework}, only support {KaiwuDRLDefine.AISRV_FRAMEWORK_TRPC} or {KaiwuDRLDefine.AISRV_FRAMEWORK_SOCKETSERVER} or {KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL}")


if __name__ == '__main__':
    sys.exit(main())
