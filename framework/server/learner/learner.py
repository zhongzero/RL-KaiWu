#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file learner.py
# @brief
# @author kaiwu
# @date 2022-04-26

import faulthandler
import signal
import os
import io
import sys
import time
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG
from framework.common.utils.cmd_argparser import cmd_args_parse
from framework.common.config.algo_conf import AlgoConf
from framework.common.config.app_conf import AppConf
from framework.common.utils.common_func import get_local_rank

def proc_flags(configure_file):
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_learner_configure()

    # actor上需要配置加载pybind11
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir + '/../../common/pybind11/zmq_ops')

    # 加载配置文件conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 加载配置文件conf/app_conf.json
    AppConf.load_conf(CONFIG.app_conf)

    # learner需要设置在GPU机器上运行
    if 'GPU' == CONFIG.learner_device_type:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    
    # 设置TensorFlow日志级别
    set_tensorflow_log_level()

def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass

'''
下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
sgame_1v1:
1. train_batch_size = 1024

sgame_5v5:
1. train_batch_size = 512

gym:
1. 
'''
def app_check_param():
    pass


'''
在进程启动前进行检测参数合理性
'''
def check_param():

    app_check_param()
    
    # learner的批处理大小需要小于等于replay_buff的capacity
    if CONFIG.train_batch_size > CONFIG.replay_buffer_capacity:
        print(f'train_batch_size {CONFIG.train_batch_size} > replay_buffer_capacity {CONFIG.replay_buffer_capacity}')
        return False
    
    return True

def train_loop():
    # 根据配置文件conf/learner_conf.json找到本次使用的train类
    train = AlgoConf[CONFIG.algo].trainer()

    # 临时方案TODO, 启动learner_server, 包括learner_server_reverb和learner_server_zmq
    if CONFIG.use_learner_server:
        from framework.server.learner.learner_server import LearnerServerReverb, LearnerServerZmq

        # LearnerServerReverb进程集合
        learner_server_reverbs = []
        for i in range(int(CONFIG.revervb_utils_count)):
            learner_server_reverb = LearnerServerReverb(i)
            learner_server_reverb.start()
            learner_server_reverbs.append(learner_server_reverb)
        
        # 人为的增加sleep时间
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)

        learner_server_zmq = LearnerServerZmq(learner_server_reverbs)
        learner_server_zmq.start()

    train.loop()

'''
启动命令样例: python3 learner.py --conf=/data/projects/kaiwu-fwk/conf/framework/learner.toml
'''

def main():

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_LEARNER)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print('conf param error, please check')
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 开始轮训处理
    train_loop()

if __name__ == '__main__':
    sys.exit(main())
