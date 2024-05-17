#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from distutils.dir_util import remove_tree
import hashlib
import os
import shutil
import tempfile
import time
import traceback
import schedule
import re
import datetime
from multiprocessing import Process, Queue, Value
from framework.common.utils.tf_utils import *
from framework.common.checkpoint.model_pool_apis import ModelPoolAPIs
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import clean_dir, insert_any_string, make_tar_file, set_schedule_event, tar_flie_extract, TimeIt
from framework.common.utils.common_func import get_first_last_line_from_file, fix_checkpoint_file, get_random_line_from_file
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.monitor.monitor_proxy import MonitorProxy
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.checkpoint.model_path_manager import MODEL_PATH_MANGER

# 如果是tensorrt的加载dump_weights
if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine().MODEL_TENSORRT:
    from framework.server.cpp.tools.dump_tf_weights import dump_weights
    
'''
actor <--> learner之间传递的Model文件任务重, 单个进程处理
'''
class ModelFileSync(Process):

    def __init__(self) -> None:
        super(ModelFileSync, self).__init__()

        self.exit_flag = Value('b', False)

        # 调用modelpool, model_pool_addrs 在配置conf/framework/configure.toml, 格式形如'127.0.0.1:10013', 以,号分割
        self.remote_addrs = CONFIG.modelpool_remote_addrs

        self.model_pool_apis = ModelPoolAPIs(self.remote_addrs)
        self.model_pool_apis.check_server_set_up()

        # modelpool 相关统计, 需要区分是actor还是learner进程
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            self.push_to_model_pool_succ_cnt = 0
            self.push_to_model_pool_err_cnt = 0
        else:
            self.pull_from_model_pool_succ_cnt = 0
            self.pull_from_model_pool_err_cnt = 0

    '''  
    创建上传和下载model文件的临时目录
    1. learner上的上传目录为/tmp下临时目录, 需要代码里删除, 形如/tmp/tmpyiu2tmhp/
    2. actor的下载目录为ckpt_dir下的业务名_算法名的plugins, 代码里不需要删除, 形如/data/ckpt/sgame_ppo/plugins/
    3. actor的加载model的目录为ckpt_dir下的业务名_算法名的modles, 代码里不需要删除, 形如/data/ckpt/sgame_ppo/modles/
    '''
    def make_model_dirs(self, logger):
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            self.plugins_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/plugins'
            if not os.path.exists(self.plugins_path):
                os.makedirs(self.plugins_path)
            logger.info(f'model_file_sync mkdir {self.plugins_path} success', g_not_server_label)
            
            self.models_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/models'
            if not os.path.exists(self.models_path):
                os.makedirs(self.models_path)
            logger.info(f'model_file_sync mkdir {self.models_path} success', g_not_server_label)

        # convert_models, 作为actor和learner都需要创建的
        self.convert_model_dir = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/convert_models_{CONFIG.svr_name}'
        if not os.path.exists(self.convert_model_dir):
            os.makedirs(self.convert_model_dir)
        logger.info(f'model_file_sync mkdir {self.convert_model_dir} success', g_not_server_label)
    
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('model_file_sync ModelFileSync stop success', g_not_server_label)

    def befor_run(self):

        # 日志
        self.logger = KaiwuLogger()
        pid = os.getpid()

        # 由于actor和learner都需要有ModelFileSync句柄
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/model_file_sync_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'model_file_sync')
        self.logger.info(f"{CONFIG.svr_name} model_file_sync, use {CONFIG.ckpt_sync_way}, use_which_deep_learning_framework is {CONFIG.use_which_deep_learning_framework}", g_not_server_label)

        self.logger.info(f'model_file_sync process pid is {pid}', g_not_server_label)

        self.make_model_dirs(self.logger)

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        self.process_run_count = 0

        t = time.time()
        self.last_run_schedule_time_by_push_pull = t
        self.last_run_schedule_time_by_sync_stat = t

        # 注册定时器任务
        # set_schedule_event(CONFIG.model_file_sync_per_minutes, self.push_and_pull_model_file)
        # set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.model_file_sync_stat)

        t = time.time()
        self.last_run_schedule_time_by_push_pull = t
        self.last_run_schedule_time_by_sync_stat = t

    def model_file_sync_stat(self):
        if int(CONFIG.use_prometheus):
            monitor_data = {}
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
                monitor_data[KaiwuDRLDefine.PULL_FROM_MODEL_POOL_SUCC_CNT] = self.pull_from_model_pool_succ_cnt
                monitor_data[KaiwuDRLDefine.PULL_FROM_MODEL_POOL_ERR_CNT] = self.pull_from_model_pool_err_cnt
            else:
                monitor_data[KaiwuDRLDefine.PUSH_TO_MODEL_POOL_SUCC_CNT] = self.push_to_model_pool_succ_cnt
                monitor_data[KaiwuDRLDefine.PUSH_TO_MODEL_POOL_ERR_CNT] = self.push_to_model_pool_err_cnt
    
            self.monitor_proxy.put_data(monitor_data)

            # 由于是一直朝上增长的指标, 不需要指标复原, 能看见代码在正常运行, 可以根据周期间隔计算出时间段内的执行次数

    def push_and_pull_model_file(self):
        
        '''
        下面是actor/learner与modolpool交互情况:
        1. 如果是tensorflow, learner推送checkpoint文件, actor拉取checkpoint文件
        2. 如果是tensorrt, learner推送wight文件, actor拉取wight文件
        '''
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
                self.push_checkpoint_to_model_pool(self.logger)
            elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORRT:
                self.push_wight_to_model_pool()
            else:
                self.logger.error(f'model_file_sync unsupport use_which_deep_learning_framework: {CONFIG.use_which_deep_learning_framework}', g_not_server_label)
                return

            if int(CONFIG.self_play):
                if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
                    self.push_old_checkpoint_to_model_pool()
                elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORRT:
                    self.push_old_wight_to_model_pool()
                else:
                    self.logger.error(f'model_file_sync unsupport use_which_deep_learning_framework: {CONFIG.use_which_deep_learning_framework}', g_not_server_label)
                    return

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            if CONFIG.self_play_actor:
                if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
                    self.pull_old_checkpoint_from_model_pool()
                elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORRT:
                    self.pull_old_wight_from_model_pool()
                else:
                    self.logger.error(f'model_file_sync unsupport use_which_deep_learning_framework: {CONFIG.use_which_deep_learning_framework}', g_not_server_label)
                    return
            else:
                if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:
                    self.pull_checkpoint_from_model_pool(self.logger)
                elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORRT:
                    self.pull_wight_from_model_pool()
                else:
                    self.logger.error(f'model_file_sync unsupport use_which_deep_learning_framework: {CONFIG.use_which_deep_learning_framework}', g_not_server_label)
                    return
    
        else:
            self.logger.error(f'model_file_sync unsupport svr_name: {CONFIG.svr_name}', g_not_server_label)
            return

    '''
    流程如下:
    1. 根据形如/data/ckpt/hero_ppo/checkpoint找到最新的step生成的checkpoint文件, 形如:
    all_model_checkpoint_paths: "/data/ckpt//hero_ppo/model.ckpt-3247"
    2. 在/data/ckpt/hero_ppo/ | grep 3247找出满足需求的model-3247.data-00000-of-00001, model-3247.index, model-3247.meta, checkpoint
    3. 对2制作成tar文件, 生成tar文件路径
    '''
    def make_model_tar_file(self):
        
        # 获取到Model文件路径所在的路径
        model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'

        checkpoint_file = f'{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        last_line = None
        try:
            _, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        if not last_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_line):
            return None

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
        if not checkpoint_id:
            return None
        checkpoint_id = int(checkpoint_id.group())
        if checkpoint_id < 0:
            return None

        target_dir = tempfile.mkdtemp()
   
        # 寻找包含checkpoint_id的meta, data, index
        for root, dirs, file_list in os.walk(model_path):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in MODEL_PATH_MANGER.exclude_directories()]
            for file_name in file_list:
                if f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}' in file_name:
                    shutil.copy(os.path.join(root, file_name), target_dir)
        
        # 需要增加checkpoint文件
        shutil.copy(checkpoint_file, target_dir)
        
        # 放在/tmp目录下生成tar文件
        output_file_name = f'{model_path}/{KaiwuDRLDefine.KAIWU_CHECK_POINT_FILE}_{CONFIG.app}_{CONFIG.algo}_{checkpoint_id}.tar.gz'
        make_tar_file(output_file_name, target_dir)

        # 删除/tmp的临时文件
        remove_tree(target_dir)

        return output_file_name

    def make_old_model_tar_file(self):
        
        # 获取到Model文件路径所在的路径
        model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'

        checkpoint_file = f'{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        '''
        获取当前随机比较旧的一个model文件上传
        '''
        last_two_line = None
        try:
            last_two_line = get_random_line_from_file(checkpoint_file)
            #last_two_line = get_last_two_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        # 如果文件不存在提前返回
        if not last_two_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_two_line):
            return None
        
        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_two_line)
        if not checkpoint_id:
            return None
        checkpoint_id = int(checkpoint_id.group())
        if checkpoint_id < 0:
            return None

        target_dir = tempfile.mkdtemp()
   
        # 寻找包含checkpoint_id的meta, data, index
        for root, dirs, file_list in os.walk(model_path):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in MODEL_PATH_MANGER.exclude_directories()]
            for file_name in file_list:
                if f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}' in file_name:
                    shutil.copy(os.path.join(root, file_name), target_dir)
        
        # 需要增加checkpoint文件
        
        shutil.copy(checkpoint_file, target_dir)
        new_ckpt_file = f'{target_dir}/{KaiwuDRLDefine.CHECK_POINT_FILE}'
        fix_checkpoint_file(new_ckpt_file, checkpoint_id)
                    
        # 放在/tmp目录下生成tar文件
        output_file_name = f'{model_path}/{KaiwuDRLDefine.KAIWU_CHECK_POINT_FILE}_{CONFIG.app}_{CONFIG.algo}_{checkpoint_id}.tar.gz'
        
        make_tar_file(output_file_name, target_dir)

        # 删除/tmp的临时文件
        remove_tree(target_dir)

        return output_file_name
    
    '''
    获取checkooint里最新的文件, 形如model.ckpt-0.meta,model.ckpt-0.index,model.ckpt-0.data
    '''
    def get_newest_checkpoint_file_id(self):
        # 获取到Model文件路径所在的路径
        model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'

        checkpoint_file = f'{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'
        
        last_line = None
        try:
            _, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        if not last_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_line):
            return -1

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
        if not checkpoint_id:
            return -1
        checkpoint_id = int(checkpoint_id.group())

        return checkpoint_id
    
    '''
    获取checkpoint里的次新文件(或者按照条件的任何文件), 形如model.ckpt-0.meta,model.ckpt-0.index,model.ckpt-0.data
    '''
    def get_second_last_checkpoint_file_id(self):
        # 获取到Model文件路径所在的路径
        model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'

        checkpoint_file = f'{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        '''
        获取当前随机比较旧的一个model文件上传
        '''
        last_two_line = None
        try:
            # last_two_line = get_last_two_line_from_file(checkpoint_file)
            last_two_line = get_random_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        # 如果文件不存在提前返回
        if not last_two_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_two_line):
            return -1
        
        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_two_line)
        if not checkpoint_id:
            return -1
        checkpoint_id = int(checkpoint_id.group())

        return checkpoint_id
    
    # push旧的模型wight文件到model_pool
    def push_old_wight_to_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            '''
            learner在启动时就会产生checkpoint文件, 即可以走dump_weights逻辑
            '''
                
            # 当前最新的checkpoint文件
            second_last_checkpoint_id = self.get_second_last_checkpoint_file_id()

            # 文件不存在则提前返回
            if not second_last_checkpoint_id:
                return False

            ckpt_prefix = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{second_last_checkpoint_id}'
            output_filename = f'{self.convert_model_dir}/trt_weights.wts2_old'
            shareFCWeights = True

            # 生成wight文件, 耗时较多, 采用日志打印
            with TimeIt() as ti:
                dump_weights(ckpt_prefix, output_filename, shareFCWeights)
            self.logger.info(f'model_file_sync success dump wights  cost {ti.interval} s, ckpt_prefix {ckpt_prefix}, shareFCWeights {shareFCWeights}', g_not_server_label)

            wigth_file = output_filename

            # push 到modelpool
            with open(wigth_file, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(model=model, hyperparam=None, key=f'{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}_old',\
                        md5sum=local_md5, save_file_name=wigth_file.split('/')[-1])

            self.push_to_model_pool_succ_cnt += 1

            self.logger.info(f'model_file_sync push to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}', g_not_server_label)
            return True
        
        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync push wight to modelpool error, \
                 as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False


    '''
    流程如下:
    1. 根据当前learner的checkpoint文件生产wight文件
    2. 采用modelpool_api发送给modelpool
    '''
    def push_wight_to_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            '''
            learner在启动时就会产生checkpoint文件, 即可以走dump_weights逻辑
            '''
    
            # 当前最新的checkpoint文件
            newest_checkpoint_id = self.get_newest_checkpoint_file_id()
            if int(newest_checkpoint_id) < 0:
                return False

            ckpt_prefix = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{newest_checkpoint_id}'
            output_filename = f'{self.convert_model_dir}/trt_weights.wts2'
            shareFCWeights = True

            # 生成wight文件, 耗时较多, 采用日志打印
            with TimeIt() as ti:
                dump_weights(ckpt_prefix, output_filename, shareFCWeights)
            self.logger.info(f'model_file_sync success dump wights  cost {ti.interval} s, ckpt_prefix {ckpt_prefix}, shareFCWeights {shareFCWeights}', g_not_server_label)

            wigth_file = output_filename

            # push 到modelpool
            with open(wigth_file, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(model=model, hyperparam=None, key=f'{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}',\
                        md5sum=local_md5, save_file_name=wigth_file.split('/')[-1])

            self.push_to_model_pool_succ_cnt += 1

            self.logger.info(f'model_file_sync push to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}', g_not_server_label)
            return True
        
        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync push wight to modelpool error, \
                 as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False

    '''
    流程如下:
    1. make_model_tar_file 制作生成的tar文件
    2. 采用modelpool_api发送给modelpool

    因为该函数可能在on-policy下调用, 故日志句柄由外界传入, 在model_file_sync进程内则直接使用logger
    '''
    def push_checkpoint_to_model_pool(self, logger=None):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            # 生成checkpoint的tar文件
            output_file_name = self.make_model_tar_file()
            if not output_file_name:
                logger.error(f'output_file_name is None', g_not_server_label)
                return False

            # push 到modelpool
            with open(output_file_name, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(model=model, hyperparam=None, key=f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}',\
                        md5sum=local_md5, save_file_name=output_file_name.split('/')[-1])
            
            self.push_to_model_pool_succ_cnt += 1

            # 删除output_file_name
            os.remove(output_file_name)
            
            logger.info(f'model_file_sync push {output_file_name} to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}', g_not_server_label)

            return True

        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            logger.error(f'model_file_sync push checkpoint to modelpool error, \
                 as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False
    
    # push旧的模型checkpoint到model_pool
    def push_old_checkpoint_to_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            # 生成checkpoint的tar文件
            output_file_name = self.make_old_model_tar_file()

            # 如果文件不存在则提前返回
            if not output_file_name:
                return False

            # push 到modelpool
            with open(output_file_name, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(model=model, hyperparam=None, key=f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}_old',\
                        md5sum=local_md5, save_file_name=output_file_name.split('/')[-1])
            
            self.push_to_model_pool_succ_cnt += 1

            # 删除output_file_name
            os.remove(output_file_name)
            
            self.logger.info(f'model_file_sync push {output_file_name} to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}', g_not_server_label)
            return True

        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync push {output_file_name} local_md5 {local_md5} to modelpool error, \
                 as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False
    
    
    '''
    流程如下:
    1. 读取当前的checkpoint文件内容
    2. 修改
    3. 写回当前文件
    '''
    def rename_checkpoint_file(self, checkpoint_path):
        if not checkpoint_path:
            return
        
        file_data = ''
        to_insert_str = 'models/'
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                line  = insert_any_string(line, to_insert_str, KaiwuDRLDefine.KAIWU_MODEL_CKPT, 'before')
                file_data += line

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
                f.write(file_data)
    
    '''
    流程如下:
    1. 采用modelpool_api从modelpool获取wight文件, 放到形如/data/ckpt/app_algo/convert_models_actor下, 文件名带上当前时间戳
    2. 将1中的文件, 拷贝到/data/ckpt/app_algo下
    3. actor从2中加载wight文件
    4. 清空/data/ckpt/app_algo/convert_models_actor目录
    '''
    def pull_wight_from_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR:
            return False
        
        try:
            # 拉取wight文件
            model_name = f'{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}'
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False
                
                model_file_name = model_info._file_name

                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False

                # 删除convert_model_dir下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.convert_model_dir)

                model_file_tar_path = f'{self.convert_model_dir}/{model_file_name}'
                
                # 写入二进制内容到文件里
                with open(model_file_tar_path, 'wb+') as file:
                    file.write(model)
                
                # 将wight文件拷贝到/data/ckpt目录下, 并且生成文件FINSH标志该可用
                shutil.copy(model_file_tar_path, f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}')

                with open(f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.FILE_FINISH_NAME}', 'w+') as file:
                    file.write(KaiwuDRLDefine.FILE_FINISH_NAME)

                self.pull_from_model_pool_succ_cnt += 1

                self.logger.info(f'model_file_sync pull wight {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}', g_not_server_label)
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync pull wight from modelpool error, as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False
        
    '''
    流程如下:
    1. 采用modelpool_api从modelpool获取tar文件, 解压到目录ckpt_dir下的业务名_算法名的plugins
    2. 遍历ckpt_dir下的业务名_算法名的plugins, 拷贝model-ckpt的开头的文件到ckpt_dir下的业务名_算法名的models
    3. actor从ckpt_dir下的业务名_算法名的models下加载model文件
    4. 清空ckpt_dir下的业务名_算法名的plugins

    因为该函数可能在on-policy下调用, 故日志句柄由外界传入, 在model_file_sync进程内则直接使用logger
    '''
    def pull_checkpoint_from_model_pool(self, logger=None):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR:
            return False

        try:
            # 拉取checkpoint的tar文件
            model_name = f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}'
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False

                model_file_name = model_info._file_name
                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False
                
                model_file_tar_path = f'{self.plugins_path}/{model_file_name}'
                
                # 写入二进制内容到文件里
                with open(model_file_tar_path, 'wb+') as file:
                    file.write(model)
                
                # 解压缩tar文件, 放在对应目录
                tar_flie_extract(model_file_tar_path, self.plugins_path)

                # 清空models下的文件夹, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.models_path)

                # 遍历文件夹拷贝文件到models下去
                for root, dirs, file_list in os.walk(self.plugins_path):
                    for file_name in file_list:
                        # 需要修改checkpoint内容, 使其在self.modes_path下面能找到引擎文件
                        if KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            self.rename_checkpoint_file(os.path.join(root, file_name))

                            shutil.copy(os.path.join(root, file_name), self.models_path)

                        if f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}' in file_name or KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            
                            # 获取到当前从learner/actor同步过来的model文件ID版本号
                            checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), file_name)
                            if not checkpoint_id:
                                continue
                            checkpoint_id = int(checkpoint_id.group())
                            
                            shutil.copy(os.path.join(root, file_name), self.models_path)
                
                # 删除plugins下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.plugins_path)

                self.pull_from_model_pool_succ_cnt += 1

                logger.info(f'model_file_sync pull {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}', g_not_server_label)
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            logger.error(f'model_file_sync pull checkpoint from modelpool error, as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False

    # 加载旧模型wight
    def pull_old_wight_from_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR:
            return False
        
        try:
            # 拉取wight文件
            model_name = f'{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}_old'
            model_name_list = self.model_pool_apis.pull_keys()
            '''
            获取旧的模型
            1. 如果返回的model_name_list的长度为1, 即进程启动时只有1个wight文件, 则新旧采用同一个wight文件
            2. 如果返回的model_name_list的长度大于1, 则进程启动了并且开始生成多个wight文件, 则采用列表为-2,-1的wight文件
            '''
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False
                
                # 因为都是为trt_weights.wts2, 故需要设置下别名带上_old下标
                model_file_name = f'{model_info._file_name}'

                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False

                model_file_tar_path = f'{self.convert_model_dir}/{model_file_name}'
                
                # 写入二进制内容到文件里
                with open(model_file_tar_path, 'wb+') as file:
                    file.write(model)
                
                # 同时计算下文件的md5sum写入到文件里, 便于对账, 不同进程之间对文件的校验
                
                
                # 将wight文件拷贝到/data/ckpt目录下
                shutil.copy(model_file_tar_path, f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}')

                with open(f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.FILE_FINSH_OLD_NAME}', 'w+') as file:
                    file.write(KaiwuDRLDefine.FILE_FINSH_OLD_NAME)

                self.pull_from_model_pool_succ_cnt += 1

                self.logger.info(f'model_file_sync pull wight {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}', g_not_server_label)
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync pull wight from modelpool error, as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False
    
    # 加载旧模型checkpoint  
    def pull_old_checkpoint_from_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR:
            return False

        try:
            # 拉取checkpoint的tar文件
            model_name = f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}_old'
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False

                model_file_name = model_info._file_name
                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False
                
                model_file_tar_path = f'{self.plugins_path}/{model_file_name}'
                
                # 写入二进制内容到文件里
                with open(model_file_tar_path, 'wb+') as file:
                    file.write(model)
                
                # 解压缩tar文件, 放在对应目录
                tar_flie_extract(model_file_tar_path, self.plugins_path)

                # 清空models下的文件夹, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.models_path)

                # 遍历文件夹拷贝文件到models下去
                for root, dirs, file_list in os.walk(self.plugins_path):
                    for file_name in file_list:
                        # 需要修改checkpoint内容, 使其在self.modes_path下面能找到引擎文件
                        if KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            self.rename_checkpoint_file(os.path.join(root, file_name))

                            shutil.copy(os.path.join(root, file_name), self.models_path)

                        if f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}' in file_name or KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            shutil.copy(os.path.join(root, file_name), self.models_path)

                # 删除plugins下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.plugins_path)

                self.pull_from_model_pool_succ_cnt += 1

                self.logger.info(f'model_file_sync pull {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}', g_not_server_label)
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            self.logger.error(f'model_file_sync pull from modelpool error, as {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)
            return False

    def run_once(self):
        '''
         启动定时器操作, 定时器里执行具体的操作
         1. learner --> modelpool, push
         2. modelpool --> actor, pull
        '''
        # schedule.run_pending()
        
        now = time.time()
        if now - self.last_run_schedule_time_by_push_pull > int(CONFIG.model_file_sync_per_minutes) * 60:
            self.push_and_pull_model_file()
            self.last_run_schedule_time_by_push_pull = now
        elif now - self.last_run_schedule_time_by_sync_stat > int(CONFIG.prometheus_stat_per_minutes) * 60:
            self.model_file_sync_stat()
            self.last_run_schedule_time_by_sync_stat = now

    def run(self):
        self.befor_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(f"model_file_sync failed to run {self.name} trainer. exit. Error is: {e}, traceback.print_exc() is {traceback.format_exc()}", 
                g_not_server_label)
                break
