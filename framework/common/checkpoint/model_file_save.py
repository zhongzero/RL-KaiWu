#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import datetime
import re
import shutil
import tempfile
import multiprocessing
import time
import traceback
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
import schedule
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from distutils.dir_util import copy_tree, remove_tree
from framework.common.config.config_control import CONFIG
from framework.common.utils.cos_utils import COSSave
from framework.common.checkpoint.model_path_manager import MODEL_PATH_MANGER
from framework.common.utils.common_func import get_first_last_line_from_file, get_sort_file_list, make_tar_file, set_schedule_event, tar_flie_extract, make_single_dir, write_json_to_file
from framework.common.monitor.monitor_proxy import MonitorProxy

'''
单个Model文件较大, 传递到COS耗时较多, 故采用单个进程处理
'''
class ModelFileSave(multiprocessing.Process):
    def __init__(self, ) -> None:
        super(ModelFileSave, self).__init__()

        self.exit_flag = multiprocessing.Value('b', False)

        # 只有主learner才执行上传model file任务
        self.is_checf = False

        self.local_and_remote_dirs = MODEL_PATH_MANGER.get_local_and_remote_dirs()

        # 记录进程启动时间
        self.process_start_time = time.monotonic()

        # 统计值
        self.push_to_cos_succ_cnt = 0
        self.push_to_cos_err_cnt = 0

        # 上次保存旁路的时间
        self.last_bypass_time = 0

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('model_file_save ModelFileSave stop success', g_not_server_label)
    
    '''
    流程如下:
    1. 根据形如/data/ckpt/hero_ppo/checkpoint找到最新的step生成的checkpoint文件, 形如:
    all_model_checkpoint_paths: "/data/ckpt//hero_ppo/model.ckpt-3247"
    2. 在/data/ckpt/hero_ppo/ | grep 3247找出满足需求的model-3247.data-00000-of-00001, model-3247.index, model-3247.meta, checkpoint
    3. 对2制作成tar文件, 生成tar文件路径
    '''
    def make_model_tar_file(self, key, local_and_remote_dir):
        
        if key == KaiwuDRLDefine.CKPT_DIR:
            # 放在cos_local_target_dir目录下生成tar文件
            time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
            output_file_name = f'{CONFIG.cos_local_target_dir}/{CONFIG.app}_{time_str}.{KaiwuDRLDefine.TAR_GZ}'
            make_tar_file(output_file_name, local_and_remote_dir)

            write_json_to_file({
                "create_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                "train_time": int(time.monotonic() - self.process_start_time)
                }, time_str, CONFIG.cos_local_target_dir)
            
            # 删除/tmp的临时文件
            remove_tree(local_and_remote_dir)

            return output_file_name
        
        elif key == KaiwuDRLDefine.RESTOR_DIR:
            pass

        elif key == KaiwuDRLDefine.SUMMARY_DIR:
            pass

        elif key == KaiwuDRLDefine.PB_MODEL_DIR:
            pass

        else:
            pass
    
    '''
    按照需要清空文件夹下的文件
    保留最近N个文件, 规避磁盘空间占用, 原则如下:
    1. 如果是KaiwuDRL负责发送到COS文件, 则每次发送完成后会删除本地文件
    2. 如果是KaiwuDRL负责生成COS文件, 但是不会发送, 则每次开始前需要确保只有最近N个文件存在
    '''
    def clearn_dir(self):
        if int(CONFIG.push_to_cos):
            return
        
        dir_list = get_sort_file_list(CONFIG.cos_local_target_dir, True)
        
        # 如果目前长度小于需要保留的长度则返回
        if len(dir_list) < CONFIG.cos_local_keep_file_num:
            return
        
        # 对于大于保留长度的文件进行删除操作
        for file in dir_list[CONFIG.cos_local_keep_file_num:]:
            file_path = os.path.join(CONFIG.cos_local_target_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    '''
    操作步骤如下:
    1. 将需要上传COS的文件目录拷贝到tmp目录下
    2. 对该目录压缩成tar.gz包
    3. 将tar.gz包上传到COS
    4. 将tmp目录下的文件目录删除
    '''
    def push_file_to_cos(self, temp_remote_dirs):
        # 清空文件夹内容
        self.clearn_dir()

        if not temp_remote_dirs:
            return

        for key, local_and_remote_dir in temp_remote_dirs.items():
            if MODEL_PATH_MANGER.need_to_sync(key):
                self.logger.info(f'local_and_remote_dir {key}/{local_and_remote_dir} need to sync to COS', g_not_server_label)

                try:
                    
                    # 注意约定的格式, 在拉取COS文件时需要设置下
                    output_file_name = self.make_model_tar_file(key, local_and_remote_dir)
                    if not output_file_name:
                        continue

                    # 如果是需要旁路, 按照旁路时间间隔旁路一份
                    if CONFIG.use_bypass:
                        # time.time()返回的是以秒为单位
                        now = int(time.time())
                        if  now - self.last_bypass_time >= int(CONFIG.bypass_per_minutes) * 60:
                            shutil.copy(output_file_name, CONFIG.bypass_dir)
                            self.last_bypass_time = now
                    
                    '''
                    由于在集群部署环境时, COS的信息无法暴露给普通使用者, 故采用的方案:
                    1. 对于有外网使用的用户, KaiwuDRL屏蔽conf下的文件, 打包到指定文件夹, 由集群开启新的容器负责传输打包的文件到COS
                    2. 对于内部用户, KaiwuDRL暴露conf下的文件, 打包, 上传到COS

                    注意如果是KaiwuDRL上传, 会删除output_file_name文件; 否则需要外界脚本来删除掉output_file_name文件
                    '''
                    if int(CONFIG.push_to_cos):
                        cos_bucket_key = f'{KaiwuDRLDefine.COS_BUCKET_KEY}{CONFIG.app}'
                        
                        key= f'{cos_bucket_key}{output_file_name.split("/")[-1]}'

                        if self.model_file_saver.push_to_cos(output_file_name, CONFIG.cos_bucket, key):
                            self.logger.info(f'model_file_save file push to cos success, local_and_remote_dir {output_file_name}', g_not_server_label)
                            self.push_to_cos_succ_cnt += 1
                        else:
                            self.logger.error(f'model_file_save file push to cos error, local_and_remote_dir {output_file_name}', g_not_server_label)
                            self.push_to_cos_err_cnt += 1
                    
                        # 删除output_file_name
                        os.remove(output_file_name)

                except Exception as e:
                    self.logger.error(f'model_file_save push to cos error, as error is {str(e)}, traceback.print_exc() is {traceback.format_exc()}', 
                    g_not_server_label)
    
    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/model_file_save_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'model_file_save')

        # COS保存句柄, 从conf/configure下获取COS的配置
        if int(CONFIG.push_to_cos):
            self.model_file_saver = COSSave(self.logger, CONFIG.cos_secret_id, CONFIG.cos_secret_key, CONFIG.cos_region, CONFIG.cos_token)
            self.model_file_saver.connect_to_cos()

        # 注册定时器任务
        # set_schedule_event(CONFIG.model_file_save_per_minutes, self.save_model_file_to_cos)
        # set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.model_file_save_stat)

        t = time.time()
        self.last_run_schedule_time_by_to_cos = t
        self.last_run_schedule_time_by_save_stat = t

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        self.process_run_count = 0

        # 建立必要的文件目录
        make_single_dir(CONFIG.cos_local_target_dir)

        self.logger.info('model_file_save process pid is {}', os.getpid(), g_not_server_label)
    
    def model_file_save_stat(self):
        if CONFIG.use_prometheus:
            monitor_data = {
                KaiwuDRLDefine.PUSH_TO_COS_SUCC_CNT : self.push_to_cos_succ_cnt,
                KaiwuDRLDefine.PUSH_TO_COS_ERR_CNT : self.push_to_cos_err_cnt
            }

            self.monitor_proxy.put_data(monitor_data)

            # 由于是一直朝上增长的指标, 不需要指标复原, 能看见代码在正常运行, 可以根据周期间隔计算出时间段内的执行次数

    '''
    learner --> COS的model文件同步, 每隔多少分钟执行
    '''
    def save_model_file_to_cos(self):
        
        # 拷贝到临时目录
        temp_remote_dirs = self.copy_to_temp_dir()

        # 上传Model文件到COS
        self.push_file_to_cos(temp_remote_dirs)

        # 删除临时目录
        self.remove_temp_dir(temp_remote_dirs)

        self.logger.info(f'model_file_save train model file save to cos success, succ cnt {self.push_to_cos_succ_cnt}, err cnt is {self.push_to_cos_err_cnt}',
                         g_not_server_label)

    def run_once(self):

        # 启动定时器操作, 定时器里执行具体的保存操作, 但是
        # schedule.run_pending()

        now = time.time()
        if now - self.last_run_schedule_time_by_to_cos > int(CONFIG.model_file_save_per_minutes) * 60:
            self.save_model_file_to_cos()
            self.last_run_schedule_time_by_to_cos = now
        elif now - self.last_run_schedule_time_by_save_stat > int(CONFIG.prometheus_stat_per_minutes) * 60:
            self.model_file_save_stat()
            self.last_run_schedule_time_by_save_stat = now
        else:
            pass
    
    def run(self) -> None:
        self.before_run()

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
                self.logger.error(f'model_file_save run_once, err: {str(e)}, traceback.print_exc() is {traceback.format_exc()}',
                 g_not_server_label)

    def remove_temp_dir(self, temp_remote_dirs):
        # 删除/tmp下临时目录
        for temp_dir in temp_remote_dirs.values():
            if os.path.exists(temp_dir):
                remove_tree(temp_dir)
    
    '''
    按照类型来做极小拷贝:
    KaiwuDRLDefine.CKPT_DIR : f'{self.ckpt_dir}/{CONFIG.app}_{CONFIG.algo}',
    KaiwuDRLDefine.RESTOR_DIR : self.restore_dir,
    KaiwuDRLDefine.SUMMARY_DIR : f'{self.summary_dir}/{CONFIG.app}_{CONFIG.algo}',
    KaiwuDRLDefine.PB_MODEL_DIR : self.pb_model_dir
    '''
    def copy_need_model_file(self, key, target_dir):
        if not target_dir or not key:
            return
        
        if key == KaiwuDRLDefine.CKPT_DIR:
            # 获取到Model文件路径所在的路径
            model_path = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'

            checkpoint_file = f'{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

            last_line = None
            try:
                _, last_line = get_first_last_line_from_file(checkpoint_file)
            except Exception as e:
                pass
            if not last_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_line):
                return

            # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
            checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
            if not checkpoint_id:
                return -1
            checkpoint_id = int(checkpoint_id.group())
    
            # 寻找包含checkpoint_id的meta, data, index
            for root, dirs, file_list in os.walk(model_path):
                # 排除指定目录
                dirs[:] = [d for d in dirs if d not in MODEL_PATH_MANGER.exclude_directories()]
                for file_name in file_list:
                    if f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}' in file_name:
                        shutil.copy(os.path.join(root, file_name), target_dir)
            
            # 需要增加checkpoint文件
            shutil.copy(checkpoint_file, target_dir)

        elif key == KaiwuDRLDefine.RESTOR_DIR:
            pass

        elif key == KaiwuDRLDefine.SUMMARY_DIR:
            pass

        elif key == KaiwuDRLDefine.PB_MODEL_DIR:
            pass

        else:
            pass
    
    def copy_to_temp_dir(self):
        # 格式形如: ckpt_dir --> /tmp/xxx
        temp_remote_dirs = {}
        # 生成/tmp下临时目录, 需要回滚时操作
        target_dir = tempfile.mkdtemp()
        
        try:
            for key, local_and_remote_dir in self.local_and_remote_dirs.items():

                if not os.path.exists(local_and_remote_dir):
                    continue
    
                # 可能会异常, 此时会跳转到Exception处理

                '''
                因为每次拷贝的数据文件比较多, 故这里是先过滤, 再进行拷贝, 减少数据文件拷贝耗时
                '''
                self.copy_need_model_file(key, target_dir)

                #copy_tree(local_and_remote_dir, target_dir)
                temp_remote_dirs[key] =  target_dir

        except Exception as e:
            self.logger.error(f'model_file_save Error copying local folders, err: {str(e)}, traceback.print_exc() is {traceback.format_exc()}',
            g_not_server_label)

            # 拷贝失败时，本次停止上传COS, 删除目录
            for temp_dir in temp_remote_dirs.values():
                 remove_tree(temp_dir)
            temp_remote_dirs.clear()

            # 本次异常生成的/tmp临时目录也需要删除
            remove_tree(target_dir)
        
        return temp_remote_dirs

    '''
    根据不同的启动方式进行处理:
    1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
    2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动

    因为该函数是被actor和learner调用, 故需要定义下COSSave, 注意不要和model_file_save的混淆了(进程里调用)
    '''
    def start_actor_process_by_type(self, logger):
        if not CONFIG.start_actor_learner_process_type:
            logger.info(f'predict process start, type is {CONFIG.start_actor_learner_process_type}, no need get mode file from cos')
            return
        
        # COS保存句柄, 从conf/configure下获取COS的配置
        cos_saver = COSSave(logger, CONFIG.cos_secret_id, CONFIG.cos_secret_key, CONFIG.cos_region, CONFIG.cos_token)
        cos_saver.connect_to_cos()

        # 获取当天上传最近的一次COS文件列表, 即最大可能恢复
        cos_bucket_key = f'{KaiwuDRLDefine.COS_BUCKET_KEY}{CONFIG.app}'

        file_list= cos_saver.query_object_list(CONFIG.cos_bucket, cos_bucket_key)
        if not file_list:
            logger.error(f'get cos object list is None')
            return
        
        file_list_content = file_list.get('Contents', None)
        if not file_list_content:
            logger.error(f'get cos object list content is None')
            return

        # 按照list[-1]去获取最新的文件
        key = file_list_content[-1].get('Key', None)
        if not key:
            logger.error(f'get cos object list content last Key is None')
            return
        
        destination_file_name = f'{CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.COS_LAST_MODEL_FILE}'

        if cos_saver.get_from_cos(CONFIG.cos_bucket, key, destination_file_name):
            logger.info(f'get cos last Key success')
        else:
            logger.error(f'get cos last Key error')
            return

        # 解压tar文件到ckpt目录
        tar_flie_extract(destination_file_name, f'{CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}')

        # 删除临时的文件
        os.remove(destination_file_name)

        logger.info(f'{CONFIG.svr_name} get cos file is succ, last file is {destination_file_name}, destination dir is {CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}')
