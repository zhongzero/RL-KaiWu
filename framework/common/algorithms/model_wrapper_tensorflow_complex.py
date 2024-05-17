#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# rerf: https://github.com/horovod/horovod, need pip install horovod, gcc 7.3.1
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
import horovod.tensorflow as hvd

from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG
from framework.common.config.cluster_conf import ClusterConf
from framework.common.algorithms.model import ModeKeys
from framework.common.utils.common_func import get_local_rank, get_local_size

'''
ModelWrapperTensorflowComplex类, actor和learner都会使用, 主要用于预测, 训练等
'''
class ModelWrapperTensorflowComplex:

    def __init__(self, model, logger, server = None) -> None:
        
        self.model = model
        self.logger = logger
        self.chief_only_hooks = []
        self.train_hooks = []
        self.predict_hooks = []

        mpi_local_rank = get_local_rank()

        # 加载TensorFlow需要的配置文件
        self.cluster_conf = ClusterConf(
        CONFIG.learner_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.actor_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.learner_grpc_ports.split(','),
        CONFIG.actor_grpc_ports.split(','),
        # 下面配置项目每个进程启动时, 进程配置文件加载
        CONFIG.svr_name, 
        CONFIG.svr_index, 
        CONFIG.svr_ports,
        mpi_local_rank
        )

        '''
        根据加载的配置文件里进程不同名字来处理, 主要是actor和learner
        '''
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            hvd.init()
            self.local_rank = hvd.local_rank()

        # 主learner
        self.is_chief = (CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER and hvd.rank() == 0)

        # learner_device
        self.learner_device = '/job:learner/task:0'

        # actor_device
        self.actor_device = '/job:actor/task:%d' % self.cluster_conf.task_index

        # config
        cpu_num = int(CONFIG.cpu_num / get_local_size()) 
        self.config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=cpu_num,
            inter_op_parallelism_threads=cpu_num)

        self.sess = None
        self.local_step = -1
        self.global_step = -1

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            self.config.gpu_options.allow_growth = True
            self.cached_local_step = -1

        '''
        目前KaiwuDRL通信情况:
        1. 采用的horovod进行learner间的通信
        2. 采用modelpool进行learner/actor之间的模型同步
        '''

        '''
        if not server:
            self.server = tf.distribute.Server(
                    self.cluster_conf.cluster_spec(), 
                    job_name=self.cluster_conf.job_name, 
                    task_index=self.cluster_conf.task_index, 
                    config=self.config)
        else:
            self.server = server
        '''
        
        # saver
        self.saver = None

    def build_train_graph(self, input_fn, *args, **kwargs):
        with tf.device(f"{self.learner_device}/cpu:0"):
            input_tensors = input_fn(*args, **kwargs)
        
        with tf.device(f"{self.learner_device}/{CONFIG.learner_device_type}"):
            self.global_step = tf.compat.v1.train.get_or_create_global_step()
            with tf.compat.v1.variable_scope("%s/global" % self.model.name):
                self.global_spec = self.model.build_model(ModeKeys.TRAIN, input_tensors)
            if CONFIG.enable_mixed_precision:
                self.opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                    self.global_spec.optimizer)
            else:
                self.opt = self.global_spec.optimizer
            
            self.opt = hvd.DistributedOptimizer(self.opt, sparse_as_dense=True)
            params = tf.compat.v1.trainable_variables("%s/global/network" % self.model.name)
            grads_and_var = self.compute_gradients(params)
            grads, var = zip(*grads_and_var)
            if CONFIG.max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, CONFIG.max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self.train_op = self.opt.apply_gradients(grads_and_var, global_step=self.global_step)
            with tf.compat.v1.variable_scope("hvd_broadcast"):
                self.bcast_op = hvd.broadcast_global_variables(0)
    
    # 计算梯度
    def compute_gradients(self, params):
        return self.opt.compute_gradients(self.global_spec.loss, params)

    def build_predict_graph(self, input_fn, *args, **kwargs):
        with tf.device(f"{self.actor_device}/cpu:0"):
            input_tensors = input_fn(*args, **kwargs)

        # 需要按照learner实际部署的是在GPU还是CPU机器上, 如果learner是部署在CPU机器上, 则learner_device_rank = 0; 如果是部署在GPU机器上, 则是learner_local_rank
        learner_device_rank = 0 if CONFIG.learner_device_type.lower() == 'CPU' else self.cluster_conf.learner_local_rank
        
        with tf.device(f'{self.learner_device}/{CONFIG.learner_device_type}:{learner_device_rank}'):
            self.global_step = tf.compat.v1.train.get_or_create_global_step()
            with tf.compat.v1.variable_scope("%s/global" % self.model.name):
                self.global_spec = self.model.build_model(ModeKeys.PREDICT, input_tensors)
        
        params = tf.global_variables()

        # 当图生成后, tf.train.Saver做restore操作
        self.saver = tf.train.Saver(params)

    '''
    create_train_session
    '''
    def create_train_session(self):

        self.sess = tf.compat.v1.train.MonitoredTrainingSession(
            master=self.server.target,
            checkpoint_dir=f'{CONFIG.restore_dir}/{self.model.name}/' if self.is_chief else None,
            summary_dir=f'{CONFIG.summary_dir}/{self.model.name}/' if self.is_chief else None,
            save_checkpoint_secs=CONFIG.save_checkpoint_secs,
            save_checkpoint_steps=None,
            save_summaries_steps=CONFIG.save_summaries_steps,
            chief_only_hooks=self.chief_only_hooks,
            hooks=self.train_hooks,
            is_chief=True,
            config=self.config
        )

        self.sess._tf_sess().run(self.bcast_op)

        self.train_count = 0

        # 打印网络值和权重
        # print_variables(self.sess, KaiwuDRLDefine.SERVER_LEARNER)

        # 如果需要打开TensorFlow Profile
        if CONFIG.print_profile:
             self.profiler = model_analyzer.Profiler(self.sess.graph)

    '''
    create_predict_session
    '''
    def create_predict_session(self):
        '''
        这里强制设置is_chief为True, 导致actor和learner区分为独立的网络了

        后期需要优化成同一个网络, 但是需要actor和learner之间的参数同步
        '''
        self.sess = tf.compat.v1.train.MonitoredTrainingSession(
            master=self.server.target,
            checkpoint_dir=None,
            summary_dir=None,
            save_checkpoint_secs=CONFIG.save_checkpoint_secs,
            save_checkpoint_steps=None,
            save_summaries_steps=CONFIG.save_summaries_steps,
            chief_only_hooks=None,
            hooks=self.predict_hooks,
            is_chief=True,
            config=self.config
        ) 

        self.predict_count = 0

        # 打印网络值和权重
        print_variables(self.sess, KaiwuDRLDefine.SERVER_ACTOR)

        # 如果需要打开TensorFlow Profile
        if CONFIG.print_profile:
             self.profiler = model_analyzer.Profiler(self.sess.graph)

    def should_stop(self):
        return self.sess.should_stop()

    def close(self):
        return self.sess.close()
    
    def before_train(self):
        pass

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False

        self.train_count += 1

        return has_model_file_changed, -1

    def before_predict(self):
        pass

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    '''
    profile 前的根据step来确定run_options和run_metadata
    '''
    def profile_before_run(self, step):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        if step in range(CONFIG.print_profile_start_step, CONFIG.print_profile_end_step):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            return run_options, run_metadata
        
        return run_options, run_metadata
    
    '''
    profile 后的操作
    '''
    def proflie_after_run(self, step, run_options, run_metadata, trim_name_regexes=None):
        if step in range(CONFIG.print_profile_start_step, CONFIG.print_profile_end_step):
            self.profiler.add_step(step, run_metadata)

        if step >= CONFIG.print_profile_end_step:
            builder = option_builder.ProfileOptionBuilder()
            builder.with_timeline_output(timeline_file=f'{CONFIG.log_dir}/timeline_{get_local_rank()}.json')
            self.profiler.profile_graph(builder.build())

            builder = option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
            if trim_name_regexes:
                builder.with_node_names(trim_name_regexes=trim_name_regexes)
            builder.order_by('micros')
            builder.with_file_output(outfile=f'{CONFIG.log_dir}/time_and_memory_{get_local_rank()}.txt')
            self.profiler.profile_name_scope(builder.build())

            # 本次执行完成后, 设置print_profile为False
            CONFIG.print_profile = False

    '''
    train 函数
    '''
    def train(self, extra_tensors=None):
        self.before_train()

        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        if CONFIG.print_profile:
            step = int(self.get_global_step())
            run_options, run_metadata = self.profile_before_run(step)

        fetches = {'train_op': self.train_op}

        if extra_tensors:
            fetches.update(extra_tensors)
        values = self.sess.run(fetches,
                               options=run_options,
                               run_metadata=run_metadata)
        values.pop('train_op')

        has_model_file_changed, model_file_id = self.after_train()

        if CONFIG.print_profile:
            self.proflie_after_run(step, run_options, run_metadata, trim_name_regexes=['gradients'])

        return values, has_model_file_changed, model_file_id

    '''
    predict函数
    '''
    def predict(self, extra_tensors, batch_size):
        self.before_predict()

        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        if CONFIG.print_profile:
            run_options, run_metadata = self.profile_before_run(self.predict_count)

        fetches = {}
        fetches.update(self.global_spec.predict_output)
        if extra_tensors:
            fetches.update(extra_tensors)
        
        values = self.sess.run(fetches, options=run_options, run_metadata=run_metadata)
        if CONFIG.print_profile:
            self.proflie_after_run(self.predict_count, run_options, run_metadata)

        self.after_predict(batch_size)

        return values

    def add_chief_only_hooks(self, hooks):
        if hvd.rank() == 0 and hooks:
             self.chief_only_hooks.extend(hooks)
    
    def add_train_hooks(self, hooks):
        if hooks:
            self.train_hooks.extend(hooks)
    
    def add_predict_hooks(self, hooks):
        if hooks:
            self.predict_hooks.extend(hooks)
    
    def get_global_step(self):
        return self.sess._tf_sess().run(self.global_step)

    @property
    def tf_sess(self):
        return self.sess._tf_sess()
        
    @property
    def train_stat(self):
        return self.train_count
    
    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return 'ModelWrapperTensorflowComplex'
    
    def is_chief(self):
        return self.is_chief
    
    '''
    加载最新的model文件
    '''
    def load_last_new_model(self, models_path):
        with tf.Graph().as_default():
            ckpt = tf.train.get_checkpoint_state(models_path)
            if ckpt:
                # 加载最新的模型
                self.saver.restore(self.sess, ckpt.all_model_checkpoint_paths[-1])

    # 预加载模型文件, 对于tensorflow来说, 可以直接放在引擎文件目录下即可
    def preload_model_file(self, preload_model_file, id):
        pass