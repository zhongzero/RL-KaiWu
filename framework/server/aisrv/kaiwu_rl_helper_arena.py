#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
@Project: kaiwu-fwk
@File    :kaiwu_rl_helper_arena.py
@Author  :kaiwuDRL
@Date    :2022/9/22 9:10

'''

import os
import threading
import time
import traceback
import datetime
import numpy as np
from framework.common.utils.common_func import Context, TimeIt
from framework.server.aisrv.kaiwu_environ import KaiwuEnviron
from framework.interface.agent_context import AgentContext
from framework.common.config.config_control import CONFIG
from framework.common.config.app_conf import AppConf
from framework.common.config.algo_conf import AlgoConf
from framework.interface.exception import SkipEpisodeException, ClientQuitException, TimeoutEpisodeException
from framework.common.logging.kaiwu_logger import KaiwuLogger
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
import arena

SAMPLE_CUT_POINT = [14, 39, 81, 123, 126, 187, 188, 189, 190, 191, 192, 193]

# 实现标准的强化学习训练流程


class KaiWuRLArenaHelper(threading.Thread):
    __slots__ = ("policies", "simu_ctx", "exit_flag", "client_address", "slot_id", "data_queue", "client_id", "episode_start_time",
                 "ep_frame_cnt", "agent_ctxs", "logger", "env", "steps", "reward_value", "use_sample_server")

    def __init__(self, parent_simu_ctx) -> None:
        super().__init__()

        self.policies = {}
        # 根据policy来设置下, 强化学习是AsyncPolicy, 形如train --> AsyncPolicy
        for policy_name, policy_builder in parent_simu_ctx.policies_builder.items():
            self.policies[policy_name] = policy_builder.build()

        # 上下文放在该变量里
        self.simu_ctx = Context(**parent_simu_ctx.__dict__)

        # 是否结束标志位
        self.exit_flag = self.simu_ctx.exit_flag
        # 客户端ID
        self.client_address = self.simu_ctx.client_address
        # policy
        self.simu_ctx.policies = self.policies
        # slot_id
        self.slot_id = self.simu_ctx.slot_id

        # 数据队列
        self.data_queue = self.simu_ctx.data_queue

        # 设置线程名字
        self.setName(f'kaiwu_rl_helper_{self.slot_id}')

        self.client_id = None

        # 下面是episode的统计指标
        self.episode_start_time = 0
        self.ep_frame_cnt = 0

        # 智能体agent的上下文agent_ctxs, 格式为{"agent_id" : agent_ctx}
        self.agent_ctxs = {}

        # 日志模块
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/aisrv_kaiwu_rl_helper_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
        self.logger.info(
            f'kaiwu_rl_helper start at pid {pid}, ppid is {threading.currentThread().ident}, thread id is {self.get_pid()}')

        # 将日志句柄作为参数传递
        self.simu_ctx.logger = self.logger

        # 启动Env, 即消息流转函数
        self.env = KaiwuEnviron(
            self.simu_ctx, self.exit_flag, self.client_address)

        # 动作执行步数，用于样本计数
        self.steps = 0

        # reward统计值
        self.reward_value = 0

        # 是否用sample_server进行样本存储
        self.use_sample_server = CONFIG.use_sample_server

        # aisrv发给actor的请求返回给处理该值的model文件版本号
        self.from_actor_model_version = -1

        # learner通知aisrv此时最新的model文件版本号
        self.from_learner_model_version = -1

        # 暂停/继续线程执行
        self.should_pause = False

        # 有多少agent_id
        self.agent_ids = []

        # 业务会在aisrv里上报自定义的监控指标, 故这里增加上, map形式, 由业务自己定义
        self.app_monitor_data = {}

    '''
    获取当前使用的actor和learner列表
    '''

    def get_current_actor_learner_address(self):

        actor_addrs, learner_addrs = None, None
        polic_build = self.policies[CONFIG.policy_name]
        if polic_build:
            actor_addrs, learner_addrs = polic_build.get_current_actor_learner_prxoy_list()

        return actor_addrs, learner_addrs

    '''
    获取当前训练的reward
    '''

    def get_current_reward_value(self):
        return self.reward_value

    '''
    修改kaiwu_rl_helper的actor和learner地址
    1. actor_add_or_reduce, 针对actor的增减
    2. actor_ips, actor_ip列表
    3. learner_add_or_reduce, 针对learner的增减
    4. learner_ips, learner_ip列表

    返回的参数:
    1. False, 即本次没有更新, 不能修改old_actor_address和old_learner_address
    2. True, 即本次更新完成, 需要修改old_learner_address和old_learner_address
    '''

    def kaiwu_rl_helper_change_actor_learner_ip(self, actor_add_or_reduce, actor_ips, learner_add_or_reduce, learner_ips):
        if actor_add_or_reduce and not actor_ips:
            return False

        if learner_add_or_reduce and not learner_ips:
            return False

        # 针对当前的policy_name进行处理
        policy = self.policies[CONFIG.policy_name]

        # 下面针对具体的actor和learner的增减进行处理
        if actor_add_or_reduce and actor_ips:
            for actor_ip in actor_ips:
                if KaiwuDRLDefine.PROCESS_ADD == actor_add_or_reduce:
                    policy.add_actor_proxy_list(actor_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == actor_add_or_reduce:
                    policy.reduce_actor_proxy_list(actor_ip)
                else:
                    pass

        if learner_add_or_reduce and learner_ips:
            for learner_ip in learner_ips:
                if KaiwuDRLDefine.PROCESS_ADD == learner_add_or_reduce:
                    policy.add_learner_proxy_list(learner_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == learner_add_or_reduce:
                    policy.reduce_learner_proxy_list(learner_ip)
                else:
                    pass

        # 操作完成后需要继续线程活动
        self.logger.info(
            f'kaiwu_rl_helper {actor_add_or_reduce} {actor_ips} {learner_add_or_reduce} {learner_ips} expansion success')

        return True

    '''
    返回policies
    '''

    def get_policies(self):
        return self.policies

    # 获取线程ID
    def get_pid(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

        return -1

    @property
    def identity(self):
        return f'kaiwu_rl_helper_{self.slot_id}'

        # return "(client_conn_id: %s, client_id: %s)" % (self.client_address, self.client_id or "")

    # 获取是否处于pause状态
    def get_process_in_pause_statues(self):
        return self.should_pause

    # 暂停处理pause
    def process_pause(self):
        self.should_pause = True

    # 继续处理continue
    def process_continue(self):
        self.should_pause = False

    '''
    下面是样本生产后的格式:
    {
        'x': array([], shape=(0, 4, 2), dtype=float32), 
        'a': array([], shape=(0, 4, 1), dtype=float32), 
        'old_neg_logp_a': array([], shape=(0, 4), dtype=float32), 
        'y_r': array([], shape=(0, 4), dtype=float32), 
        'old_vpred': array([], shape=(0, 4), dtype=float32), 
        'm': array([], shape=(0, 4), dtype=float32), 
        's': array([], shape=(0, 1), dtype=int64)
    }

    建表语句：
    (TensorSpec(shape=(4,), dtype=tf.int32, name='a'), 
    TensorSpec(shape=(1,), dtype=tf.float32, name='m'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='old_neg_logp_a'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='old_vpred'), 
    TensorSpec(shape=(4, 4), dtype=tf.float32, name='x'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='y_r'))
    '''

    # 目前业务sgame, 每次action
    def gen_expr(self, agent_id, policy_id, extra_info=None):
        agent_ctx = self.agent_ctxs[agent_id]
        expr_processor = agent_ctx.expr_processor[policy_id]
        
        # 由于expr_processor类是单例，因此只用调用一次即可
        expr_processor.gen_expr(
            extra_info['must_need_sample_info'], extra_info['network_sample_info'], self.from_actor_model_version, self.from_learner_model_version)
    
    '''
    before_run函数
    '''
    def before_run(self):
        # 注意传入的参数格式
        arena.setup(run_mode='proxy', skylarena_url=f'tcp://{self.simu_ctx.client_address}')

    '''
    发送给actor预测请求并且从actor获取预测响应
    '''
    def predict(self, predict_data):
        if not predict_data:
            return
        
        # 组装数据
        for agent_id in self.agent_ids:
            agent_ctx = self.agent_ctxs[agent_id]
            policy_id = agent_ctx.main_id

            agent_ctx.pred_input = {}
            agent_ctx.pred_input[policy_id] = predict_data
  
        # aisrv朝actor发送预测请求
        for agent_id in self.agent_ids:
            agent_ctx = self.agent_ctxs[agent_id]
            for policy_id in agent_ctx.policy:
                # 调用AsyncPolicy的send_pred_data函数
                success, actor_address = agent_ctx.policy[policy_id].send_pred_data(
                        self.slot_id, agent_ctx.pred_input[policy_id], agent_ctx)
                self.logger.debug(f'kaiwu_rl_helper aisrv send to actor: {agent_ctx.pred_input[policy_id]}')
                if not success:
                    self.logger.error(f"kaiwu_rl_helper policy_id {policy_id} agent_id {agent_id} send_pred_data to actor {actor_address} failed")
                    continue
        
        # aisrv从actor获取预测响应
        for agent_id in self.agent_ids:
            agent_ctx = self.agent_ctxs[agent_id]
            agent_ctx.pred_output = {}
            for policy_id in agent_ctx.policy:
                pred_output = agent_ctx.policy[policy_id].get_pred_result(
                    self.slot_id, agent_ctx)
                
                self.logger.debug(f'kaiwu_rl_helper aisrv recv from actor: {pred_output}')
                if not pred_output:
                    self.logger.error("kaiwu_rl_helper get_pred_result failed")
                else:
                    agent_ctx.pred_output[policy_id] = pred_output
    
        # 提取数据
        format_action_list = []
        lstm_info = []
        for agent_id in self.agent_ids:
            agent_ctx = self.agent_ctxs[agent_id]
            
            for policy_id in agent_ctx.policy:
                format_action = agent_ctx.pred_output[policy_id][agent_id]['format_action']
                network_sample_info = agent_ctx.pred_output[policy_id][agent_id]['network_sample_info']
                # lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                format_action_list.append(format_action)

                self.from_actor_model_version = agent_ctx.pred_output[policy_id][agent_id]['model_version']

        return format_action_list, network_sample_info, lstm_info
            
    '''
    样本生产
    '''
    def gen_train_data(self, agent_id, policy_id, del_last=False):
        agent_ctx = self.agent_ctxs[agent_id]
        expr_processor = agent_ctx.expr_processor[policy_id]
        policy = agent_ctx.policy[policy_id]

        # 需要发送样本数据才会发送
        if expr_processor.should_train():
            with TimeIt() as ti:
                train_data, train_data_prioritezeds, train_frame_cnt, return_rew = expr_processor.proc_exprs(del_last)

            # self.logger.debug(f'kaiwu_rl_helper this loop train_frame_cnt is {train_frame_cnt}, drop_frame_cnt is {drop_frame_cnt}, train_data is {train_data}')

            if train_frame_cnt > 0:
                policy.send_train_data(train_data, train_data_prioritezeds, agent_ctx)
                self.logger.debug(f'kaiwu_rl_helper aisrv send to learner: {train_data}')
            
            return return_rew

    def stop(self):
        self.exit_flag.value = True
        self.env.finsh()
        for __, policy in self.policies.items():
            policy.stop()

        self.logger.info("kaiwu_rl_helper success stop")

    def normalize_policy_ids(self, ids):
        assert isinstance(
            ids, (str, list)), "only str or list of str is supported"
        if isinstance(ids, str):
            ids = [ids]
        return ids

    # 单个agent_id的停止
    def stop_agent(self, agent_id):
        self.logger.info(f"kaiwu_rl_helper stop agent {agent_id}")
        agent_ctx = self.agent_ctxs[agent_id]

        policy_ids = self.normalize_policy_ids(
            self.env.policy_mapping_fn(agent_id))
        for policy_id in policy_ids:
            if agent_ctx.policy[policy_id].need_train():
                if not self.use_sample_server:
                    agent_ctx.expr_processor[policy_id].finalize()

        del self.agent_ctxs[agent_id]

    # 单个agent_id的启动
    def start_agent(self, agent_id):
        ''' 形如以下配置
        "policies": {
                    "train": {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder", // 
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    },
                    "predict": {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }

        '''
        agent_ctx = AgentContext()
        agent_ctx.done = False
        agent_ctx.agent_id = agent_id

        # 设置主要main_id, policy_ids为策略列表
        np.random.seed(int(time.time()*1000) % (2**20))
        policy_ids = list(AppConf[CONFIG.app].policies.keys())

        # 每一个agent在启动时就需要确定唯一的policy，但是两边对弈的agent可以是不同policy
        if int(CONFIG.self_play):
            # 如果agent为self_play_agent那么其策略为策略列表中的对应策略
            if agent_id == CONFIG.self_play_agent_index:
                agent_ctx.main_id = policy_ids[CONFIG.self_play_agent_index]
                policy_ids = [policy_ids[CONFIG.self_play_agent_index]]
                assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"

            elif agent_id == CONFIG.self_play_old_agent_index:
                # 当agent_id为对手策略时，80%设置为新策略，20%为旧策略
                if np.random.uniform() <= (1 - float(CONFIG.self_play_new_ratio)):
                    agent_ctx.main_id = policy_ids[CONFIG.self_play_old_agent_index]
                    policy_ids = [policy_ids[CONFIG.self_play_old_agent_index]]
                    assert agent_ctx.main_id == CONFIG.self_play_old_policy, "Check your config of self_play_old_policy"
                else:
                    agent_ctx.main_id = policy_ids[CONFIG.self_play_agent_index]
                    policy_ids = [policy_ids[CONFIG.self_play_agent_index]]
                    assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"
        else:
            # 如果不是self-play模式，那么agent自动加载第一种policy
            agent_ctx.main_id = policy_ids[0]
            policy_ids = [policy_ids[0]]

        self.logger.info(
            f"kaiwu_rl_helper start agent {agent_id} with {policy_ids[0]}")

        ''' policy conf, 形如
                    "train_one": {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }
        '''
        agent_ctx.policy_conf = {}
        '''
        policy, 形如"policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
        '''
        agent_ctx.policy = {}
        # 预测的响应结果
        agent_ctx.pred_output = {}
        # 样本生成
        agent_ctx.expr_processor = {}
        agent_ctx.start_time = time.monotonic()

        # aisrv发送给actor的message id, 从1自增
        agent_ctx.message_id = 1

        # aisrv发送给actor的model_version, 由actor负责赋值
        agent_ctx.model_version = -1

        #policy_ids的列表长度根据运行模式不一致, 比如self-play是1, 非self-play的需要看具体情况
        for policy_id in policy_ids:
            policy_conf = AppConf[CONFIG.app].policies[policy_id]
            policy = self.policies[policy_id]
            agent_ctx.policy_conf[policy_id] = policy_conf
            agent_ctx.policy[policy_id] = policy

            if policy.need_train():
                assert hasattr(
                    policy_conf, "algo"), "trainable policy need to specify algo"
                '''
                 {
                     "ppo": {
                         "actor_model": "framework.common.algorithms.model.Model",
                         "learner_model": "framework.common.algorithms.model.Model",
                         "trainer": "framework.server.learner.ppo_trainer.PPOTrainer",
                         "predictor": "framework.server.actor.ppo_predictor.PPOPredictor",
                         "expr_processor": "framework.common.algorithms.ppo_processor.PPOProcessor",
                         "default_config": "framework.common.algorithms.ppo.PPODefaultConfig"
                     }
                 }
                 '''

                agent_ctx.expr_processor[policy_id] = AlgoConf[policy_conf.algo].expr_processor()
                agent_ctx.expr_processor[policy_id].on_init(1, 'game_id')
                agent_ctx.expr_processor[policy_id].agent_policy.append(policy_id)

        self.agent_ctxs[agent_id] = agent_ctx
