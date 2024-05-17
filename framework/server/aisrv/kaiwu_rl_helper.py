#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
@Project: kaiwu-fwk
@File    :kaiwu_rl_helper.py
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

SAMPLE_CUT_POINT = [14, 39, 81, 123, 126, 187, 188, 189, 190, 191, 192, 193]

# 实现标准的强化学习训练流程


class KaiWuRLHelper(threading.Thread):
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

    """ 
    处理预测
        : 处理前的参数形如: {agent_id: {pred_input_key: pred_input_value, ...}}
        : 处理后的结果形如: {agent_id: {pred_output_key: pred_output_value}, ...}
    """

    def predict(self, agent_ids):
        # 两个agent是同一个policy情况下的预测处理，只适用5v5，将两个样本拼在一起进行预测，保证两个样本能够同时被处理，减小预测耗时
        if len(agent_ids) == 2 and list(self.agent_ctxs[0].policy.keys())[0] == list(self.agent_ctxs[1].policy.keys())[0] and CONFIG.aisrv_actor_protocl == KaiwuDRLDefine.PROTOCL_PROTOBUF:
            policy_id = list(self.agent_ctxs[0].policy.keys())[0]
            success, actor_address = self.agent_ctxs[0].policy[policy_id].send_pred_data_v2(self.slot_id,
                                                                                            self.agent_ctxs[0].pred_input[policy_id],
                                                                                            self.agent_ctxs[1].pred_input[policy_id],
                                                                                            self.agent_ctxs[0],
                                                                                            self.agent_ctxs[1])
            if not success:
                self.logger.error(
                    f"kaiwu_rl_helper policy_id {policy_id}  send_pred_data_v2 to actor {actor_address} failed")
        else:
            for agent_id in agent_ids:
                agent_ctx = self.agent_ctxs[agent_id]
                for policy_id in agent_ctx.policy:
                    # 调用AsyncPolicy的send_pred_data函数
                    success, actor_address = agent_ctx.policy[policy_id].send_pred_data(
                        self.slot_id, agent_ctx.pred_input[policy_id], agent_ctx)
                    # self.logger.debug(f'kaiwu_rl_helper aisrv send to actor: {agent_ctx.pred_input[policy_id]}')
                    if not success:
                        self.logger.error(
                            f"kaiwu_rl_helper policy_id {policy_id} agent_id {agent_id} send_pred_data to actor {actor_address} failed")
                        continue

        for agent_id in agent_ids:
            agent_ctx = self.agent_ctxs[agent_id]
            agent_ctx.pred_output = {}
            for policy_id in agent_ctx.policy:
                # 调用AsyncPolicy的get_pred_result函数
                pred_output = agent_ctx.policy[policy_id].get_pred_result(
                    self.slot_id, agent_ctx)
                # self.logger.debug(f'kaiwu_rl_helper aisrv recv from actor: {pred_output}')
                if not pred_output:
                    self.logger.error("kaiwu_rl_helper get_pred_result failed")
                else:
                    agent_ctx.pred_output[policy_id] = pred_output

    # gym单局游戏的处理逻辑
    def episode_main_loop(self, states):
        while not self.exit_flag.value:
            self.episode_start_time = time.monotonic()

            valid_agents = list(states.keys())
            self.logger.debug("kaiwu_rl_helper start episode_main_loop")

            # 1, 准备预测的数据, 包含特征值抽取
            eigent_data = None  # 从on_update请求里获取的数据, 放在states里
            with TimeIt() as ti:
                for agent_id in valid_agents:
                    # agent_id不存在则新建
                    if agent_id not in self.agent_ctxs:
                        self.start_agent(agent_id)
                    agent_ctx = self.agent_ctxs[agent_id]

                    agent_ctx.state, agent_ctx.pred_input, agent_ctx.eigent_value = {}, {}, {}

                    # 特征值抽取
                    agent_ctx.eigent_value = agent_ctx.policy_conf[agent_ctx.main_id].eigent_value().get_eigent_data(
                        self.client_address, eigent_data, self.simu_ctx)

                    for policy_id, state in states[agent_id].items():
                        s = state.get_state()
                        assert all(state.state_space()[
                                   k].dtype == s[k].dtype for k in s)
                        agent_ctx.pred_input[policy_id] = s
                        agent_ctx.state[policy_id] = state

            self.logger.debug("kaiwu_rl_helper prepare data success")

            # 2, 预测
            with TimeIt() as ti:
                self.predict(valid_agents)
            self.logger.debug("kaiwu_rl_helper predict success")

            # 3, action
            actions = {}  # agent_id -> action_obj
            for agent_id in valid_agents:
                agent_ctx = self.agent_ctxs[agent_id]
                action_dict = {}
                action_class = agent_ctx.policy_conf[agent_ctx.main_id].action
                for action_name in action_class.action_space():
                    action_dict[action_name] = agent_ctx.pred_output[agent_ctx.main_id][agent_id][action_name]
                actions[agent_id] = action_class(**action_dict)
                agent_ctx.action = actions[agent_id]
            self.logger.debug("kaiwu_rl_helper agent_id -> action_obj success")

            # 4, step
            with TimeIt() as ti:
                new_states, ex_rewards, dones = self.env.step(actions, extra_info={
                    agent_id: self.agent_ctxs[agent_id].pred_output for agent_id in valid_agents
                })
            self.logger.debug("kaiwu_rl_helper step success")

            # 5, 判断是否结束
            terminal = dones.pop("_all_done_")
            if terminal:
                break
            self.logger.debug("kaiwu_rl_helper _all_done_ success")

            # 6, 生成经验数据
            for agent_id in valid_agents:
                agent_ctx = self.agent_ctxs[agent_id]
                reward = agent_ctx.policy_conf[agent_ctx.main_id].reward()
                done = dones[agent_id]
                if not done:
                    reward.extend_ex_reward(ex_rewards[agent_id])

                agent_ctx.reward = reward
                agent_ctx.done = done
            self.logger.debug("kaiwu_rl_helper extend_ex_reward success")

            # 7, 生成训练数据
            for agent_id in valid_agents:
                agent_ctx = self.agent_ctxs[agent_id]
                for policy_id in agent_ctx.policy:
                    if agent_ctx.policy[policy_id].need_train():
                        self.gen_train_data(agent_id, policy_id)
                if agent_ctx.done:
                    self.stop_agent(agent_id)
            self.logger.debug("kaiwu_rl_helper gen_train_data success")

            # 8, 更新当前states为new_states
            states = new_states

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

    def gen_expr_server(self, agent_id, policy_id, sample_info_list, must_need_sample_info):
        # sample_server只有一个，两个agent的样本会同时存储
        policy = self.agent_ctxs[agent_id].policy[policy_id]
        policy.gen_frame_sample(
            self.slot_id, sample_info_list, must_need_sample_info)

    def sample_server_gameover(self, agent_id, policy_id):
        policy = self.agent_ctxs[agent_id].policy[policy_id]
        return policy.sample_server_gameover(self.slot_id)

    def add_policy_to_sample_server(self, agent_id, policy_id, main_id):
        policy = self.agent_ctxs[agent_id].policy[policy_id]
        return policy.add_policy_to_sample_server(self.slot_id, main_id)

    def gen_train_data(self, agent_id, policy_id, del_last=False):
        agent_ctx = self.agent_ctxs[agent_id]
        expr_processor = agent_ctx.expr_processor[policy_id]
        policy = agent_ctx.policy[policy_id]
        if CONFIG.app == KaiwuDRLDefine.APP_GYM:
            expr_processor.gen_expr()

        # 需要发送样本数据才会发送
        if expr_processor.should_train():
            with TimeIt() as ti:
                train_data, train_data_prioritezeds, train_frame_cnt, return_rew = expr_processor.proc_exprs(
                    del_last)
                if policy_id == agent_ctx.main_id:

                    # 这里是需要满足reverb的存储格式, 扩展维度, 否则会报错
                    if CONFIG.app == KaiwuDRLDefine.APP_GYM:
                        for k in train_data.keys():
                            if len(train_data[k].shape) == 1:
                                train_data[k] = np.expand_dims(
                                    train_data[k], axis=-1)

            # self.logger.debug(f'kaiwu_rl_helper this loop train_frame_cnt is {train_frame_cnt}, drop_frame_cnt is {drop_frame_cnt}, train_data is {train_data}')

            if train_frame_cnt > 0:
                policy.send_train_data(train_data, train_data_prioritezeds, agent_ctx)
                # self.logger.debug(f'kaiwu_rl_helper aisrv send to learner: {train_data}')
            return return_rew

    def run_episode(self):

        with TimeIt() as ti:
            if CONFIG.app == KaiwuDRLDefine.APP_GYM:
                # 某局游戏的初始化操作
                init_states = self.env.reset()
                self.episode_main_loop(init_states)

            elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_1V1:
                self.sgame_1v1_episode_main_loop()

            elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                self.sgame_5v5_episode_main_loop()

            elif CONFIG.app == KaiwuDRLDefine.APP_GORGE_WALK_V1:
                self.gorge_walk_episode_main_loop()
                
            elif CONFIG.app == KaiwuDRLDefine.APP_GORGE_WALK_V2:
                self.gorge_walk_episode_main_loop()
                
            else:
                pass

    def gorge_walk_episode_main_loop(self):
        counter = 0
        states, must_need_sample_info = self.env.next_valid()
        if not states:
            raise

        while not self.exit_flag.value:
            try:
                valid_agents = list(states.keys())
                # 如果没有初始化，则进行初始化
                for agent_id in valid_agents:
                    if agent_id not in self.agent_ctxs:
                        self.start_agent(agent_id)

                    agent_ctx = self.agent_ctxs[agent_id]
                    agent_ctx.state, agent_ctx.pred_input = {}, {}

                    policy_id = agent_ctx.main_id
                    s = states[agent_id].get_state()
                    agent_ctx.pred_input[policy_id] = s
                    agent_ctx.state[policy_id] = states[agent_id]

                # 执行预测
                self.predict(valid_agents)
                self.logger.debug("kaiwu_rl_helper predict success")

                # 解析action
                format_action_list = []
                for agent_id in valid_agents:
                    agent_ctx = self.agent_ctxs[agent_id]
                    for policy_id in agent_ctx.policy:
                        format_action = agent_ctx.pred_output[policy_id][agent_id]['format_action']
                        network_sample_info = agent_ctx.pred_output[policy_id][agent_id]['network_sample_info']
                        lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                        format_action_list.append(format_action)

                # __gorge_walk_step
                self.env.on_handle_action(format_action_list)
                _states, must_need_sample_info = self.env.next_valid()


                # 存储和发送样本的agent_id和policy_id
                agent_id = valid_agents[0]
                agent_ctx = self.agent_ctxs[agent_id]
                policy_id = agent_ctx.main_id

                # 存储样本
                if agent_ctx.policy[policy_id].need_train():
                    self.gen_expr(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, {
                        'must_need_sample_info': {'last_state': states, 'state': _states, 'action': format_action_list, 'info': must_need_sample_info},
                        'network_sample_info': {'log_prob': network_sample_info[0], 'value': network_sample_info[1], 'lstm_cell':lstm_info[0], 'lstm_hidden':lstm_info[1]},
                    })

                # 满足条件发送样本
                if (counter+1) % int(CONFIG.send_sample_size) == 0:
                    for policy_id in agent_ctx.policy:
                        if agent_ctx.policy[policy_id].need_train():
                            self.gen_train_data(agent_id, policy_id)

                # 处理游戏结束信号
                if self.env.run_handler.done:
                    self.logger.info('kaiwu_rl_helper game is over')
                    # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                    for policy_id in agent_ctx.policy:
                        if agent_ctx.policy[policy_id].need_train():
                            self.gen_train_data(agent_id, policy_id)

                        self.logger.debug(
                            "kaiwu_rl_helper gen_train_data success")

                    self.stop_agent(agent_id)
                    # 游戏结束后，不需要预测，但是需要返回一个空的action
                    self.env.on_handle_action([[0, 0]])

                    # 主动退出循环, 框架需要的操作
                    self.exit_flag.value = True
                    # 处理异常退出情况,保存样本
                    if not self.env.run_handler.done:
                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        pass
                    # 结束aisrv的循环
                    self.env.msg_buff.output_q.put(None)
                    break

                # 更新当前state，准备下一轮迭代
                states = _states
                counter += 1

            except AssertionError as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                break
            except TimeoutEpisodeException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break

    def gorge_walk_episode_main_loop_v1(self):
        counter = 0
        while not self.exit_flag.value:
            try:
                states, must_need_sample_info = self.env.next_valid()
                if states:
                    valid_agents = list(states.keys())

                    # 判断游戏结束, 生成和发送样本
                    if self.env.run_handler.done:
                        self.logger.info('kaiwu_rl_helper game is over')

                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        agent_id = valid_agents[0]
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            if agent_ctx.policy[policy_id].need_train():
                                if not self.env.run_handler.first_frame:
                                    self.gen_expr(valid_agents[0], self.agent_ctxs[agent_id].main_id, {
                                        'must_need_sample_info': {'last_state': last_states, 'state': states, 'action': last_format_action_list, 'info': must_need_sample_info},
                                        'network_sample_info': None,
                                    })

                                self.gen_train_data(agent_id, policy_id)
                            # self.logger.debug("kaiwu_rl_helper gen_train_data success")
                            if agent_ctx.done:
                                self.stop_agent(agent_id)
                        # 游戏结束后，不需要预测，但是需要返回一个空的action
                        self.env.on_handle_action([0] * len(valid_agents))
                        break

                    # 准备数据
                    with TimeIt() as ti:
                        for agent_id in valid_agents:
                            if agent_id not in self.agent_ctxs:
                                self.start_agent(agent_id)

                            agent_ctx = self.agent_ctxs[agent_id]
                            agent_ctx.state, agent_ctx.pred_input = {}, {}

                            policy_id = agent_ctx.main_id
                            s = states[agent_id].get_state()
                            agent_ctx.pred_input[policy_id] = s
                            agent_ctx.state[policy_id] = states[agent_id]

                    # 执行预测
                    self.predict(valid_agents)
                    self.logger.debug("kaiwu_rl_helper predict success")

                    # 处理action, 保留样本
                    format_action_list = []
                    network_sample_info_list = []
                    lstm_cell_list, lstm_hidden_list = [], []
                    for agent_id in valid_agents:
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            format_action = agent_ctx.pred_output[policy_id][agent_id]['format_action']
                            network_sample_info = agent_ctx.pred_output[
                                policy_id][agent_id]['network_sample_info']
                            lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                            format_action_list.append(format_action)
                            # network_sample_info_list.append(network_sample_info[0])
                            # lstm_cell_list.append(lstm_info[0][0])
                            # lstm_hidden_list.append(lstm_info[0][1])

                            self.from_actor_model_version = agent_ctx.pred_output[policy_id][agent_id]['model_version']

                    # format_action_list = [0]
                    format_action_list = [[0, 0]]

                    self.env.on_handle_action(format_action_list)

                    # 每次action 操作后, 保留样本
                    if not self.env.run_handler.first_frame and agent_ctx.policy[policy_id].need_train():
                        self.gen_expr(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, {
                            'must_need_sample_info': {'last_state': last_states, 'state': states, 'action': last_format_action_list, 'info': must_need_sample_info},
                            'network_sample_info': None,
                        })

                    last_states, last_must_need_sample_info, last_format_action_list = states, must_need_sample_info, format_action_list

                    counter += 1
                    if counter % int(CONFIG.send_sample_size) == 0:
                        self.gen_train_data(agent_id, policy_id)

                else:
                    if must_need_sample_info == "end":
                        # 主动退出循环
                        self.exit_flag.value = True
                        # 处理异常退出情况,保存样本
                        if not self.env.run_handler.done:
                            # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                            pass
                        # 结束aisrv的循环
                        self.env.msg_buff.output_q.put(None)

            except AssertionError as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                break
            except TimeoutEpisodeException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break

    def run(self) -> None:
        try:
            self.env.init()
        except AssertionError as e:
            self.logger.error(
                f"kaiwu_rl_helper self.env.init() {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            self.env.reject(e)
        except Exception as e:
            self.logger.error(
                f"kaiwu_rl_helper self.env.init() {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            self.env.reject(e)
        else:
            self.client_id = self.env.client_id
            try:
                self.agent_ctxs = {}
                self.simu_ctx.agent_ctxs = self.agent_ctxs

                def run_episode_once():
                    self.run_episode()

                while not self.exit_flag.value:
                    try:
                        run_episode_once()

                    except SkipEpisodeException:
                        self.logger.error(
                            "kaiwu_rl_helper run_episode_once() SkipEpisodeException {}", str(e))
                        pass
            except ClientQuitException:
                self.logger.error(
                    f"kaiwu_rl_helper run_episode_once() ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            except TimeoutEpisodeException as e:
                self.logger.error(
                    f"kaiwu_rl_helper run_episode_once() TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
            except AssertionError as e:
                self.logger.error(
                    f"kaiwu_rl_helper run_episode_once() AssertionError {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper run_episode_once() Exception {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                if not self.exit_flag.value:
                    self.env.reject(e)
        finally:
            self.logger.info("kaiwu_rl_helper finally")
            self.stop()

    def sgame_1v1_episode_main_loop(self):

        while not self.exit_flag.value:
            try:
                states, must_need_sample_info = self.env.next_valid()
                if states:
                    valid_agents = list(states.keys())

                    # 判断游戏结束, 生成和发送样本
                    if self.env.run_handler.done:
                        self.logger.info('kaiwu_rl_helper game is over')
                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        agent_id = valid_agents[0]
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            if agent_ctx.policy[policy_id].need_train():
                                self.reward_value = self.gen_train_data(
                                    agent_id, policy_id)
                                # 传入上报数据
                                self.data_queue.append(abs(self.reward_value))
                            # self.logger.debug("kaiwu_rl_helper gen_train_data success")
                            if agent_ctx.done:
                                self.stop_agent(agent_id)
                        # 游戏结束后，不需要预测
                        break

                    # 准备数据
                    with TimeIt() as ti:
                        for agent_id in valid_agents:
                            if agent_id not in self.agent_ctxs:
                                self.start_agent(agent_id)

                            agent_ctx = self.agent_ctxs[agent_id]
                            agent_ctx.state, agent_ctx.pred_input = {}, {}

                            policy_id = agent_ctx.main_id
                            s = states[agent_id].get_state()
                            agent_ctx.pred_input[policy_id] = s
                            agent_ctx.state[policy_id] = states[agent_id]

                    # self.logger.debug("kaiwu_rl_helper prepare Msg success")

                    # 执行预测
                    self.predict(valid_agents)
                    # self.logger.debug("kaiwu_rl_helper predict success")

                    # 处理action, 保留样本
                    format_action_list = []
                    network_sample_info_list = []
                    lstm_cell_list, lstm_hidden_list = [], []
                    for agent_id in valid_agents:
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            format_action = agent_ctx.pred_output[policy_id][agent_id]['format_action']
                            network_sample_info = agent_ctx.pred_output[
                                policy_id][agent_id]['network_sample_info']
                            lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                            format_action_list.append(format_action)
                            network_sample_info_list.append(
                                network_sample_info[0])
                            lstm_cell_list.append(lstm_info[0][0])
                            lstm_hidden_list.append(lstm_info[0][1])

                    self.env.on_handle_action(format_action_list)

                    # 每次action 操作后, 保留样本
                    self.gen_expr(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, {
                        'network_sample_info': network_sample_info_list,
                        'must_need_sample_info': must_need_sample_info
                    })
                    self.env.run_handler.update_lstm(
                        lstm_cell_list, lstm_hidden_list)
                    # self.logger.debug("kaiwu_rl_helper handle action and gen_expr success")

                else:
                    if must_need_sample_info == "end":
                        # 主动退出循环
                        self.exit_flag.value = True
                        # 处理异常退出情况,保存样本
                        if not self.env.run_handler.done:
                            # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                            if self.agent_ctxs:
                                agent_id = list(self.agent_ctxs.keys())[0]
                                agent_ctx = self.agent_ctxs[agent_id]
                                for policy_id in agent_ctx.policy:
                                    if agent_ctx.policy[policy_id].need_train():
                                        self.reward_value = self.gen_train_data(
                                            agent_id, policy_id, del_last=True)
                                self.logger.info(
                                    "kaiwu_rl_helper gen_train_data success")

                        # 结束aisrv的循环
                        self.env.msg_buff.output_q.put(None)

            except AssertionError as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_1v1_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_1v1_episode_main_loop ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                break
            except TimeoutEpisodeException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_1v1_episode_main_loop TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_1v1_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break

    def sgame_5v5_episode_main_loop(self):

        while not self.exit_flag.value:
            try:
                states, must_need_sample_info = self.env.next_valid()
                if states:
                    valid_agents = list(states.keys())

                    # 判断游戏结束, 生成和发送样本
                    if self.env.run_handler.done:
                        self.logger.info('kaiwu_rl_helper game is over')
                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        agent_id = valid_agents[0]
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            if agent_ctx.policy[policy_id].need_train():
                                if not self.use_sample_server:
                                    self.reward_value += self.gen_train_data(
                                        agent_id, policy_id)
                                else:
                                    self.reward_value += self.sample_server_gameover(
                                        agent_id, policy_id)

                                # 传入上报数据
                                self.data_queue.append(abs(self.reward_value))

                            self.logger.debug(
                                "kaiwu_rl_helper gen_train_data success")
                            if agent_ctx.done:
                                self.stop_agent(agent_id)

                        # 游戏结束后，不需要预测
                        break

                    # 准备数据
                    sample_info_list = {i: [] for i in valid_agents}
                    with TimeIt() as ti:
                        for agent_id in valid_agents:
                            if agent_id not in self.agent_ctxs:
                                self.start_agent(agent_id)
                                if self.use_sample_server:
                                    self.add_policy_to_sample_server(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id,
                                                                     self.agent_ctxs[agent_id].main_id)
                            agent_ctx = self.agent_ctxs[agent_id]
                            agent_ctx.state, agent_ctx.pred_input = {}, {}
                            policy_id = agent_ctx.main_id
                            s = states[agent_id].get_state()
                            agent_ctx.pred_input[policy_id] = s
                            agent_ctx.state[policy_id] = states[agent_id]

                            sample_info_list[agent_id].append(s['observation'])
                            sample_info_list[agent_id].append(s['lstm_hidden'])
                            sample_info_list[agent_id].append(s['lstm_cell'])
                            # sample_info_list[agent_id].append(must_need_sample_info['reward'][agent_id])

                    self.logger.debug("kaiwu_rl_helper prepare Msg success")

                    # 执行预测
                    self.predict(valid_agents)
                    self.logger.debug("kaiwu_rl_helper predict success")

                    logits_list = []
                    value_list = []
                    meta_msg_list = []
                    lstm_cell_list = []
                    lstm_hidden_list = []
                    for agent_id in valid_agents:
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            network_sample_info = agent_ctx.pred_output[policy_id][agent_id]
                            #lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info'][0]
                            logits, value, meta_msg, lstm_cell, lstm_hidden = network_sample_info
                            logits_list.append(logits)
                            value_list.append(value)
                            meta_msg_list.append(meta_msg)
                            lstm_cell_list.append(lstm_cell)
                            lstm_hidden_list.append(lstm_hidden)
                            sample_info_list[agent_id].append(
                                value.reshape([5, -1]))

                    lstm_hidden_list = np.concatenate(
                        lstm_hidden_list).reshape([10, -1])
                    lstm_cell_list = np.concatenate(
                        lstm_cell_list).reshape([10, -1])

                    logits = np.concatenate(logits_list).reshape(10, -1)
                    value = np.concatenate(value_list).reshape(10, -1)
                    meta_msg = np.concatenate(meta_msg_list).reshape(10, -1)
                    action_all = np.concatenate(
                        [logits, value, meta_msg], axis=1)
                    #action_all = np.reshape(action_all, [10, -1])
                    action_all = np.split(action_all, SAMPLE_CUT_POINT, axis=1)
                    format_action = []
                    for i in range(10):
                        format_action.append([arr[i].tolist()
                                             for arr in action_all])
                    '''
                    ##供5v5测试使用的伪造输出数据
                    format_action_list = [[[1 for _ in range(14)], [1 for _ in range(25)], [1 for _ in range(42)], [1 for _ in range(42)],
                                    [1 for _ in range(3)], [1 for _ in range(61)], [1 for _ in range(1)], [1 for _ in range(1)],
                                    [1 for _ in range(1)], [1 for _ in range(1)], [1 for _ in range(1)], [1 for _ in range(1)],
                                    [1 for _ in range(64)]] for _ in range(10)]
                    '''
                    # 处理action, 保留样本
                    legal_actions, sub_actionss, actionss, probs_lists, is_trains, rewards = self.env.on_handle_action(
                        format_action)

                    hero_nums = len(legal_actions)
                    for i in range(hero_nums//5):
                        sample_info_list[i].append(legal_actions[i*5:i*5+5])
                        sample_info_list[i].append(
                            sub_actionss[i * 5:i * 5 + 5])
                        sample_info_list[i].append(actionss[i * 5:i * 5 + 5])
                        sample_info_list[i].append(
                            probs_lists[i * 5:i * 5 + 5])
                        sample_info_list[i].append(is_trains[i * 5:i * 5 + 5])

                        sample_info_list[i].insert(3, rewards[i * 5:i * 5 + 5])
                    # feature, lstm_hidden, lstm_cell, reward, value, legal_actions,sub_actionss,actionss,probs_lists, is_trains

                    # 每次action 操作后, 保留样本
                    if not self.use_sample_server:
                        self.gen_expr(valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, {
                            'network_sample_info': sample_info_list,
                            'must_need_sample_info': must_need_sample_info
                        })
                    else:
                        # 采用sample_server的方式来存储样本
                        # 默认第一个agent对应的是new_policy，所以其simtux中的policy是有sample_server_list的
                        self.gen_expr_server(
                            valid_agents[0], self.agent_ctxs[valid_agents[0]].main_id, sample_info_list, must_need_sample_info)

                    self.env.run_handler.update_lstm(
                        lstm_cell_list, lstm_hidden_list)
                    self.logger.debug(
                        "kaiwu_rl_helper handle action and gen_expr success")

                    # 样本数量到达固定值，则发送样本节约内存，并行对局越多固定步数则应越小
                    if not self.use_sample_server:
                        if self.steps > 0 and self.steps % int(CONFIG.send_sample_size) == 0:
                            self.logger.info(
                                'kaiwu_rl_helper send samples to learner during gaming')

                            # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                            agent_id = valid_agents[0]
                            agent_ctx = self.agent_ctxs[agent_id]
                            for policy_id in agent_ctx.policy:
                                if agent_ctx.policy[policy_id].need_train():
                                    self.reward_value += self.gen_train_data(
                                        agent_id, policy_id)

                            self.logger.info(
                                f'kaiwu_rl_helper reward sum of current frames is {self.reward_value}')
                            self.logger.debug(
                                "kaiwu_rl_helper gen_train_data success")

                            # 游戏继续

                    self.steps += 1

                else:
                    if must_need_sample_info == "end":
                        # 主动退出循环
                        self.exit_flag.value = True
                        # 处理异常退出情况,无需保存样本

                        if not self.env.run_handler.done:
                            # 样本生成器是单例模式, 里面包含所有Agent生成的样本, 因此只需要调用一次
                            if self.agent_ctxs:
                                agent_id = list(self.agent_ctxs.keys())[0]
                                agent_ctx = self.agent_ctxs[agent_id]
                                for policy_id in agent_ctx.policy:
                                    if agent_ctx.policy[policy_id].need_train():
                                        if not self.use_sample_server:
                                            self.reward_value += self.gen_train_data(
                                                agent_id, policy_id, del_last=True)
                                        else:
                                            self.reward_value += self.sample_server_gameover(
                                                agent_id, policy_id)

                                # 传入上报数据
                                self.data_queue.append(abs(self.reward_value))
                                self.logger.info(
                                    "kaiwu_rl_helper gen_train_data success")

                        # 现在handler的结束时通过回复end包来完成的
                        # 一种是正常对局结束，由bt回复end包，一种是bt异常退出，由aisrv_socket来回复end包
                        # 在收到end包后，handler结束循环，需要在发送空包到msg_buff.output_q来结束aisrv_socket的循环
                        self.env.msg_buff.output_q.put(None)

            except AssertionError as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_5v5_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_5v5_episode_main_loop ClientQuitException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                break
            except TimeoutEpisodeException as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_5v5_episode_main_loop TimeoutEpisodeException {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.error(
                    f"kaiwu_rl_helper sgame_5v5_episode_main_loop {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
                self.env.reject(e)
                break

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

                # 样本生成类暂时还没有稳定, 采用配置项来设置
                if CONFIG.app == KaiwuDRLDefine.APP_GYM:
                    agent_ctx.expr_processor[policy_id] = AlgoConf[policy_conf.algo].expr_processor(self.simu_ctx,
                                                                                                    agent_ctx,
                                                                                                    policy_id)
                    agent_ctx.expr_processor[policy_id].initialize()
                
                elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                    if not self.use_sample_server:
                        agent_ctx.expr_processor[policy_id] = AlgoConf[policy_conf.algo].expr_processor(
                        )
                        agent_ctx.expr_processor[policy_id].agent_policy.append(
                            policy_id)

                else:
                    agent_ctx.expr_processor[policy_id] = AlgoConf[policy_conf.algo].expr_processor(
                    )
                    agent_ctx.expr_processor[policy_id].agent_policy.append(
                        policy_id)
  

        self.agent_ctxs[agent_id] = agent_ctx
