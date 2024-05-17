#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

class Expr:
    """
    所有Experence的基类
    """
    def __init__(self, state, action, reward, done):
        """
        :param state: State类型
        :param action: Action类型
        :param reward: Reward类型
        :param done: bool
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done


'''
样本生成的基类
'''
class ExprProcessor:
    def __init__(self, simu_ctx, agent_ctx, policy_id):
        self._simu_ctx = simu_ctx
        self._agent_ctx = agent_ctx
        self._policy_id = policy_id

    def gen_expr(self):
        raise NotImplementedError

    def proc_exprs(self):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError
    
    def finish(self):
        raise NotImplementedError
    
    def should_train(self):
        raise NotImplementedError


'''
强化学习的样本生成类
'''
class RLExprProcessor(ExprProcessor):
    def __init__(self, simu_ctx, agent_ctx, policy_id):
        super(RLExprProcessor, self).__init__(simu_ctx, agent_ctx, policy_id)

        # 每个agent只有主要的policy和环境进行交互
        self._reward_shaper = agent_ctx.policy_conf[agent_ctx.main_id].reward_shaper(simu_ctx, agent_ctx)

        self._exprs = []
        # 确保overlap时，expr的reward不会被重复计算
        self._summed_rewards = []
        self._reward_sum = 0
        self._in_reward_sum = 0
        self._ex_reward_sum = 0

    def initialize(self):
        self._reward_shaper.initialize()
    
    def finalize(self):
        self._reward_shaper.finalize()
    
    def should_train(self):
        return self._reward_shaper.should_train(self._exprs)

    def _gen_expr(self):
        """
        :return: 一个训练样本(experience)对象
        """
        raise NotImplementedError
    
    def gen_expr(self):
        assert len(self._exprs) == len(self._summed_rewards)
        self._exprs.append(self._gen_expr())
        self._summed_rewards.append(False)
    
    def _proc_exprs(self):
        """
        :return train_data 训练数据
        :return train_frame_cnt 训练的样本数
        :return drop_frame_cnt 无效的样本数
        """
        raise NotImplementedError

    @property
    def reward_sum(self):
        # exprs的reward总和（不包含overlap的部分）
        return self._reward_sum

    @property
    def in_reward_sum(self):
        # exprs的外部reward（人工设计）总和（不包含overlap的部分）
        return self._in_reward_sum

    @property
    def ex_reward_sum(self):
        # exprs的环境reward总和（不包含overlap）
        return self._ex_reward_sum
    
    def proc_exprs(self):
        # handle rewards
        repeat_expr = sum([1 if summed_reward else 0 for summed_reward in self._summed_rewards])

        # 重复的expr不需要再次计算reward
        self._reward_shaper.assign_rewards(self._exprs[repeat_expr:])
        self._reward_sum = sum([expr.reward.get_reward() if not summed_rwd else 0
                                for summed_rwd, expr in zip(self._summed_rewards, self._exprs)])
        self._in_reward_sum = sum([expr.reward.get_in_reward() if not summed_rwd else 0
                                   for summed_rwd, expr in zip(self._summed_rewards, self._exprs)])
        self._ex_reward_sum = sum([expr.reward.get_ex_reward() if not summed_rwd else 0
                                   for summed_rwd, expr in zip(self._summed_rewards, self._exprs)])

        # gen train samples
        train_data, train_frame_cnt, drop_frame_cnt = self._proc_exprs()

        # update exprs
        self._exprs.clear()
        self._summed_rewards.clear()

        return train_data, train_frame_cnt, drop_frame_cnt