#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Reward:
    """
    将一帧中相关的ex_rewards和in_rewards组装形成最终本帧的reward值
    """

    def __init__(self):
        self._ex_rewards = []  # 外部环境赋予的reward
        self._in_rewards = []  # 内部模型计算获得reward
        self.in_reward_weights = [1.0] # 内部模型计算的reward权重
        self.ex_reward_weights = [1.0] # 外部环境赋予的reward权重

    def extend_ex_reward(self, ex_rewards):
        """
        :param ex_rewards: 由环境生成的extrinsic reward
        """
        self._ex_rewards.extend(ex_rewards)

    def add_in_reward(self, in_reward):
        """
        :param in_reward: 由模型生成的intrinsic reward
        """
        self._in_rewards.append(in_reward)

    def get_reward(self):
        """
        :return: float类型, 按照一定的规则形成最终的reward值
        """
        return self.get_in_reward() + self.get_ex_reward()

    def get_in_reward(self):
        final_reward = 0.0
        for weight, in_reward in zip(self.in_reward_weights, self._in_rewards):
            final_reward += weight * in_reward
        return final_reward

    def get_ex_reward(self):
        final_reward = 0.0
        for weights, ex_reward in zip(self.ex_reward_weights, self._ex_rewards):
            final_reward += weights * ex_reward
        return final_reward
