#!/usr/bin/env python
# -*- coding: utf-8 -*-


class RewardShaper:
    """
    辅助生成更好的reward数据, 主要接口如下:
    """

    def __init__(self, simu_ctx, agent_ctx):
        """
        :param simu_ctx: simualtor context, 包含全局的上下文信息
        :param agent_ctx: agent context, 包含agent相关的上下文信息
        """
        self._simu_ctx = simu_ctx
        self._agent_ctx = agent_ctx

    def initialize(self):
        """
        根据static_info来初始化reward_shaper
        """

    def should_train(self, exprs):
        """
        返回bool值告知框架是否可以进行下一轮训练
        :param exprs: experence列表, 每一项都是Expr类型
        """
        raise NotImplementedError

    def assign_rewards(self, exprs):
        """
        为每一个state赋予intrinsic reward值
        :param exprs: experence列表, 每一项都是Expr类型
        """
        raise NotImplementedError

    def finalize(self):
        """
        RewardShaper的资源回收工作
        """
