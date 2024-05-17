#!/usr/bin/env python
# -*- coding: utf-8 -*-


class EigenValue:
    def get_eigent_data(self, client_conn_id, eigent_value, agent_ctx):
        """
        发送预测数据
        :param client_conn_id: string类型, 用于标识客户端
        :param pred_data: 字典类型，包含所有预测的输入数据
        :param agent_ctx: Context对象, 包含aisrv相关的上下文
        """
        raise NotImplementedError