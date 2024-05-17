#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass


@dataclass
class AgentContext:
    """
    agent_id: 指示当前agent id编号
    done: 指示当前episode是否已经完成
    policy_conf: policy_conf的dict对象, 从app_conf.json中的"policies"字段获取，格式为{policy_id: policy_conf}
    policy: 根据policy_conf创建出来的policy的dict对象, 格式为{policy_id: policy_obj}
    main_id: 主的policy id, 只有在主policy中才拥有完整的MDP过程(即只有一个action、reward和reward_shaper)
    pred_output: predict返回值, 格式为{policy_id: data dict}
    expr_processor: 根据algo_conf.json创建得到的expr_processor对象
    start_time: 当前agent的启动时间
    reward: 根据app_conf.json配置定义的reward对象
    """
    agent_id: int = -1
    done: bool = False
    policy_conf: dict = None
    policy: dict = None
    main_id: str = None
    pred_output: dict = None
    expr_processor: dict = None
    start_time: float = 0
    reward: object = None
