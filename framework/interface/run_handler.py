#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
业务处理类, aisrv加载执行
'''
class RunHandler:
    __slots__=("_simu_ctx")
    def __init__(self, simu_ctx):
        """
        初始类, 其中simu_ctx作为框架和业务沟通的媒介, 其包含的参数如下:
        1. client_address, 客户端ID
        2. logger, 日志句柄, 框架提供, 业务可以直接使用, 采用self.logger.info()等即可
        3. exit_flag, 本次训练任务是否结束
        """
        self._simu_ctx = simu_ctx

    def on_init(self, client_id, req_data):
        """
        标识一个客户端建立建立，可用来完成初始化的工作
        :param client_id: str类型, 用于唯一标识客户端
        :param req_data: byte数组, 业务自定义数据
        """

    def on_ep_start(self, client_id, ep_id, req_data):
        """
        标识一个episode的开始, 一般在所有on_agent_start调用之前被调用, 可以用来完成episode的初始化工作
        :param client_id: str类型, 用于唯一标识客户端
        :param ep_id: int类型, 用于唯一标识一个episode
        :param req_data: byte数组, 业务自定义数据
        """

    def on_agent_start(self, client_id, ep_id, agent_id, req_data):
        """
        标识一个agent的开始, 可以用来完成agent的初始化工作
        :param client_id: str类型, 用于唯一标识客户端
        :param ep_id: int类型, 用于唯一标识一个episode
        :param agent_id: int类型, 用于唯一标识一个agent
        :param req_data: byte数组, 业务自定义数据
        """

    def on_update_req(self, client_id, ep_id, req_data):
        """
        在收到update请求之后被调用, 用来处理状态数据
        :param client_id: str类型, 用于唯一标识客户端
        :param ep_id: int类型, 用于唯一标识一个episode
        :param req_data: dict类型, key是准备好的agent_id, value是一个数组, 每个数组元素是一个byte数组
        :return: new_states: dict类型, key是准备好的agent的agent_id, value有两种情况:
        1. State的子类, 下一个状态s_t+1
        2. dict类型, key是准备好的policy_id, value是State的子类
        :return: ex_rewards: dict类型, key是准备好的agent的agent_id, value是float数组, 包含了从环境中返回的各种reward信号, r_t
        当以上返回字典为空字典时，认为当前帧是无效帧，返回空动作给客户端
        :raise: RestartException标识回给客户端一个Restart响应
        """
        raise NotImplementedError

    def on_update_rsp(self, actions, extra_info=None):
        """
        在返回actions给客户端之前被调用
        :param actions: key是准备好的agent_id, value是Action的子类
        :param extra_info: key是准备好的agent_id, value是dict类型, 其中包含Network所有输出tensor的值
        :return: rsp_data: dict类型, key是int类型的agent_id, value是action编码之后的bytes
        """
        raise NotImplementedError

    def on_agent_end(self, client_id, ep_id, agent_id, req_data):
        """
        标识一个agent的结束, 可以用来完成agent的析构工作
        :param client_id: str类型, 用于唯一标识客户端
        :param ep_id: int类型, 用于唯一标识一个episode
        :param agent_id: int类型, 用于唯一标识一个agent
        :param req_data: byte数组, 业务自定义数据
        """

    def on_ep_end(self, client_id, ep_id, req_data):
        """
        标识一个episode的结束, 一般在所有on_agent_end调用之后被调用, 可以用来完成episode的清理工作
        :param client_id: str类型, 用于唯一标识客户端
        :param ep_id: int类型, 用于唯一标识一个episode
        :param req_data: byte数组, 业务自定义数据
        """

    def on_event(self, client_id, req_data):
        """
        表示一个event事件的发生
        :param client_id: str类型, 用于唯一标识客户端
        :param req_data: byte数组, 业务自定义数据
        :return: rsp_data: byte数组, 业务自定义数据
        """
        return b''

    def on_quit(self, client_id):
        """
        标识一个客户端断开连接, 一般在收到Quit包之后被调用, 但在异常断联的情况也会被调用
        可用来完成资源释放的工作
        :param client_id: str类型, 用于唯一标识客户端
        """