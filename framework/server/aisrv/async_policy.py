#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from framework.common.config.config_control import CONFIG
from framework.interface.policy import Policy, PolicyBuilder
from framework.common.utils.common_func import str_to_addr
from framework.server.aisrv.actor_proxy import ActorProxy
from framework.server.aisrv.learner_proxy import LearnerProxy
from framework.server.aisrv.sample_server import SampleServer
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine

class AsyncPolicy(Policy):
    def __init__(self, policy_name, actor_proxy_list, learner_proxy_list, sample_sever_list):
        self.policy_name = policy_name

        self._exit_flag = False

        # 与该aisrv相连的actor列表
        self._actor_proxy_list = actor_proxy_list
        # 与该aisrv相连的learner列表
        self._learner_proxy_list = learner_proxy_list
        # aisrv的sample_server列表
        self._sample_sever_list = sample_sever_list

    def identity(self, client_conn_id, agent_id):
        return "AsyncPolicy(id=%s, client_conn_id: %s, agent_id: %d)" % (self.policy_name, client_conn_id, agent_id)

    '''
    如果configure.json的配置项run_mode是train则需要训练
    如果configure.json的配置项run_mode是eval则不需要训练
    '''
    def need_train(self):
        return CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN

    def stop(self):
        self._exit_flag = True

    # aisrv从actor获取预测响应
    def get_pred_result(self, slot_id, agent_ctx):
        agent_id = agent_ctx.agent_id
        actor_index = slot_id % len(self._actor_proxy_list)
        
        # 异步场景下需要多次尝试, 获取响应结果
        retry_num = 0
        while retry_num < int(CONFIG.socket_retry_times) and not self._exit_flag:
            result_map = self._actor_proxy_list[actor_index].get_predict_data(slot_id)
            if result_map:
                return result_map
            retry_num += 1
        if retry_num >= int(CONFIG.socket_retry_times) or self._exit_flag:
            raise RuntimeError(f'agent {agent_id}  slot_id {slot_id} failed to get data from actor proxy {self._actor_proxy_list[actor_index].get_zmq_server_ip()}')

    # aisrv朝actor发送预测请求
    def send_pred_data(self, slot_id, pred_data, agent_ctx):
        agent_id = agent_ctx.agent_id
        actor_index = slot_id % len(self._actor_proxy_list)

        success = False
        success = self._actor_proxy_list[actor_index].put_predict_data(slot_id, agent_id, agent_ctx.message_id, agent_ctx.model_version, pred_data)
        if success:
            agent_ctx.message_id += 1
        
        # 失败时将对应的actor地址打印, 便于对账处理
        return success, self._actor_proxy_list[actor_index].get_zmq_server_ip()
    
    # aisrv朝actor发送时是将同一帧的两个样本拼凑到一起再发送，使用于self-play且两个agent同一个policy的情况
    def send_pred_data_v2(self, slot_id, pred_data_0, pred_data_1, agent_ctx_0, agent_ctx_1):
        agent_id_0 = agent_ctx_0.agent_id
        agent_id_1 = agent_ctx_1.agent_id
        actor_index = slot_id % len(self._actor_proxy_list)

        success = False
        success = self._actor_proxy_list[actor_index].put_predict_data_v2(slot_id,
                                                                          agent_id_0,
                                                                          agent_id_1,
                                                                          agent_ctx_0.message_id,
                                                                          agent_ctx_1.message_id,
                                                                          pred_data_0,
                                                                          pred_data_1)
        if success:
            agent_ctx_0.message_id += 1
            agent_ctx_1.message_id += 1
        
        # 失败时将对应的actor地址打印, 便于对账处理
        return success, self._actor_proxy_list[actor_index].get_zmq_server_ip()

    # 朝learner发送训练数据
    def send_train_data(self, train_data, train_data_prioritezeds, agent_ctx):
        agent_id = agent_ctx.agent_id
        learn_index = agent_id % len(self._learner_proxy_list)

        self._learner_proxy_list[learn_index].put_data(agent_id, train_data, train_data_prioritezeds)
    
    # sample_server-->learener_proxy-->learner
    def gen_frame_sample(self, slot_id, sample_info_list, must_need_sample_info):

        sample_index = slot_id % len(self._sample_sever_list)
        self._sample_sever_list[sample_index].gen_frame_sample(slot_id, sample_info_list, must_need_sample_info)
        
    def sample_server_gameover(self, slot_id):

        sample_index = slot_id % len(self._sample_sever_list)
        return self._sample_sever_list[sample_index].sample_server_gameover(slot_id)
    
    def add_policy_to_sample_server(self, slot_id, main_id):

        sample_index = slot_id % len(self._sample_sever_list)
        self._sample_sever_list[sample_index].add_policy_to_sample_server(slot_id, main_id)

    '''
    actor_proxy_list 增加一项
    '''
    def add_actor_proxy_list(self, actor_proxy):
        if not actor_proxy:
            return

        self._actor_proxy_list.append(actor_proxy)

    '''
    actor_proxy_list 减少一项, 因为list下的元素较少, 采用遍历即可
    '''
    def reduce_actor_proxy_list(self, actor_ip):
        if not actor_ip:
            return
        
        for actor_proxy in self._actor_proxy_list:
            if actor_proxy.get_zmq_server_ip() == actor_ip:
                actor_proxy.stop()

    '''
    learner_proxy_list 增加一项
    '''
    def add_learner_proxy_list(self, learner_proxy):
        if not learner_proxy:
            return

        self._learner_proxy_list.append(learner_proxy)

    '''
    learner_proxy_list 减少一项, 因为list下的元素较少, 采用遍历即可
    '''
    def reduce_learner_proxy_list(self, learner_ip):
        if not learner_ip:
            return
        
        for learner_proxy in self._learner_proxy_list:
            if learner_proxy.get_reverb_ip() == learner_ip:
                learner_proxy.stop()
    
    def get_actor_proxy_cnt(self):
        return len(self._actor_proxy_list)
    
    def get_learner_proxy_cnt(self):
        return len(self._learner_proxy_list)

    def get_sample_server_cnt(self):
        return len(self._sample_sever_list)
    
    def get_current_actor_learner_prxoy_list(self):
        actor_addrss_list = []
        for i in range(len(self._actor_proxy_list)):
            actor_addrss_list.append(self._actor_proxy_list[i].get_zmq_server_ip())
        
        learner_addrss_list = []
        for i in range(len(self._learner_proxy_list)):
            learner_addrss_list.append(self._learner_proxy_list[i].get_reverb_ip())

        return actor_addrss_list, learner_addrss_list

class AsyncBuilder(PolicyBuilder):
    def __init__(self, policy_name, simu_ctx):
        super().__init__(policy_name, simu_ctx)

        # 注意配置项actor_addrs、learner_addrs是采用yaml格式的配置
        actor_addrs = CONFIG.actor_addrs[policy_name]
        actor_proxy_num = CONFIG.actor_proxy_num
        actor_proxy_num =len(actor_addrs)
        self.actor_proxy_list = [None] * actor_proxy_num
        
        for i in range(0, actor_proxy_num):
            actor_addr = str_to_addr(actor_addrs[i])
            self.actor_proxy_list[i] = ActorProxy(policy_name, i, actor_addr, simu_ctx)
            self.actor_proxy_list[i].start()

        learner_addrs = CONFIG.learner_addrs[policy_name]
        learner_proxy_num = CONFIG.learner_proxy_num
        learner_proxy_num = len(learner_addrs)
        self.learner_proxy_list = [None] * learner_proxy_num
        #sample_server暂不支持多个
        assert CONFIG.sample_server_count==1
        self.sample_sever_list = [None] * CONFIG.sample_server_count
        
        #在self-play 且model为旧模型时不需要配置learner
        if not (int(CONFIG.self_play) and policy_name == CONFIG.self_play_old_policy):
            for i in range(0, learner_proxy_num):
                learner_addr = str_to_addr(learner_addrs[i])
                self.learner_proxy_list[i] = LearnerProxy(policy_name, learner_addr, simu_ctx)
                self.learner_proxy_list[i].start()
        
            
            if CONFIG.use_sample_server:
                for i in range(0, CONFIG.sample_server_count):
                    self.sample_sever_list [i] = SampleServer(self.learner_proxy_list)
                    self.sample_sever_list[i].start()

        # 获取到AsyncPolicy对象, 用于对actor, learner的扩缩容操作
        self.async_policy = None
        self.policy_name = policy_name
        self.simu_ctx = simu_ctx

    def build(self):
        self.async_policy = AsyncPolicy(self._policy_name, self.actor_proxy_list, self.learner_proxy_list, self.sample_sever_list)

        return self.async_policy
    
    '''
    下面是对运行中的任务进行actor和learner扩缩容操作, 主要包括:
    1. 新增actor
    2. 缩容actor
    3. 新增learner
    4. 缩容actor

    注意:
    1. add的操作是新增multiprocessing.Process
    2. reduce的操作是减少特定IP对应的multiprocessing.Process
    '''
    def add_actor_proxy_list(self, actor_proxy):
        if not actor_proxy:
            return
        
        actor_current_max_idx = self.async_policy.get_actor_proxy_cnt()
        actor_proxy = ActorProxy(self.policy_name, actor_current_max_idx, actor_proxy, self.simu_ctx)
        actor_proxy.start()

        return self.async_policy.add_actor_proxy_list(actor_proxy)
    
    def reduce_actor_proxy_list(self, actor_ip):
        if not actor_ip:
            return
        
        return self.async_policy.reduce_actor_proxy_list(actor_ip)
    
    def add_learner_proxy_list(self, learner_proxy):
        if not learner_proxy:
            return
        
        learner_proxy = LearnerProxy(self.policy_name, learner_proxy, self.simu_ctx)
        learner_proxy.start()

        return self.async_policy.add_learner_proxy_list(learner_proxy)
    
    def reduce_learner_proxy_list(self, learner_ip):
        if not learner_ip:
            return

        return self.async_policy.reduce_learner_proxy_list(learner_ip)
    
