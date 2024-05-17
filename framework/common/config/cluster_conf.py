#!/usr/bin/env python
# -*- coding: utf-8 -*-


class DomainNameParser:
    '''
    需要和job_master确认下发的配置learner.toml, actor.toml内容
    '''
    def __init__(self) -> None:
        pass


class ClusterConf:
    def __init__(self,
                 learner_ip_addrs,
                 actor_ip_addrs,
                 learner_grpc_ports,
                 actor_grpc_ports,
                 svr_name,
                 svr_index,
                 svr_ports,
                 local_rank):
        """集群配置信息, TensorFlow需要使用, 注意actor和learner的配置是不一样的
        Arguments:
            learner_ip_addrs {list} -- 所有learner节点的ip地址
            actor_ip_addrs {list} -- 所有actor节点的ip地址
            learner_grpc_ports {list} -- 对于每一个learner节点, 其中的一张GPU卡, 对应一个任务, 对应一个grpc_port, 用于和分配到的actor(可能没有)组成小的cluster
            actor_grpc_ports {list} -- 每一个actor节点, 只有一张GPU卡, 对应一个任务, 对应一个grpc_port, 用于和分配到的learner任务组成小的cluster
            svr_name {str} -- server name, 一般为learner、actor
            svr_index {int} -- 即为pod index, 该节点在同一组容器中的index
            svr_ports {list} -- 用于和aisrv连接的端口, 节点内每一个任务对应一个端口
            local_rank {int} -- 该任务在节点中的local rank, 也就是第几个任务
        """
        # 赋值成员变量
        self.learner_ip_addrs = learner_ip_addrs
        self.actor_ip_addrs = actor_ip_addrs
        self.learner_grpc_ports = learner_grpc_ports
        self.actor_grpc_ports = actor_grpc_ports
        assert svr_name in ['actor', 'learner']
        self.svr_name = svr_name
        assert svr_index < len(getattr(self, "%s_ip_addrs" % svr_name))
        self.svr_index = svr_index
        self.svr_ports = svr_ports
        self.local_rank = local_rank

        # 计数
        num_learner_nodes = len(learner_ip_addrs)
        num_tasks_per_learner = len(learner_grpc_ports)
        num_learner_tasks = num_learner_nodes * num_tasks_per_learner
        num_actor_nodes = len(actor_ip_addrs)
        num_tasks_per_actor = len(actor_grpc_ports)
        num_actor_tasks = num_actor_nodes * num_tasks_per_actor

        # 排序
        num_tasks_per_svr = num_tasks_per_learner if self.svr_name == 'learner' else num_tasks_per_actor
        self.world_rank = num_tasks_per_svr * self.svr_index + self.local_rank

        # 分配：learner不一定有对应的actor，但是actor一定且只有一个对应actor
        # learner node -> actor world rank
        node2actors_map = {node_i: [actor_wr for actor_wr in range(num_actor_tasks) \
                                    if actor_wr % num_learner_nodes == node_i] \
                           for node_i in range(num_learner_nodes)}
        # actor world rank -> actor svr index, actor local rank
        wr2actor_map = {actor_wr: (int(actor_wr // num_tasks_per_actor), actor_wr % num_tasks_per_actor)
                        for actor_wr in range(num_actor_tasks)}
        # learner's svr index and local rank -> [actors' svr index and local rank]
        self.learner2actors_maps = [{task_i: [wr2actor_map[actor_wr] \
                                              for local_i, actor_wr in enumerate(node2actors_map[node_i]) \
                                              if local_i % num_tasks_per_learner == task_i] \
                                     for task_i in range(num_tasks_per_learner)} \
                                    for node_i in range(num_learner_nodes)]
        # actor's world rank -> learner's svr_index and local_rank, actor rank in cluster.
        self.actor2leaner_map = dict()
        for actor_wr in range(num_actor_tasks):
            node_i = actor_wr % num_learner_nodes
            actors4nodei = node2actors_map[node_i]
            local_i = actors4nodei.index(actor_wr)
            card_i = local_i % num_tasks_per_learner
            actor_rank = int(local_i // num_tasks_per_learner)
            self.actor2leaner_map[actor_wr] = (node_i, card_i, actor_rank)

    @property
    def job_name(self):
        """
        定义tf.distribute.Server中的job_name, 仅可为learner, 或者actor
        return:
            job_name {str}
        """
        return self.svr_name

    @property
    def task_index(self):
        """
        定义tf.distribute.Server中的task_index, 仅可为learner, 或者actor
        return:
            task_index {int} -- 在此小的cluster中, 该任务的index
        """
        if self.svr_name == 'learner':
            return 0
        else:
            return self.actor2leaner_map[self.world_rank][2]

    @property
    def ip_address(self):
        if self.svr_name == 'learner':
            return self.learner_ip_addrs[self.svr_index]
        else:
            return self.actor_ip_addrs[self.svr_index]

    @property
    def svr_port(self):
        """
        return:
            svr_port {int} -- 该任务对应的svr_port
        """
        return self.svr_ports[self.local_rank]

    @property
    def grpc_port(self):
        """
        return:
            grpc_port {int} -- 该任务对应的grpc_port
        """
        if self.svr_name == 'learner':
            return self.learner_grpc_ports[self.local_rank]
        else:
            return self.actor_grpc_ports[self.local_rank]

    @property
    def learner_local_rank(self):
        assert self.svr_name == 'actor'
        node_i, card_i, _ = self.actor2leaner_map[self.world_rank]
        return card_i
    
    def cluster_spec(self):
        """
        return:
            cluster_spec {dict} -- key: learner和actor, value: list of ip address with grpc_port
        """
        if self.svr_name == 'learner':
            l_addr = f'{self.ip_address}:{self.grpc_port}'
            actors = self.learner2actors_maps[self.svr_index][self.local_rank]
            a_addrs = [f'{self.actor_ip_addrs[ai]}:{self.actor_grpc_ports[ar]}' for ai, ar in actors]
        else:
            node_i, card_i, _ = self.actor2leaner_map[self.world_rank]
            l_addr = f'{self.learner_ip_addrs[node_i]}:{self.learner_grpc_ports[card_i]}'
            actors = self.learner2actors_maps[node_i][card_i]
            a_addrs = [f'{self.actor_ip_addrs[ai]}:{self.actor_grpc_ports[ar]}' for ai, ar in actors]

        cluster_spec = {
            'learner': [l_addr, ]
        }

        if a_addrs:
            cluster_spec['actor'] = a_addrs

        return cluster_spec