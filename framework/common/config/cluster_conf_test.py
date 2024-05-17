#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from framework.common.config.cluster_conf import ClusterConf
from framework.common.config.config_control import CONFIG

class TestClusterConf(unittest.TestCase):
    def setUp(self):
        learner_ip_addrs = ["localhost", "localhost"]
        actor_ip_addrs = ["localhost", "localhost", "localhost", "localhost"]

        learner_grpc_ports = [8001]
        actor_grpc_ports = [8002]

        self.learner_cluster_conf = ClusterConf(
            learner_ip_addrs, actor_ip_addrs,
            learner_grpc_ports, actor_grpc_ports,
            'learner', 0,
            [9001], 0
        )

        self.actor_cluster_conf_0 = ClusterConf(
            learner_ip_addrs, actor_ip_addrs,
            learner_grpc_ports, actor_grpc_ports,
            'actor', 0,
            [9003], 0
        )
        self.actor_cluster_conf_3 = ClusterConf(
            learner_ip_addrs, actor_ip_addrs,
            learner_grpc_ports, actor_grpc_ports,
            'actor', 3,
            [9005], 0
        )

    def test_get_job_name(self):
        self.assertEqual(self.learner_cluster_conf.job_name, "learner")
        self.assertEqual(self.actor_cluster_conf_0.job_name, "actor")
        self.assertEqual(self.actor_cluster_conf_3.job_name, "actor")

    def test_get_task_index(self):
        self.assertEqual(self.learner_cluster_conf.task_index, 0)
        self.assertEqual(self.actor_cluster_conf_0.task_index, 0)
        self.assertEqual(self.actor_cluster_conf_3.task_index, 1)

    def test_cluster_spec(self):
        self.assertEqual(self.learner_cluster_conf.cluster_spec(), {
            'learner': ['localhost:8001'], 'actor': ['localhost:8002', 'localhost:8002']
        })
        self.assertEqual(self.actor_cluster_conf_0.cluster_spec(), {
            'learner': ['localhost:8001'], 'actor': ['localhost:8002', 'localhost:8002']
        })
        self.assertEqual(self.actor_cluster_conf_3.cluster_spec(), {
            'learner': ['localhost:8001'], 'actor': ['localhost:8002', 'localhost:8002']
        })

    def test_get_ips(self):
        learner_ip_addrs = "localhost,localhost"
        actor_ip_addrs = "localhost,localhost,localhost,localhost"
        learner_grpc_ports = 8001
        actor_grpc_ports =  8002

        print(learner_ip_addrs.split(','))
        print(actor_ip_addrs.split(','))
    
    def test_actor_cluster_conf(self):

        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/actor.toml")
        CONFIG.parse_actor_configure()

        cluster_conf = ClusterConf(
        CONFIG.learner_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.actor_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.learner_grpc_ports.split(','),
        CONFIG.actor_grpc_ports.split(','),
        # 下面配置项目每个进程启动时, 进程配置文件加载
        CONFIG.svr_name, 
        CONFIG.svr_index, 
        CONFIG.svr_ports,
        CONFIG.local_rank
        )

        print('actor: ' + str(cluster_conf.cluster_spec()))
    
    def test_learner_cluster_conf(self):
        
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/learner.toml")
        CONFIG.parse_learner_configure()

        cluster_conf = ClusterConf(
        CONFIG.learner_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.actor_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.learner_grpc_ports.split(','),
        CONFIG.actor_grpc_ports.split(','),
        # 下面配置项目每个进程启动时, 进程配置文件加载
        CONFIG.svr_name, 
        CONFIG.svr_index, 
        CONFIG.svr_ports,
        CONFIG.local_rank
        )

        print('learner: ' + str(cluster_conf.cluster_spec()))

if __name__ == '__main__':
    unittest.main()
