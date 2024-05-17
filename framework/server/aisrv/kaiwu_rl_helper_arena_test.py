#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import multiprocessing
import time
import unittest
from unittest import mock

from framework.common.utils.common_func import Context
from framework.interface.agent_context import AgentContext
from framework.interface.exception import TimeoutEpisodeException
from framework.server.aisrv.kaiwu_environ_test import MockPolicyBuilder, MockPolicyConf, \
    MockState, MockRunHandler
from framework.server.aisrv.kaiwu_rl_helper_arena import KaiWuRLHelper
from framework.common.config.app_conf import AppConf
from framework.common.config.config_control import CONFIG
from framework.server.aisrv.kaiwu_environ_test import MockAppConf
from framework.common.utils.slots import Slots

class MockKaiwuEnviron:
    def __init__(self, simu_ctx, exit_flag, client_conn_id):
        pass

    def initialize(self):
        pass

    def reject(self, e):
        pass

    @property
    def client_id(self):
        return 12345

    @property
    def ep_id(self):
        return 1

    def reset(self):
        pass          

class TestKaiWuRLHelper(unittest.TestCase):
    def setUp(self) -> None:
        # 全局参数
        self.algo = "ppo"
        self.app = "gym"
        self.fake_agent_id_0 = 0
        self.fake_agent_id_1 = 1
        self.fake_policy_name_one = "train_one"
        self.fake_policy_name_two = "train_two"

        AppConf.load_conf("/data/projects/kaiwu-fwk/conf/app_conf.json")

        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        # mock kaiwu_rl_helper中的import module
        args = {
            'run_mode': 'train',
            'app': self.app,
            'learner_comm_mode': "",  # TODO：暂时不走 if CONFIG.learner_comm_mode == CommMode.Reverb 的逻辑
            'job_master_addr': "127.0.0.1:8080",
            'svr_index': 0,
            'svr_port': 1000,
            'episode_timeout_s': 0.0001,  # 确保能跳出episode_main_loop的while循环
        }
        KaiWuRLHelper.KaiwuEnviron = MockKaiwuEnviron
        # mock AppConf
        #   AppConf加载了app_conf.json中所有的policy conf的相关信息
        #   但实际使用还是和用户实现的policy_mapping_fn有关
        #   1. create instance of MockTupleType
        mock_tuple_type = MockAppConf.MockTupleType()
        #   2. mock instance
        #       2.1 首先是获取用户配置的policies
        fake_one_app_conf = {
            self.app: {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "policies": {
                    self.fake_policy_name_one: {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    },
                    self.fake_policy_name_two: {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                },
                "builder": "app.metal.metal_builder.MetalBuilder"
            }
        }
        #       2.2 根据其中的run_handler设置tuple_type的run_handler, 这里使用mock，而不是locate到真正的GymRunHandler
        mock_tuple_type.run_handler = MockRunHandler
        #       2.4 根据其中的policies设置tuple_type的policies,
        #           tuple_type的policies将每个policy映射到对于的policy_conf, 这里使用mock
        policies_dict = fake_one_app_conf[self.app]["policies"]
        mock_tuple_type.policies = {name: MockPolicyConf(**policy) for name, policy in policies_dict.items()}
        #   3. 构造app和tuple_type之前的映射，传入MockAppConf
        mock_app_conf_args = {
            self.app: mock_tuple_type
        }
        KaiWuRLHelper.AppConf = MockAppConf(mock_app_conf_args)

        # 创建fake kaiwu_rl_helper实例
        fake_exit_flag = multiprocessing.Value('b', False)
        fake_client_conn_id = 12345

        #   这里加载了app_conf.json中所有的policy(policy conf的相关信息呗加载在AppConf)，
        #   但实际使用还是和用户实现的policy_mapping_fn有关
        fake_parent_simu_ctx = Context()
        fake_policy_builders = {
            self.fake_policy_name_one: MockPolicyBuilder(policy_name=self.fake_policy_name_one,
                                                         simu_ctx=fake_parent_simu_ctx),
            self.fake_policy_name_two: MockPolicyBuilder(policy_name=self.fake_policy_name_two,
                                                         simu_ctx=fake_parent_simu_ctx)
        }
        fake_parent_simu_ctx.policy_builders = fake_policy_builders
        fake_parent_simu_ctx.exit_flag = fake_exit_flag
        fake_parent_simu_ctx.client_conn_id = fake_client_conn_id
        fake_parent_simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))


        policies_builder = {}
        policies_conf = AppConf[CONFIG.app].policies
        for policy_name, policy_conf in policies_conf.items():
                policies_builder[policy_name] = policy_conf.policy_builder(policy_name, fake_parent_simu_ctx)
        fake_parent_simu_ctx.policies_builder = policies_builder

        self.kaiwu_rl_helper = KaiWuRLHelper(parent_simu_ctx=fake_parent_simu_ctx)

    # mock self.env 的 policy_mapping_fn函数
    def mock_policy_mapping_fn(self, agent_id: int):
        if agent_id == self.fake_agent_id_0:
            return self.fake_policy_name_one
        elif agent_id == self.fake_agent_id_1:
            return self.fake_policy_name_two
        else:
            raise ValueError("illegal agent_id")

    def _build_fake_agent_ctxs(self):
        # mock AppConf[CONFIG.app].policies
        fake_one_app_conf = {
            self.app: {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "policies": {
                    self.fake_policy_name_one: {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    },
                    self.fake_policy_name_two: {
                        "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                },
                "builder": "app.metal.metal_builder.MetalBuilder"
            }
        }
        policies_dict = fake_one_app_conf["gym"]["policies"]
        mock_app_conf_policies = {name: MockPolicyConf(**policy) for name, policy in policies_dict.items()}

        # 为每个agent构建agent_ctx
        policy_ids = self.kaiwu_rl_helper.normalize_policy_ids(self.mock_policy_mapping_fn(self.fake_agent_id_0))
        fake_agent_ctx_0 = AgentContext()
        fake_agent_ctx_0.main_id = policy_ids[0]
        fake_agent_ctx_0.agent_id = self.fake_agent_id_0
        fake_agent_ctx_0.policy = {}
        fake_agent_ctx_0.policy_conf = {}
        fake_agent_ctx_0.pred_output = {}
        fake_agent_ctx_0.expr_processor = {}
        fake_agent_ctx_0.start_time = time.monotonic()

        for policy_id in policy_ids:
            policy_conf = mock_app_conf_policies[policy_id]
            policy = self.kaiwu_rl_helper.policies[policy_id]
            fake_agent_ctx_0.policy[policy_id] = policy
            fake_agent_ctx_0.policy_conf[policy_id] = policy_conf

        # 为每个agent构建agent_ctx
        policy_ids = self.kaiwu_rl_helper.normalize_policy_ids(self.mock_policy_mapping_fn(self.fake_agent_id_1))
        fake_agent_ctx_1 = AgentContext()
        fake_agent_ctx_1.main_id = policy_ids[0]
        fake_agent_ctx_1.agent_id = self.fake_agent_id_1
        fake_agent_ctx_1.policy = {}
        fake_agent_ctx_1.policy_conf = {}
        fake_agent_ctx_1.pred_output = {}
        fake_agent_ctx_1.expr_processor = {}
        fake_agent_ctx_1.start_time = time.monotonic()

        for policy_id in policy_ids:
            policy_conf = mock_app_conf_policies[policy_id]
            policy = self.kaiwu_rl_helper.policies[policy_id]
            fake_agent_ctx_1.policy[policy_id] = policy
            fake_agent_ctx_1.policy_conf[policy_id] = policy_conf

        # 放入agent_ctxs中
        self.kaiwu_rl_helper.agent_ctxs[self.fake_agent_id_0] = fake_agent_ctx_0
        self.kaiwu_rl_helper.agent_ctxs[self.fake_agent_id_1] = fake_agent_ctx_1

        # build state and pred_input
        # mock parameter states of def episode_main_loop(self, states)
        fake_states = {
            self.fake_agent_id_0: {self.fake_policy_name_one: MockState()},
            self.fake_agent_id_1: {self.fake_policy_name_two: MockState()}
        }
        for fake_agent_id in [self.fake_agent_id_0, self.fake_agent_id_1]:
            fake_agent_ctx = self.kaiwu_rl_helper.agent_ctxs[fake_agent_id]
            fake_agent_ctx.state, fake_agent_ctx.pred_input = {}, {}
            for policy_id, state in fake_states[fake_agent_id].items():
                s = state.get_state()
                fake_agent_ctx.pred_input[policy_id] = s
                fake_agent_ctx.state[policy_id] = state

    def test_identity(self):
        identity = self.kaiwu_rl_helper.identity
        self.assertTrue(isinstance(identity, str))
        self.assertTrue(len(identity) > 0)
        self.assertTrue(len(self.kaiwu_rl_helper.simu_ctx.policies) == 2)

    """
    KaiWuRLHelper(self, agent_ids)函数单测
    
    背景：
    每个agent都有自己特定的一个或多个policy
    policy中定义了预测数据发送函数(send_pred_data)和预测结果接受函数(get_pred_result)
    该函数遍历传入的agent, 并调用相应的policy定义的上述两个函数
    """

    def test_predict(self):
        self._build_fake_agent_ctxs()

        self.kaiwu_rl_helper.predict(agent_ids=[self.fake_agent_id_0, self.fake_agent_id_1])

        self.assertTrue(len(self.kaiwu_rl_helper.agent_ctxs) == 2)

        for __, agent_ctx in self.kaiwu_rl_helper.agent_ctxs.items():
            self.assertTrue(isinstance(agent_ctx, AgentContext))
            self.assertTrue(len(agent_ctx.pred_output) == 1)
            self.assertTrue(len(agent_ctx.pred_input) == 1)
            self.assertTrue(len(agent_ctx.state) == 1)

        # reset
        self.kaiwu_rl_helper.agent_ctxs = {}

    """
    KaiWuRLHelper的episode_main_loop函数单测
    
    背景:
    该函数首先根据传入的states(基于第一个update_req)来计算出predict data
    然后调用predict函数进行预测并获取预测结果(发送给actor并接收结果)
    然后根据预测结果计算出action dict 和 extra_info dict
    然后调用step函数将action返回给客户端, 并接收下一个update_req, 并获取新的states和ex_rewards等
    然后根据ex_rewards等信息计算出reward
    然后调用gen_train_data函数生产训练数据并将训练数据出发给learner
    最后更新states, 开始下一轮循环, 或结束循环
    """

    def test_episode_main_loop(self):
        # mock input
        fake_states = {
            self.fake_agent_id_0: {self.fake_policy_name_one: MockState()},
            self.fake_agent_id_1: {self.fake_policy_name_two: MockState()}
        }

        # mock self.env 的 step函数
        fake_new_states = {
            self.fake_agent_id_0: {self.fake_policy_name_one: MockState()},
            self.fake_agent_id_1: {self.fake_policy_name_two: MockState()}
        }
        fake_ex_rewards = {agent_id: [] for agent_id in fake_new_states}
        fake_dones = {agent_id: False for agent_id in fake_new_states}
        fake_dones['_all_done_'] = False
        self.kaiwu_rl_helper.env.step = mock.Mock(return_value=(fake_new_states, fake_ex_rewards, fake_dones))

        # mock self.env 的 policy_mapping_fn函数
        self.kaiwu_rl_helper.env.policy_mapping_fn = mock.Mock(side_effect=self.mock_policy_mapping_fn)

        self.assertRaises(TimeoutEpisodeException, self.kaiwu_rl_helper.episode_main_loop, fake_states)

    """
    KaiWuRLHelper的run_episode函数单测
    
    背景：
    该函数首先调用KaiwuEnviron中的reset函数确保收到客户端发来的ep_start_req, agent_start_req和第一个有效的update_req请求
    然后调用episode_main_loop对于对局中的多次update_req进行处理
    """

    @mock.patch("framework.server.aisrv.kaiwu_rl_helper.KaiWuRLHelpe.episode_main_loop")
    def test_run_episode(self, mock_episode_main_loop):
        self.kaiwu_rl_helper.run_episode()

        mock_episode_main_loop.assert_called_once()

    """
    KaiWuRLHelper的run(self) -> None函数单测
    
    背景：
    该函数首先调用KaiwuEnviron中的init函数确保收到客户端发来的init_req请求
    然后不断的调用run_episode函数开始模拟多次对局
    最后调用stop()函数退出
    """

    @mock.patch("framework.server.aisrv.kaiwu_rl_helper.KaiWuRLHelper.stop")
    def test_run(self, mock_stop):
        def side_effect_fn(fn):
            # break while loop
            self.kaiwu_rl_helper.exit_flag.value = True

        self.kaiwu_rl_helper._tm_profile = mock.Mock(side_effect=side_effect_fn)
        self.kaiwu_rl_helper.run()
        mock_stop.assert_called_once()

    """
    KaiWuRLHelper的start_agent(self, agent_id)函数单测
    KaiWuRLHelper的stop_agent(self, agent_id)函数单测

    背景：
    start_agent函数给每个agent的新建上下文agent_ctx = AgentContext()
    并对self.agent_registry, self.agent_ctxs, self.agent_stats_dict这三个变量初始化
    
    stop_agent函数对agent进行收尾工作, 包括
    记录结束时间, 运行长度, 并调用上下文中的收尾函数, 最后汇报统计量, 并清空self.agent_ctxs, self.agent_stats_dict
    """

    def test_start_stop_agent(self):

        self.kaiwu_rl_helper.env.policy_mapping_fn = mock.Mock(side_effect=self.mock_policy_mapping_fn)

        self.kaiwu_rl_helper.start_agent(agent_id=self.fake_agent_id_0)
        
        self.kaiwu_rl_helper.start_agent(agent_id=self.fake_agent_id_1)

        self.assertTrue(len(self.kaiwu_rl_helper.agent_ctxs) == 2)

        self.kaiwu_rl_helper.stop_agent(agent_id=self.fake_agent_id_0)
        self.kaiwu_rl_helper.stop_agent(agent_id=self.fake_agent_id_1)

        self.assertTrue(len(self.kaiwu_rl_helper.agent_ctxs) == 0)

        # reset
        self.kaiwu_rl_helper.agent_ctxs = {}

    """
    KaiWuRLHelper的gen_train_data(self, agent_id, policy_id)函数单测
    
    背景：
    该函数首先根据传入的agent_id和policy_id获取当前使用的expr_processor
    然后调用expr_processor相关函数来生成train_data, 并将train_data通过用户自定的policy中的send_train_data发送出去
    最后更新相关统计量
    """

    def test_gen_train_data(self):
        self.kaiwu_rl_helper.env.policy_mapping_fn = mock.Mock(side_effect=self.mock_policy_mapping_fn)

        # start agent first
        self.kaiwu_rl_helper.start_agent(agent_id=self.fake_agent_id_0)
        self.kaiwu_rl_helper.start_agent(agent_id=self.fake_agent_id_1)

        # get train data second
        self.kaiwu_rl_helper.gen_train_data(agent_id=self.fake_agent_id_0, policy_id=self.fake_policy_name_one)
        self.kaiwu_rl_helper.gen_train_data(agent_id=self.fake_agent_id_1, policy_id=self.fake_policy_name_two)

        # reset
        self.kaiwu_rl_helper.agent_ctxs = {}

if __name__ == "__main__":
    unittest.main()
