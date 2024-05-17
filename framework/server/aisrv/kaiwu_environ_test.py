#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import flatbuffers
from functools import wraps
import multiprocessing
import time
import unittest
from unittest import mock

from framework.common.utils.common_func import Context
from framework.interface.action import Action
from framework.interface.exception import ClientQuitException, SkipEpisodeException
from framework.interface.policy import Policy, PolicyBuilder
from framework.interface.network import Network
from framework.interface.reward import Reward
from framework.interface.reward_shaper import RewardShaper
from framework.interface.run_handler import RunHandler
from framework.interface.state import State
from framework.server.aisrv.flatbuffer.kaiwu_msg import *
from framework.server.aisrv.kaiwu_environ import KaiwuEnviron
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from framework.server.aisrv import kaiwu_environ

class MockRunHandler(RunHandler):
    def __init__(self, simu_ctx):
        super(MockRunHandler, self).__init__(simu_ctx)

    def on_quit(self, client_id):
        pass

    def on_init(self, client_id, req_data):
        pass

    def policy_mapping_fn(self, agent_id):
        return 'train'

    def on_ep_start(self, client_id, ep_id, req_data):
        pass

    def on_agent_start(self, client_id, ep_id, agent_id, req_data):
        pass

    def on_update_req(self, client_id, ep_id, req_data):
        fake_states = {0: MockState(), 1: MockState()}
        fake_ex_rewards = {0: [], 1: []}
        return fake_states, fake_ex_rewards

    def on_update_rsp(self, actions, extra_info=None):
        return {
            0: b'Hello World!',
            1: b'Hello World!'
        }

    def on_agent_end(self, client_id, ep_id, agent_id, req_data):
        pass

    def on_ep_end(self, client_id, ep_id, req_data):
        pass

    def on_event(self, client_id, req_data):
        return b'Hello World!'


class MockPolicy(Policy):
    def __init__(self, policy_name):
        super().__init__()
        self.policy_name = policy_name

    def send_pred_data(self, client_conn_id, pred_data, agent_ctx):
        pass

    def get_pred_result(self, client_conn_id, agent_ctx):
        return ''

    def need_train(self):
        return True

    def send_train_data(self, client_conn_id, train_data, agent_ctx):
        pass

    def stop(self):
        pass


class MockPolicyBuilder(PolicyBuilder):
    def __init__(self, policy_name, simu_ctx):
        super(MockPolicyBuilder, self).__init__(policy_name, simu_ctx)

    def build(self):
        return MockPolicy(self._policy_name)


class MockState(State):
    def __init__(self):
        super(MockState, self).__init__()

    def get_state(self):
        return {}

    @staticmethod
    def state_space():
        return {}

    def __str__(self):
        return str({})


class MockAction(Action):
    def get_action(self):
        return {}

    @staticmethod
    def action_space():
        return {}

    def __str__(self):
        return str({})


class MockReward(Reward):
    def __init__(self):
        super(MockReward, self).__init__()


class MockNetwork(Network):
    def __init__(self):
        super(MockNetwork, self).__init__()

    def build_network(self, input_tensors):
        pass


class MockRewardShaper(RewardShaper):
    def __init__(self, simu_ctx, agent_ctx):
        super(MockRewardShaper, self).__init__(simu_ctx, agent_ctx)

    def initialize(self):
        pass

    def should_train(self, exprs):
        pass

    def assign_rewards(self, exprs):
        pass

    def finalize(self):
        pass


class MockPolicyConf:
    def __init__(self, policy_builder, **kwargs):
        """
        每个policy只有一个强制的字段就是policy_builder, 指定policy的工厂类,
        如果是trainable的policy, 一般还需要指定algo字段
        其他的字段可以扩展, 比如对于RL类型的policy, 一般需要定义state、action、reward以及reward_shaper等
        """
        # 这里使用mock，而不是locate到真正的policy_builder
        self.policy_builder = MockPolicyBuilder

        algo = kwargs.pop('algo', None)
        if algo is not None:
            self.algo = algo

        # 这里使用mock，而不是locate到真正的类
        attrs = {
            "state": MockState,
            "action": MockAction,
            "reward": MockReward,
            "network": MockNetwork,
            "reward_shaper": MockRewardShaper
        }
        self.__dict__.update(**attrs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


def build_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        builder, encoded = func(*args, **kwargs)

        builder.Finish(encoded)
        buf = builder.Output()
        return buf

    return wrapper


class MockKaiwuMsg:
    @staticmethod
    @build_decorator
    def build_init_req():
        builder = flatbuffers.Builder(0)

        init_req = KaiwuMsgHelper.encode_init_req(builder, 'fake client_id', 'fake client_version', b'Hello World!')

        return builder, init_req

    @staticmethod
    @build_decorator
    def build_init_req_request():
        builder = flatbuffers.Builder(0)

        init_req = KaiwuMsgHelper.encode_init_req(builder, 'fake client_id', 'fake client_version', b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.init_req, init_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_ep_start_req():
        builder = flatbuffers.Builder(0)

        ep_start_req = KaiwuMsgHelper.encode_ep_start_req(builder, 'fake client_id', 1, b'Hello World!')

        return builder, ep_start_req

    @staticmethod
    @build_decorator
    def build_ep_start_req_request():
        builder = flatbuffers.Builder(0)

        ep_start_req = KaiwuMsgHelper.encode_ep_start_req(builder, 'fake client_id', 1, b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_start_req, ep_start_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_agent_start_req():
        builder = flatbuffers.Builder(0)

        agent_start_req = KaiwuMsgHelper.encode_agent_start_req(builder, 'fake client_id', 1, 0, b'Hello World')

        return builder, agent_start_req

    @staticmethod
    @build_decorator
    def build_agent_start_req_request():
        builder = flatbuffers.Builder(0)

        agent_start_req = KaiwuMsgHelper.encode_agent_start_req(builder, 'fake client_id', 1, 0, b'Hello World')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.agent_start_req, agent_start_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_update_req():
        builder = flatbuffers.Builder(0)

        update_req = KaiwuMsgHelper.encode_update_req(builder, 'fake client_id', 1, {
            0: [b"Hello World 0"],
            1: [b"Hello World 1"]
        })

        return builder, update_req

    @staticmethod
    @build_decorator
    def build_update_req_request():
        builder = flatbuffers.Builder(0)

        update_req = KaiwuMsgHelper.encode_update_req(builder, 'fake client_id', 1, {
            0: [b"Hello World 0"],
            1: [b"Hello World 1"]
        })

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.update_req, update_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_agent_end_req():
        builder = flatbuffers.Builder(0)

        agent_end_req = KaiwuMsgHelper.encode_agent_end_req(builder, 'fake client_id', 1, 0, b'Hello World!')

        return builder, agent_end_req

    @staticmethod
    @build_decorator
    def build_agent_end_req_request():
        builder = flatbuffers.Builder(0)

        agent_end_req = KaiwuMsgHelper.encode_agent_end_req(builder, 'fake client_id', 1, 0, b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.agent_end_req, agent_end_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_ep_end_req():
        builder = flatbuffers.Builder(0)

        ep_end_req = KaiwuMsgHelper.encode_ep_end_req(builder, 'fake client_id', 1, b'Hello World!')

        return builder, ep_end_req

    @staticmethod
    @build_decorator
    def build_ep_end_req_request():
        builder = flatbuffers.Builder(0)

        ep_end_req = KaiwuMsgHelper.encode_ep_end_req(builder, 'fake client_id', 1, b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_end_req, ep_end_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_event_req():
        builder = flatbuffers.Builder(0)

        event_req = KaiwuMsgHelper.encode_event_req(builder, 'fake client_id', b'Hello World!')

        return builder, event_req

    @staticmethod
    @build_decorator
    def build_event_req_request():
        builder = flatbuffers.Builder(0)

        event_req = KaiwuMsgHelper.encode_event_req(builder, 'fake client_id', b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.event_req, event_req)

        return builder, request

    @staticmethod
    @build_decorator
    def build_quit_request():
        builder = flatbuffers.Builder(0)

        # avoid built-in name 'quit'
        encoded_quit = KaiwuMsgHelper.encode_quit(builder, 'fake client_id', 0, 'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.quit, encoded_quit)

        return builder, request

    @staticmethod
    @build_decorator
    def build_heartbeat_request():
        builder = flatbuffers.Builder(0)

        heartbeat = KaiwuMsgHelper.encode_heartbeat(builder, 'fake client_id', b'Hello World!')

        request = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.heartbeat, heartbeat)

        return builder, request

class MockFlags:
    def __init__(self, args=None):
        if args is None:
            args = {}
        self._dict = args

    def __getattr__(self, key):
        if key in self._dict:
            return self._dict.get(key)
        elif key in dir(self):
            return getattr(self, key)

        return mock.Mock()

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]

        return mock.Mock()

    def __setitem__(self, key, value):
        self._dict[key] = value

    def to_dict(self):
        return self._dict

class MockAppConf:
    class MockTupleType:
        def __init__(self):
            self.run_handler = mock.Mock()

            self.policies = {}

            self.builder = mock.Mock()

    def __init__(self, args=None):
        if args is None:
            args = {}
        self._dict = args

    def __getattr__(self, key):
        if key in dir(self):
            return getattr(self, key)

        return mock.Mock()

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]

        return self.MockTupleType()

class TestKaiwuEnviron(unittest.TestCase):
    def setUp(self) -> None:
        self.app = "gym"
        args = {
            'app': self.app,
            'update_frame_rate_range': [24, 26, 28, 30, 32],
            "frame_interval": 1,  # 每隔几帧发送一次update消息
            'dequeue_retry_times': 60,
            'dequeue_timeout_ms': 1000,
        }

        # mock AppConf
        #   1. create instance of MockTupleType
        mock_tuple_type = MockAppConf.MockTupleType()
        #   2. mock instance
        #       2.1 首先是获取用户配置的policies
        fake_one_app_conf = {
            self.app: {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "policies": {
                    "train": {
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
        #       2.3 根据其中的builder设置tuple_type的builder, 这里使用mock，而不是locate到真正MetalBuilder
        # mock_tuple_type.builder = MockBuilder
        #       2.4 根据其中的policies设置tuple_type的policies,
        #           tuple_type的policies将每个policy映射到对于的policy_conf, 这里使用mock
        policies_dict = fake_one_app_conf["gym"]["policies"]
        mock_tuple_type.policies = {name: MockPolicyConf(**policy) for name, policy in policies_dict.items()}
        #   3. 构造app和tuple_type之前的映射，传入MockAppConf
        mock_app_conf_args = {
            self.app: mock_tuple_type
        }
        kaiwu_environ.AppConf = MockAppConf(mock_app_conf_args)

 
        fake_simu_ctx = Context()
        exit_flag = multiprocessing.Value('b', False)
        fake_client_conn_id = 12345
        self.kaiwu_environ = KaiwuEnviron(simu_ctx=fake_simu_ctx, exit_flag=exit_flag,
                                                  client_conn_id=fake_client_conn_id)

    def test_identity(self):
        identity = self.kaiwu_environ.identity

        self.assertTrue(isinstance(identity, str))
        self.assertTrue(len(identity) > 0)

    """
    KaiwuEnviron.init(self)单测

    背景：
    KaiWuRLHelper在刚开始运行的时候会调用init()函数, 用于接收init_req请求,完成相关初始化
    """

    def test_initialize(self):
        self.kaiwu_environ.init()

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_init")
    def test_initialize_exception(self, mock_handle_init):
        mock_handle_init.side_effect = Exception("Fake _handle_init Exception")

        self.kaiwu_environ._client_id = "fake client id"  # 用于触发self._run_handler.on_quit(self._client_id)
        self.assertRaises(Exception, self.kaiwu_environ.init)

        self.kaiwu_environ._client_id = ""  # reset

    """
    KaiwuEnviron.reset(self)单测

    背景：
    KaiWuRLHelper在每个episode开始运行时会调用reset()函数, 用于接受ep_start_req, agent_start_req, 和 first update_req一系列请求
    reset()函数也会对收到的其他请求做异常处理
    """

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.next_valid")
    def test_reset_return_states(self, mock_next_valid):
        # create fake return of KaiwuEnviron.handle_update
        # 参考_handle_update的实现
        #   fake_states 存放每个agent对应的状态（dict类型，内部是policy_id: State继承类的实例）
        #   fake_ex_rewards 存放每个agent对应rewards(list类型)
        fake_agent_id = 0
        fake_policy_id = "train"
        fake_states = {fake_agent_id: {fake_policy_id: MockState()}}
        fake_ex_rewards = {fake_agent_id: []}
        fake_dones = {agent_id: False for agent_id in fake_states}
        fake_dones['_all_done_'] = False
        mock_next_valid.return_value = fake_states, fake_ex_rewards, fake_dones

        ret_states = self.kaiwu_environ.reset()

        self.assertEqual(ret_states, fake_states)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.next_valid")
    def test_reset_exception(self, mock_next_valid):
        # create fake return of KaiwuEnviron.handle_ep_end
        # 参考_handle_ep_end的实现
        # fake_value_2 为 {'_all_done_': True}， 其余为空dict
        fake_value_0 = {}
        fake_value_1 = {}
        fake_value_2 = {'_all_done_': True}
        mock_next_valid.return_value = fake_value_0, fake_value_1, fake_value_2

        self.assertRaises(SkipEpisodeException, self.kaiwu_environ.reset)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.next_valid")
    def test_reset_assert(self, mock_next_valid):
        # create fake return of KaiwuEnviron.handle_agent_end
        # 参考_handle_agent_end的实现
        fake_agent_id = 0
        fake_value_0 = {}
        fake_value_1 = {}
        fake_value_2 = {fake_agent_id: False, '_all_done_': False}  # 为了跳出while loop将True改成False

        mock_next_valid.return_value = fake_value_0, fake_value_1, fake_value_2

        self.assertRaises(AssertionError, self.kaiwu_environ.reset)

    """
    KaiwuEnviron.step(self, actions, extra_info=None)单测

    背景：
    KaiWuRLHelper在一局episode中对多次调用step函数用来对agent的action进行处理(调用handle_step函数), 并接受下一个特定请求
    参数1: actions, 字典: {agent_id: action类实例}
        action类为policy_conf中的action类, policy_conf example 如下
            "policy_builder": "framework.server.aisrv.async_policy.AsyncBuilder",
            "algo": "ppo",
            "state": "app.gym.gym_proto.GymLSTMState",
            "action": "app.gym.gym_proto.GymAction",
            "reward": "app.gym.gym_proto.GymReward",
            "network": "app.gym.gym_network.GymLSTMDeepNetwork",
            "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
    参数2: extra_info
    """

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_step")
    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.next_valid")
    def test_step_return(self, mock_next_valid, mock_handle_step):
        fake_agent_id = 0
        fake_actions = {fake_agent_id: MockAction()}

        # create fake return of KaiwuEnviron.handle_update
        fake_agent_id = 0
        fake_policy_id = "train"
        fake_states = {fake_agent_id: {fake_policy_id: MockState()}}
        fake_ex_rewards = {fake_agent_id: []}
        fake_dones = {agent_id: False for agent_id in fake_states}
        fake_dones['_all_done_'] = False
        mock_next_valid.return_value = fake_states, fake_ex_rewards, fake_dones

        ret_states, ret_ex_rewards, ret_dones = self.kaiwu_environ.step(fake_actions)

        mock_handle_step.assert_called_once()
        mock_next_valid.assert_called_once()
        self.assertEqual(fake_states, ret_states)
        self.assertEqual(fake_ex_rewards, ret_ex_rewards)
        self.assertEqual(fake_dones, ret_dones)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_step")
    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.next_valid")
    def test_step_exception(self, mock_next_valid, mock_handle_step):
        fake_agent_id = 0
        fake_actions = {fake_agent_id: MockAction()}

        mock_handle_step.side_effect = Exception("Fake _handle_step Exception")
        self.assertRaises(Exception, self.kaiwu_environ.step, fake_actions)

        mock_next_valid.assert_not_called()

    """
    KaiwuEnviron.reject(self, e)单测

    背景：
    KaiWuRLHelper在一些情况下会需要主动与客户端断连, 会调用这个函数
    该函数会构造一个reject的消息, 发送给客户端, 最后调用finsh函数做收尾工作
    """

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.send_rsp")
    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.finsh")
    def test_reject(self, mock_finalize, mock_send_rsp):
        fake_e = Exception("Fake e in reject")

        self.kaiwu_environ.reject(fake_e)
        mock_finalize.assert_called_once()
        mock_send_rsp.assert_called_once()

        self.kaiwu_environ.exit_flag.value = False  # reset

    """
    KaiwuEnviron.policy_mapping_fn(self, agent_id)单测

    背景：
    该函数会根据传入的agent_id来返回该agent使用的policy_id, 比如"train"
    具体进一步调用run_handler中的policy_mapping_fn函数
    """

    def test_policy_mapping_fn(self):
        fake_agent_id = 0
        policy_id = self.kaiwu_environ.policy_mapping_fn(fake_agent_id)

        self.assertEqual(policy_id, "train")

    """
    KaiwuEnviron.reject(self, e)单测

    背景：
    KaiWuRLHelper与客户端断连前的最后操作, 会调用这个函数
    """

    def test_finsh(self):
        self.kaiwu_environ.finsh()

        self.kaiwu_environ.exit_flag.value = False  # reset

    """
    KaiwuEnviron.next_valid(self)单测

    背景：
    KaiWuRLHelper在需要收取特定请求并获取对特定请求的处理结果时会调用next_valid函数()
    该函数会首先收取请求，然后调用相应的函数对请求进行处理
    但只会在收到特定几个请求(update_req, agent_end_req, ep_end_req)时才会将处理结果正常返回
    对于其他请求，该函数处理完后会继续收取(ep_start_req, agent_start_req, event_req, heartbeat)请求或者抛出异常(quit, 非法请求)
    """

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_ep_start")
    def test_next_valid_runtime_error(self, mock_deque_client_req, mock_handle_ep_start):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_ep_start_req_request()

        # jump out while loop
        mock_handle_ep_start.side_effect = RuntimeError("Fake _handle_ep_start RuntimeError")

        self.assertRaises(RuntimeError, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_ep_start")
    def test_next_valid_ep_start_req(self, mock_deque_client_req, mock_handle_ep_start):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_ep_start_req_request()

        # jump out while loop
        mock_handle_ep_start.side_effect = Exception("Fake _handle_ep_start Exception")

        self.assertRaises(Exception, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_agent_start")
    def test_next_valid_agent_start_req(self, mock_deque_client_req, mock_handle_agent_start):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_agent_start_req_request()

        # jump out while loop
        mock_handle_agent_start.side_effect = Exception("Fake _handle_agent_start Exception")

        self.assertRaises(Exception, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_update")
    def test_next_valid_update_req(self, mock_deque_client_req, mock_handle_update):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_update_req_request()

        # create fake return
        fake_valid = True
        fake_new_states = []
        fake_ex_rewards = []
        fake_dones = {}
        mock_handle_update.return_value = fake_valid, fake_new_states, fake_ex_rewards, fake_dones

        ret_new_states, ret_ex_rewards, ret_dones = self.kaiwu_environ._next_valid()

        self.assertEqual(fake_new_states, ret_new_states)
        self.assertEqual(fake_ex_rewards, ret_ex_rewards)
        self.assertEqual(fake_dones, ret_dones)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_agent_end")
    def test_next_valid_agent_end_req(self, mock_deque_client_req, mock_handle_agent_end):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_agent_end_req_request()

        # create fake return
        fake_agent_id = 0
        fake_value_0 = {}
        fake_value_1 = {}
        fake_value_2 = {fake_agent_id: True, '_all_done_': False}
        mock_handle_agent_end.return_value = fake_value_0, fake_value_1, fake_value_2

        ret_value_0, ret_value_1, ret_value_2 = self.kaiwu_environ._next_valid()

        self.assertEqual(fake_value_0, ret_value_0)
        self.assertEqual(fake_value_1, ret_value_1)
        self.assertEqual(fake_value_2, ret_value_2)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_ep_end")
    def test_next_valid_ep_end_req(self, mock_deque_client_req, mock_handle_ep_end):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_ep_end_req_request()
        # fake ep start ts
        self.kaiwu_environ._ep_start_ts = time.monotonic()

        # create fake return
        fake_value_0 = {}
        fake_value_1 = {}
        fake_value_2 = {'_all_done_': True}
        mock_handle_ep_end.return_value = fake_value_0, fake_value_1, fake_value_2

        ret_value_0, ret_value_1, ret_value_2 = self.kaiwu_environ._next_valid()

        self.assertEqual(fake_value_0, ret_value_0)
        self.assertEqual(fake_value_1, ret_value_1)
        self.assertEqual(fake_value_2, ret_value_2)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_event")
    def test_next_valid_event_req(self, mock_deque_client_req, mock_handle_event):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_event_req_request()

        # jump out while loop
        mock_handle_event.side_effect = Exception("Fake _handle_event Exception")

        self.assertRaises(Exception, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.finsh")
    def test_next_valid_quit(self, mock_deque_client_req, mock_finalize):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_quit_request()

        self.assertRaises(ClientQuitException, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.handle_heartbeat")
    def test_next_valid_heartbeat(self, mock_deque_client_req, mock_handle_heartbeat):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_heartbeat_request()

        # jump out while loop
        mock_handle_heartbeat.side_effect = Exception("Fake _handle_heartbeat Exception")

        self.assertRaises(Exception, self.kaiwu_environ._next_valid)

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.recv_req")
    def test_next_valid_invalid_msg_type(self, mock_recv_req):
        mock_recv_req.return_value = 10000, None

        self.assertRaises(RuntimeError, self.kaiwu_environ.next_valid)

    """
    KaiwuEnviron.handle_init(self, req)单测
    KaiwuEnviron.handle_ep_start(self, req)单测
    KaiwuEnviron.handle_agent_start(self, req)单测

    背景：
    KaiwuEnviron内部调用handle_init函数对解码后的init_req进一步解码, 获取请求内部信息(client_id, client_version, req_data)
        然后调用用户写的run_handler中的on_init函数对内部信息进行处理
        最后包装出init_rsp消息, 并调用_send_rsp将消息返回给client

    其余和_handle_init类似
    """

    def test_handle_init(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        buf = MockKaiwuMsg.build_init_req()

        fake_init_req = InitReq.InitReq.GetRootAsInitReq(buf, 0)

        self.kaiwu_environ.handle_init(fake_init_req)

    def test_handle_ep_start(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        self.kaiwu_environ._curr_agents = set()
        self.kaiwu_environ._client_id = 'fake client_id'

        buf = MockKaiwuMsg.build_ep_start_req()

        fake_ep_start_req = EpStartReq.EpStartReq.GetRootAsEpStartReq(buf, 0)

        self.kaiwu_environ.handle_ep_start(fake_ep_start_req)

    def test_handle_agent_start(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        self.kaiwu_environ._curr_agents = set()
        self.kaiwu_environ._client_id = 'fake client_id'
        self.kaiwu_environ._ep_id = 1

        buf = MockKaiwuMsg.build_agent_start_req()

        fake_agent_start_req = AgentStartReq.AgentStartReq.GetRootAsAgentStartReq(buf, 0)

        self.kaiwu_environ.handle_agent_start(fake_agent_start_req)

        self.assertTrue(len(self.kaiwu_environ._curr_agents) == 1)

    """
    KaiwuEnviron._handle_update(self, req)单测

    背景：
    KaiwuEnviron收到客户发来的update req后对调用_handle_update函数进行处理
    该函数首先调用用户实现的self.run_handler的on_update_req函数获取new_states, ex_rewards
        new_state 存放每个agent对应的状态
            状态类型有两种可能
                一种是是dict类型, 内部是policy_id: State继承类的实例
                另一种是State继承类的实例
        ex_rewards 存放每个agent对应rewards(list类型)
    然后对new_states进行normalize处理
        处理后的new_states 存放每个agent对应的状态, 状态类型是dict类型, 内部是policy_id: State继承类的实例
    最后会对new_states进行判断, 返回valid是否为True
    """

    def test_handle_update(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()
        self.kaiwu_environ._curr_agents = {0, 1}
        self.kaiwu_environ._client_id = 'fake client_id'
        self.kaiwu_environ._ep_id = 1
        self.kaiwu_environ._time_new_state = time.monotonic()

        buf = MockKaiwuMsg.build_update_req()

        fake_update_req = UpdateReq.UpdateReq.GetRootAsUpdateReq(buf, 0)

        valid, new_states, ex_rewards, dones = self.kaiwu_environ._handle_update(fake_update_req)

        self.assertTrue(valid)
        self.assertEqual(len(new_states), 2)
        self.assertEqual(len(ex_rewards), 2)
        self.assertTrue(len(dones), 3)

    """
    KaiwuEnviron.handle_step(self, actions, extra_info)单测

    背景：
    KaiwuEnviron调用handle_step函数用来对agent的action进行处理(调用handle_step函数), 并接受下一个特定请求
    参数1: actions, 字典: {agent_id: action类实例}
          action类为policy_conf中的action类, 比如: "action": "app.gym.gym_proto.GymAction",
    参数2: extra_info
    该函数首先对extra_info进行normalize,
    然后调用用户实现的self._run_handler的on_update_rsp函数, 来获取返回给客户端的rsp_data
        rsp_data存放每个agent对应的回复信息(格式为bytes)
    最后对rsp_data进行encode,并调用_send_rsp函数将数据发给client
    """

    def test_handle_step(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()
        self.kaiwu_environ._ep_id = 1

        fake_actions = {
            0: MockAction(),
            1: MockAction()
        }
        fake_extra_info = {
            0: {},
            1: {}
        }
        self.kaiwu_environ.handle_step(actions=fake_actions, extra_info=fake_extra_info)

    """
    KaiwuEnviron.handle_agent_end(self, req)单测
    KaiwuEnviron.handle_ep_end(self, req)单测
    KaiwuEnviron.handle_agent_event(self, req)单测

    背景：
    和_handle_init类似
    不同点在于handle_agent_end和handle_ep_end有返回值
    """

    def test_handle_agent_end(self):
        self.kaiwu_environ.curr_req_arrive_time = time.monotonic()

        fake_end_id = 0
        self.kaiwu_environ.curr_agents = {fake_end_id}
        self.kaiwu_environ._client_id = 'fake client_id'
        self.kaiwu_environ._ep_id = 1

        buf = MockKaiwuMsg.build_agent_end_req()

        fake_agent_end_req = AgentEndReq.AgentEndReq.GetRootAsAgentEndReq(buf, 0)

        ret_value_0, ret_value_1, ret_value_2 = self.kaiwu_environ.handle_agent_end(fake_agent_end_req)

        self.assertTrue(len(self.kaiwu_environ.curr_agents) == 0)

        self.assertEqual(len(ret_value_0), 0)
        self.assertTrue(isinstance(ret_value_0, dict))
        self.assertEqual(len(ret_value_1), 0)
        self.assertTrue(isinstance(ret_value_1, dict))

        self.assertTrue(isinstance(ret_value_2, dict))
        self.assertTrue('_all_done_' in ret_value_2)
        self.assertEqual(ret_value_2['_all_done_'], False)
        self.assertTrue(fake_end_id in ret_value_2)
        self.assertEqual(ret_value_2[fake_end_id], True)

    def test_handle_ep_end(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        self.kaiwu_environ._curr_agents = set()
        self.kaiwu_environ._client_id = 'fake client_id'
        self.kaiwu_environ._ep_id = 1

        buf = MockKaiwuMsg.build_ep_end_req()

        fake_ep_end = EpEndReq.EpEndReq.GetRootAsEpEndReq(buf, 0)

        ret_value_0, ret_value_1, ret_value_2 = self.kaiwu_environ.handle_ep_end(fake_ep_end)

        self.assertEqual(len(ret_value_0), 0)
        self.assertTrue(isinstance(ret_value_0, dict))
        self.assertEqual(len(ret_value_1), 0)
        self.assertTrue(isinstance(ret_value_1, dict))

        self.assertTrue(isinstance(ret_value_2, dict))
        self.assertTrue('_all_done_' in ret_value_2)
        self.assertEqual(ret_value_2['_all_done_'], True)

    def test_handle_event(self):
        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        self.kaiwu_environ._client_id = 'fake client_id'
        self.kaiwu_environ._ep_id = 1

        buf = MockKaiwuMsg.build_event_req()

        fake_event_req = EventReq.EventReq.GetRootAsEventReq(buf, 0)
        self.kaiwu_environ._handle_event(fake_event_req)

    """
    KaiwuEnviron.recv_req(self)单测

    """
    def test_recv_req(self, mock_deque_client_req):
        mock_deque_client_req.return_value = MockKaiwuMsg.build_init_req_request()
        msg_type, msg = self.kaiwu_environ._recv_req()

        self.assertEqual(msg_type, ReqMsg.ReqMsg.init_req)
        self.assertTrue(isinstance(msg, InitReq.InitReq))

    def test_recv_req_timeout(self, mock_deque_client_req):
        mock_deque_client_req.side_effect = TimeoutError
        self.assertRaises(TimeoutError, self.kaiwu_environ._recv_req)

    def test_recv_req_exception(self, mock_deque_client_req):
        mock_deque_client_req.side_effect = Exception("Fake decode_request Exception")
        self.assertRaises(Exception, self.kaiwu_environ._recv_req)

    """
    KaiwuEnviron.send_rsp(self, msg_type, msg)单测
    KaiwuEnviron.handle_heartbeat(self, req)单测
    KaiwuEnviron.send_rsp_to_client(self, fb_rsp)单测

    背景
    KaiwuEnviron内部调用send_rsp()函数将收到的各种类型的rsp进一步构造成response
        并通过调用send_rsp_to_client函数将response发送给client

    KaiwuEnviron内部调用handle_heartbeat()函数构造空response, 并通过调用send_rsp_to_client函数将response发送给client

    KaiwuEnviron内部调用send_rsp_to_client()函数, 该函数将response发给client
    """

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.send_rsp_to_client")
    def test_send_rsp(self, mock_send_rsp_to_client):
        # create fake init rsp
        fake_ret_code = 0
        fake_rsp = KaiwuMsgHelper.encode_init_rsp(self.kaiwu_environ.builder, fake_ret_code)

        self.kaiwu_environ.send_rsp(RspMsg.RspMsg.init_rsp, fake_rsp)

        mock_send_rsp_to_client.assert_called_once()

    @mock.patch("framework.server.aisrv.kaiwu_environ.KaiwuEnviron.send_rsp_to_client")
    def test_handle_heartbeat(self, mock_send_rsp_to_client):
        fake_req = None
        self.kaiwu_environ.handle_heartbeat(fake_req)

        mock_send_rsp_to_client.assert_called_once()

    def test_send_rsp_to_client(self):
        exist_flag = hasattr(self.kaiwu_environ, "_curr_req_arrive_time")

        self.kaiwu_environ._curr_req_arrive_time = time.monotonic()

        fake_fb_rsp = None
        self.kaiwu_environ.send_rsp_to_client(fake_fb_rsp)

        if not exist_flag:
            # reset if necessary
            delattr(self.kaiwu_environ, "_curr_req_arrive_time")


if __name__ == '__main__':
    unittest.main()
