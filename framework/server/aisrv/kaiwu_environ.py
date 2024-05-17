#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import traceback
import datetime
import flatbuffers
import os
from framework.common.config.app_conf import AppConf
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import TimeIt
from framework.interface.state import State
from framework.server.aisrv.flatbuffer.kaiwu_msg import ReqMsg, Request, RspMsg
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from framework.interface.exception import ClientQuitException, RestartException, SkipEpisodeException
from framework.server.aisrv.environ import Environ
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine


class KaiwuEnviron(Environ):
    def __init__(self, simu_ctx, exit_flag, client_conn_id):
        super().__init__(simu_ctx)

        self.client_conn_id = client_conn_id

        # 日志配置, 从调用方kaiwu_rl_helper传入
        self.logger = simu_ctx.logger

        self.run_handler = AppConf[CONFIG.app].run_handler(simu_ctx)
        self.builder = flatbuffers.Builder(0)
        self.curr_agents = set()

        self._client_id = ""
        self._ep_id = -1

        # 和kaiwu_rl_helper公用
        self.exit_flag = exit_flag
        self.seqno = 0

        # 拿到一个新的state的时间
        self.time_new_state = None
        self.update_frame_count = 0

        # MsgBuff
        self.msg_buff = simu_ctx.msg_buff

        # 有些场景需要保存上一次的预测结果
        self.last_action = [[0, 0, 10000]]

    @property
    def identity(self):
        return "Env(conn_id: %s, client_id: %s, ep_id: %d)" % \
               (self.client_conn_id, self._client_id or "", self._ep_id)

    def init(self):
        self.logger.info("kaiwu_environ init")
        msg_type, req = self.recv_req()
        assert msg_type == ReqMsg.ReqMsg.init_req, "environ initialize: invalid msg type %d" % msg_type

        try:
            self.handle_init(req)
        except Exception as e:
            if self._client_id:
                self.run_handler.on_quit(self._client_id)
            raise e

    def reset(self):
        # receive ep_start, agent_start, and first update msg
        states, _, dones = self.next_valid()

        while True:
            if dones['_all_done_']:
                raise SkipEpisodeException(self._client_id, self._ep_id)

            # will always return the first correct states
            if len(states) > 0:
                return states

            assert any(dones.values()), f"no agent has finished: {dones}"
            states, _, dones = self.next_valid()

    def step(self, actions, extra_info=None):
        if len(actions) > 0:
            try:
                self.handle_step(actions, extra_info=extra_info)
            except Exception as e:
                self.run_handler.on_quit(self._client_id)
                raise e
        return self.next_valid()

    def reject(self, e):
        # error stop
        self.exit_flag.value = True

        reject = KaiwuMsgHelper.encode_reject(self.builder, -1, str(e))
        self.send_rsp(RspMsg.RspMsg.reject, reject)
        self.finsh()

    def finsh(self):
        # 幂等操作
        self.exit_flag.value = True

    @property
    def client_id(self):
        return self._client_id

    @property
    def client_version(self):
        return self.client_version

    @property
    def ep_id(self):
        return self._ep_id

    def policy_mapping_fn(self, agent_id):
        return self.run_handler.policy_mapping_fn(agent_id)

    def next_valid(self):
        while not self.exit_flag.value:
            try:
                msg_type, req = self.recv_req()
                if self.msg_buff.input_q.qsize() != 0:
                    self.logger.error("kaiwu_environ Not Zero")
                # assert self.msg_buff.input_q.qsize()==0
                if msg_type == ReqMsg.ReqMsg.ep_start_req:
                    self.ep_start_ts = time.monotonic()
                    self.update_frame_count = 0
                    self.handle_ep_start(req)
                elif msg_type == ReqMsg.ReqMsg.init_req:
                    self.on_handle_init(req)
                elif msg_type == ReqMsg.ReqMsg.agent_start_req:
                    self.handle_agent_start(req)
                elif msg_type == ReqMsg.ReqMsg.update_req:
                    self.update_frame_count += 1
                    self._time_new_state = time.monotonic()

                    if CONFIG.app == KaiwuDRLDefine.APP_GYM:
                        valid, new_states, ex_rewards, dones = self.handle_update(req)
                        if valid:
                            return new_states, ex_rewards, dones

                    elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_1V1:
                        return self.on_handle_update(req)

                    elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
                        return self.on_handle_5v5_update(req)
                    else:
                        return self.on_handle_update(req)

                elif msg_type == ReqMsg.ReqMsg.agent_end_req:
                    return self.handle_agent_end(req)
                
                elif msg_type == ReqMsg.ReqMsg.ep_end_req:
                    res = self.on_handle_ep_end(req)
                    return res
                    
                elif msg_type == ReqMsg.ReqMsg.event_req:
                    self.handle_event(req)
                
                elif msg_type == ReqMsg.ReqMsg.quit:
                    client_id, quit_code, message = KaiwuMsgHelper.decode_quit(req)
                    self.finsh()
                    raise ClientQuitException(client_id, quit_code, message)
                
                elif msg_type == ReqMsg.ReqMsg.heartbeat:
                    self.handle_heartbeat(req)
                
                elif msg_type:
                    # when msg_type is None, skip it and continue
                    raise RuntimeError("next valid: invalid msg type: %d" % msg_type)
            except Exception as e:
                # 如果是Restart，则不断开连接，继续接收下一个消息
                if not isinstance(e, RestartException):
                    # 正常断联和异常断联都需要调用on_quit以确保资源回收
                    self.run_handler.on_quit(self._client_id)
                    raise e

        raise RuntimeError("Environ finish")

    def on_handle_init(self, req):
        # run_hanle带有msg_buf 返回的消息给BattleServer的消息由业务自定义
        self.logger.info(f"kaiwu_environ on handle client init request, {self.identity}")
        self._client_id, self._client_version, req_data = KaiwuMsgHelper.decode_init_req(req)

        try:
            self.run_handler.on_init(self._client_id, req_data)
        except Exception as e:
            traceback.print_exc()
            raise e

    def handle_init(self, req):
        self._client_id, self._client_version, req_data = KaiwuMsgHelper.decode_init_req(req)
        self.logger.info(f"kaiwu_environ Handle client init, {self.identity}")
        ret_code = 0
        try:
            self.run_handler.on_init(self._client_id, req_data)
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_init_rsp(self.builder, ret_code)
            self.send_rsp(RspMsg.RspMsg.init_rsp, rsp)

    def handle_ep_start(self, req):
        client_id, self._ep_id, req_data = KaiwuMsgHelper.decode_ep_start_req(req)
        self.logger.info(f"kaiwu_environ Handle ep start, {self.identity}")

        assert len(self.curr_agents) == 0, "episode should start before all agents"
        assert client_id == self._client_id, f"invalid client id {client_id}, right is {client_id}"

        ret_code = 0
        try:
            self.run_handler.on_ep_start(client_id, self._ep_id, req_data)
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_ep_start_rsp(
                self.builder, ret_code, self._ep_id, int(CONFIG.frame_interval))
            self.send_rsp(RspMsg.RspMsg.ep_start_rsp, rsp)

    def handle_agent_start(self, req):
        client_id, ep_id, agent_id, req_data = KaiwuMsgHelper.decode_agent_start_req(req)
        self.logger.debug(f"kaiwu_environ Handle agent start, {self.identity}")

        assert client_id == self._client_id, \
            f"mismatch client id: {client_id} -> {self._client_id}"
        assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
        assert agent_id not in self.curr_agents, f"duplicated agent {agent_id} found"

        self.curr_agents.add(agent_id)

        ret_code = 0
        try:
            self.run_handler.on_agent_start(client_id, ep_id, agent_id, req_data)
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_agent_start_rsp(self.builder, ret_code, ep_id, agent_id)
            self.send_rsp(RspMsg.RspMsg.agent_start_rsp, rsp)

    def on_handle_update(self, req):
        try:
            client_id, ep_id, req_data = KaiwuMsgHelper.decode_update_req(req)
            # self.logger.debug(f"kaiwu_environ Handle update, {self.identity}")

            assert client_id == self._client_id, \
                f"mismatch client id: {client_id} -> {self._client_id}"
            assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
            assert all([agent_id in self.curr_agents for agent_id in req_data]), \
                f"mismatch agent id: {set(req_data.keys())} -> {self.curr_agents}"

            with TimeIt() as ti:
                if hasattr(self.run_handler, 'set_last_action'):
                    self.run_handler.set_last_action(self.last_action)
                    
                valid, states, must_need_info = self.run_handler.on_update_req(
                    client_id, self._ep_id, req_data)
                if valid:
                    return states, must_need_info
                else:
                    return None, None
        except Exception as e:
            traceback.print_exc()
            raise e

    def on_handle_5v5_update(self, req):
        try:
            client_id, ep_id, req_data = KaiwuMsgHelper.decode_update_req(req)
            self.logger.debug(f"kaiwu_environ Handle update, {self.identity}")

            assert client_id == self._client_id, \
                f"mismatch client id: {client_id} -> {self._client_id}"
            assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
            #assert all([agent_id in self.curr_agents for agent_id in req_data]), \
            #    f"mismatch agent id: {set(req_data.keys())} -> {self.curr_agents}"

            with TimeIt() as ti:
                valid, states, must_need_info = self.run_handler.on_update_req(
                    client_id, self._ep_id, req_data)
                if valid:
                    return states, must_need_info
                else:
                    return None, None
        except Exception as e:
            traceback.print_exc()
            raise e

    def handle_update(self, req):
        client_id, ep_id, req_data = KaiwuMsgHelper.decode_update_req(req)

        self.logger.debug(f"kaiwu_environ Handle update, {self.identity}")

        assert client_id == self._client_id, \
            f"mismatch client id: {client_id} -> {self._client_id}"
        assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
        assert all([agent_id in self.curr_agents for agent_id in req_data]), \
            f"mismatch agent id: {set(req_data.keys())} -> {self.curr_agents}"

        new_states, ex_rewards = {}, {}
        try:
            with TimeIt() as ti:
                new_states, ex_rewards = self.run_handler.on_update_req(
                    client_id, self._ep_id, req_data)
        except RestartException as e:
            rsp = KaiwuMsgHelper.encode_restart(self.builder, e.client_id, e.ep_id, e.data)
            self.send_rsp(RspMsg.RspMsg.restart, rsp)
            raise e
        except Exception as e:
            rsp = KaiwuMsgHelper.encode_update_rsp(self.builder, -1, self._ep_id, {})
            self.send_rsp(RspMsg.RspMsg.update_rsp, rsp)
            raise e

        def normalize_states(input_states):
            output_states = {}
            for agent_id, v in input_states.items():
                if isinstance(v, dict):
                    output_states[agent_id] = v
                elif isinstance(v, State):
                    policy_id = self.run_handler.policy_mapping_fn(agent_id)
                    assert isinstance(policy_id, str), f"policy_id {policy_id} should be string type"
                    output_states[agent_id] = {}
                    output_states[agent_id][policy_id] = v
            return output_states

        new_states = normalize_states(new_states)

        assert len(new_states) == len(ex_rewards)
        assert all([
            all([isinstance(v, State) for _, v in new_state.items()])
            for _, new_state in new_states.items()
        ])
        assert all([isinstance(ex_reward, list) for _, ex_reward in ex_rewards.items()])

        valid = len(new_states) > 0
        if not valid:
            # 可能有的请求是无效帧，或者只用于初始化或者修改一些状态，不需要做预测，所以只用回空动作
            rsp = KaiwuMsgHelper.encode_update_rsp(self.builder, 0, self._ep_id, {})
            self.send_rsp(RspMsg.RspMsg.update_rsp, rsp)

        # 必须通过agent_end来标识agent完成
        dones = {agent_id: False for agent_id in new_states}
        dones['_all_done_'] = False

        return valid, new_states, ex_rewards, dones

    def on_handle_action(self, action):
        try:
            self.last_action = action
            return self.run_handler.on_handle_action(action)
        except Exception as e:
            raise e

    def handle_step(self, actions, extra_info):
        self.logger.debug(f"kaiwu_environ Handle step, {self.identity}")

        def normalize_extra(in_extra):
            out_extra = {}
            for agent_id, agent_extra in in_extra.items():
                # 只有一个policy的情况，去掉policy层级
                if len(agent_extra) == 1:
                    out_extra[agent_id] = next(iter(agent_extra.values()))
                else:
                    out_extra[agent_id] = agent_extra
            return out_extra

        ret_code, rsp_data = 0, {}
        try:
            rsp_data = self.run_handler.on_update_rsp(actions, extra_info=normalize_extra(extra_info))
            assert all([isinstance(action, bytes) for _, action in rsp_data.items()]), "rsp data should be action dict"
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_update_rsp(self.builder, ret_code, self._ep_id, rsp_data)
            self.send_rsp(RspMsg.RspMsg.update_rsp, rsp)

    def handle_agent_end(self, req):
        client_id, ep_id, agent_id, req_data = KaiwuMsgHelper.decode_agent_end_req(req)

        self.logger.debug(f"kaiwu_environ Handle agent end, {self.identity}, agent_id: {agent_id}")

        assert client_id == self._client_id, \
            f"mismatch client id: {client_id} -> {self._client_id}"
        assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
        assert agent_id in self.curr_agents, f"agent {agent_id} can't be found"

        self.curr_agents.remove(agent_id)

        ret_code = 0
        try:
            self.run_handler.on_agent_end(client_id, ep_id, agent_id, req_data)
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_agent_end_rsp(self.builder, ret_code, ep_id, agent_id)
            self.send_rsp(RspMsg.RspMsg.agent_end_rsp, rsp)

        return {}, {}, {agent_id: True, '_all_done_': False}

    def handle_ep_end(self, req):
        client_id, ep_id, req_data = KaiwuMsgHelper.decode_ep_end_req(req)
        self.logger.debug(f"kaiwu_environ Handle ep end, {self.identity}")
        assert client_id == self._client_id, \
            f"mismatch client id: {client_id} -> {self._client_id}"
        assert ep_id == self._ep_id, f"mismatch ep id: {ep_id} -> {self._ep_id}"
        assert len(self.curr_agents) == 0, f"episode should end after all agents"

        ret_code = 0
        try:
            self.run_handler.on_ep_end(client_id, ep_id, req_data)
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_ep_end_rsp(self.builder, ret_code, self._ep_id)
            self.send_rsp(RspMsg.RspMsg.ep_end_rsp, rsp)
        # return None,None
        return {}, {}, {'_all_done_': True}

    def on_handle_ep_end(self, req):
        # sgame 1v1和5v5结束逻辑
        client_id, ep_id, req_data = KaiwuMsgHelper.decode_ep_end_req(req)
        ret_code = 0
        self.run_handler.on_ep_end(client_id, ep_id, req_data)
        return None, "end"

    def handle_event(self, req):
        client_id, req_data = KaiwuMsgHelper.decode_event_req(req)

        self.logger.debug(f"kaiwu_environ Handle event, {self.identity}")

        assert hasattr(self, 'client_id') and client_id == self._client_id, \
            f"mismatch client id: {client_id} -> {self._client_id}"

        ret_code, rsp_data = 0, b''
        try:
            rsp_data = self.run_handler.on_event(client_id, req_data)
            assert isinstance(rsp_data, bytes), "rsp_data should be bytes"
        except Exception as e:
            ret_code = -1
            raise e
        finally:
            rsp = KaiwuMsgHelper.encode_event_rsp(self.builder, ret_code, rsp_data)
            self.send_rsp(RspMsg.RspMsg.event_rsp, rsp)

    # 从battlesrv --> aisrv的队列里取请求包
    def recv_req(self):
        client_req_data = self.msg_buff.recv_msg()

        try:
            req = Request.Request.GetRootAsRequest(client_req_data, 0)
            self.seqno, msg_type, msg = KaiwuMsgHelper.decode_request(req)
        except Exception as e:
            self.logger.exception(f"kaiwu_environ Error in decoding req, raw msg: {client_req_data}. error: {e}")
            raise e
        
        return msg_type, msg

    # 放入aisrv <--> battlesrv 的队列里响应包
    def send_rsp(self, msg_type, msg):
        rsp = KaiwuMsgHelper.encode_response(self.builder, self.seqno, msg_type, msg)

        self.builder.Finish(rsp)
        msg = self.builder.Output()

        self.send_rsp_to_client(bytes(msg))

        self.builder = flatbuffers.Builder(0)

    def handle_heartbeat(self, req):
        self.logger.debug(f"kaiwu_environ Handle heartbeat, {self.identity}")

        self.send_rsp_to_client(b'')

    # 放入aisrv --> gamecore的队列
    def send_rsp_to_client(self, fb_rsp):
        self.msg_buff.send_msg(fb_rsp)
