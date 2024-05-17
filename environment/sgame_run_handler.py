#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project :1v1 
@File    :sgame_run_handler.py
@Author  :kaiwu
@Date    :2022/5/20 17:43 

'''
import os
import struct
import traceback
import numpy as np
import flatbuffers
from conf.config import ModelConfig
from framework.interface.run_handler import RunHandler
from state_action_reward.sgame_state import SgameState
from environment.protocl.command_pb2 import COMMAND_TYPE_None
from framework.server.aisrv.flatbuffer.kaiwu_msg import *
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from environment.protocl.proto_king.sgame_ai_server_pb2 import AIServerResponse, \
    AICommandInfo, KaiwuAIServerRequest, KaiwuAIServerResponse, SingleReq, E_INIT, \
    E_FRAME
from tools.app.game_render import GameRender
from environment.feature_process.sgame_interface import SgameInterface
from framework.common.config.config_control import CONFIG
from sample_processor.sgame_sample_processor import SgameSampleProcessor

class SgameRunHandler(RunHandler): 
    """
    Sgame's environment belongs to the business section and will be loaded and processed by the AIServer.

    Sgame's environ 属于业务部分,会由AIServer加载, 进行处理
    """
    HERO_ID_INDEX_DICT = {
        112: 0,
        121: 1,
        123: 2,
        131: 3,
        132: 4,
        133: 5,
        140: 6,
        141: 7,
        146: 8,
        150: 9,
        154: 10,
        157: 11,
        163: 12,
        169: 13,
        175: 14,
        182: 15,
        193: 16,
        199: 17,
        510: 18,
        513: 19,
    }

    def __init__(self, simu_ctx):
        super(SgameRunHandler, self).__init__(simu_ctx)

        self.render = None
        self.first_dump = False

        self.game_id = None
        self.player_num = None
        self._client_version = None
        self._client_id = None

        self.player_ids = {}
        self.camp_list = []
        self.player_camp = {}
        self.is_gameover = False

        # Utilize the framework's provided msg_buff for message sending.
        # 利用框架提供的msg_buff进行消息发送
        self._act_que = simu_ctx.msg_buff
        self.first_frame = True
        self.first_process = True
        self.cur_frame_no = 0
        self.cur_state = None

        # The number of frames required to collect a set of data.
        # 采集一份数据需要的帧数
        self.predict_frequency = CONFIG.frame_interval
        self.hero_list = []

        self.policy_name = CONFIG.policy_name

        # Instantiate the sample production class.
        # 实例化样本生产类
        self.sample_processor = SgameSampleProcessor()
        self.sample_processor.set_logger(simu_ctx.logger)

        # flatbuffers
        self.builder = flatbuffers.Builder(0)

         # log
         # 日志
        self.logger = simu_ctx.logger

        self.lstm_unit_size = ModelConfig.LSTM_UNIT_SIZE
        
        # Feature extraction.
        # 特征值抽取
        self.sgame_eigent_value_process = SgameInterface(CONFIG.feature_process_lib_interface_configure, self.logger)
        self.logger.info(f'run_handle lib interface configure file {CONFIG.feature_process_lib_interface_configure}')
        
        self.logger.info("run_handle env init success")
        

    def _constuct_rsp_pb(self, id):
        response = AIServerResponse()
        player_id = self.player_list[id]
        # self.logger.debug("run_handle Construct player_id is {}".format(player_id))
        
        cmd_list = AICommandInfo()
        cmd_list.player_id = player_id
        cmd_list.cmd_info.command_type = COMMAND_TYPE_None
        response.cmd_list.extend([cmd_list])
        return response

    def _get_kaiwu_rsp_pb(self, rsp_pbs):
        kaiwu_response = KaiwuAIServerResponse()
        kaiwu_response.rsp_pbs.extend(rsp_pbs)
        kaiwu_response.code = 0
        return kaiwu_response

    def game_over_process(self):
        self.logger.info("run_handle GameOver, not send rsp")
        assert self._act_que.qsize() == 0

        # returne GameOver
        # 返回GameOver
        smsg = self.get_gameover_pb()
        self.put_rsp_queue(smsg)
        if self.first_dump:
            try:
                if self.render:
                    self.render.dump_one_round()
            except Exception as e:
                traceback.print_exc()
            else:
                self.first_dump = False

    # The logic for each frame.
    # 每一帧的逻辑
    def on_update_req(self, client_id, ep_id, req_data):
        """
        Args:
            client_id: Identifies the client ID number.
            ep_id: Identifies the episode ID, optional.
            req_data: Main frame data, as a dictionary (key: agent_id, value: bytes_data).
        
        参数：
            client_id: 标识客户端id号
            ep_id: 标识epiosde_id, 可以不用
            req_data: 主要帧数据, 为dict(key:agent_id,value:bytes_data)
        """
        datas = []
        for agent_id in req_data:
            sing_req = SingleReq()
            sing_req.ParseFromString(req_data[agent_id][0])
            datas.append(sing_req)
        
        return self.frame_process(datas)
    
    def update_lstm(self,lstm_cell_list,lstm_hidden_list):
        for i in range(self.player_num):
            self.lstm_cell_list[i] = lstm_cell_list[i].flatten()
            self.lstm_hidden_list[i] = lstm_hidden_list[i].flatten()
    
    @property
    def done(self):
        return self.is_gameover

    def frame_process(self, datas: [SingleReq]):
        # self.logger.debug("run_handle frame_process")
        self.process_msg(datas)

        if self.cur_frame_no>1000 and self.cur_frame_no%3000 == 0:
            if self.render:
                self.render.dump_one_round()
                self.logger.debug("run_handle Dump success")
            
            if CONFIG.use_game_render:
                self.render = GameRender(dump_path=CONFIG.replay_dump_path)
                self.render.reset(self.game_id)

        status, states = self._step_feature(datas, first_frame=self.first_frame)
        # self.logger.debug("run_handle 帧{}处理完毕".format(self.cur_frame_no))
        if status is False:
            # If the state is False, do not make any predictions.
            # 如果状态为False,则不进行预测
            # self.logger.debug(f'run_handle 帧{self.cur_frame_no} 不需要预测, 直接返回')
            return False, None, None
        
        self.first_frame = False
        assert states is not None
        self._update_gameover(states, datas)

        # add lstm in states
        # states中增加lstm
        for i in range(self.player_num):
            states[i]["lstm_hidden"] = self.lstm_hidden_list[i]
            states[i]["lstm_cell"] = self.lstm_cell_list[i]
        state_dicts = {}
        for i in range(self.player_num):
            s = SgameState(states[i])
            state_dicts[i] = s

            player_id = self.player_list[i]
            camp = self.player_camp.get(player_id)
            states[i]["hero_camp"] = camp

        # Simplify the communication logic for game over.
        # 简化游戏结束的通信逻辑
        if self.is_gameover:
            self.game_over_process()

            return True, state_dicts, states

        return True, state_dicts, states
    
    def on_handle_action(self, rp_actions_list):
        """
        Process the action and generate the response required by the BattleServer.
        Args:
            rp_actions_list: The Actor predicts and returns a list of actions.
        
        处理action, 生成BattleServer需要的Response
        参数:
            rp_actions_list: Actor预测返回的action的list
        """
        # self.logger.debug(f"run_handle 开始处理action")

        s_msgs = [None] * self.player_num
        for id in range(self.player_num):
            rp_actions = rp_actions_list[id]
            ret = self.sgame_eigent_value_process.result_proces(rp_actions, id)
            assert ret[0] != 0, "process action failed: {}".format(ret[1])

            rsp_pb = AIServerResponse()
            rsp_pb.ParseFromString(ret[1])
            if (rsp_pb.gameover_ai_server):
                self.logger.error("run_handle step_action send gameover!")
            s_msgs[id] = rsp_pb

        self.put_rsp_queue(s_msgs)
        # self.logger.debug("成功放入outputq, run_handle 放入action的RSP{}",str(s_msgs))

    def init(self, client_id, data: KaiwuAIServerRequest):
        self.first_construct = True
        self.is_gameover = False
        self.first_dump = True

        self._client_id = client_id
        self.player_num = data.player_num
        self.game_id = data.game_id
        self.player_list = [None] * self.player_num
        game_info = data.game_info
        self.hero_list.extend(game_info.player_info)
        if CONFIG.use_game_render:
            self.render = GameRender(dump_path=CONFIG.replay_dump_path)
            self.render.reset(self.game_id)

        # Initialization of the sample generation class.
        # 样本生产类初始化
        self.sample_processor.on_init(player_num=self.player_num, game_id=self.game_id)

        # The initialization message does not require a response.
        # Init的消息不需要返回消息
        self.lstm_cell_list = [np.zeros([self.lstm_unit_size])]*self.player_num
        self.lstm_hidden_list = [np.zeros([self.lstm_unit_size])]*self.player_num

        self.logger.info("run_handle init success")

    def on_init(self, client_id, req_data):
        data = KaiwuAIServerRequest()
        data.ParseFromString(req_data)
        msg_type = data.msg_type
        if msg_type == E_INIT:
            self.init(client_id, data)
        elif msg_type == E_FRAME:
            self.logger.error("run_handle Error for msg_type")
            raise Exception
            # self.frame_process(data)


    def _step_feature(self, data: [SingleReq], first_frame=True):
        ret_flag = [False] * self.player_num
        ret_num = 0
        real_num = 0
        states = [None] * self.player_num
        self.init_pb = [None] * self.player_num
        s_msgs = [None] * self.player_num
        single_req_list = data
        req_pbs = []
        for id in range(self.player_num):
            real_num += 1
            req_pb: SingleReq = single_req_list[id]
            req_pbs.append(req_pb.ai_req)
            self.cur_frame_no = req_pb.ai_req.frame_no
            if first_frame:
                self.start_frame = self.cur_frame_no

            if req_pb.ai_req.gameover:
                self.logger.info('run_handle game is over!')
                self.is_gameover = True

            if (not first_frame) and (self.cur_frame_no - self.start_frame) % self.predict_frequency > 0:
                '''
                skip this non-predict frame
                if gameover, skip c++ process, so checking gameover with py code is necessary.
                '''
                if not self.is_gameover:
                    s_msgs[id] = self._constuct_rsp_pb(id)
                    continue


            length, req_type, seq_no, obs = self.parse_feature(req_pb)

            # feature processing
            # 特征值处理
            ret = self.sgame_eigent_value_process.feature_process(length, req_type, seq_no, obs, id)

            # Failed, return no action

            if ret[0] == 0:
                assert False, "step failed: {}".format(ret[1])
            if ret[0] == 1:
                assert False, "Parsing gameover information, receive msg again!"

            elif ret[0] == 2:
                state = self._state_tuple2np(ret[1:])[0]
                # self.logger.debug("run_handle state has ret")

                if req_pb.ai_req.gameover:
                    self.logger.info('run_handle gameover at frameno {} of {}!'.format(req_pb.ai_req.frame_no, id))
                ret_num += 1
                #state["req_pb"] = req_pb.ai_req
                state["frame_no"] = self.cur_frame_no
                for hero in req_pb.ai_req.frame_state.hero_states:
                    if hero.actor_state.runtime_id == self.player_list[id]:
                        state["is_train"] = True if hero.actor_state.hp > 0 else False
                states[id] = state
                ret_flag[id] = True
                s_msgs[id] = None

            elif ret[0] == 3:
                rsp_pb = AIServerResponse()
                rsp_pb.ParseFromString(ret[1])
                self.init_pb[id] = ret[1]
                s_msgs[id] = rsp_pb
        
        # self.logger.debug(f"run_handle 处理帧:{self.cur_frame_no}")

        if self.render:
            self.render.draw_frame(req_pbs, self.cur_frame_no)
        if first_frame:
            if ret_num < real_num:
                self.put_rsp_queue(s_msgs, ignore_none=True)
                return False, None
        elif (self.cur_frame_no - self.start_frame) % self.predict_frequency > 0:
            if not self.is_gameover:
                self.put_rsp_queue(s_msgs)
                return False, None
        return True, states

    def _init_hero_info(self, pb, id):
        if self.player_list[id] is not None:
            return

        if 0 < len(pb.cmd_list):
            # set current player
            for cmd in pb.cmd_list:
                self.player_list[id] = cmd.player_id

            # get camp info
            for hero_state in pb.frame_state.hero_states:
                if hero_state.actor_state.runtime_id == self.player_list[id]:
                    self.player_ids.setdefault(hero_state.actor_state.camp, []).append(
                        hero_state.actor_state.runtime_id)
                    self.camp_list.append(hero_state.actor_state.camp)
                    self.player_camp[hero_state.actor_state.runtime_id] = hero_state.actor_state.camp

    def process_msg(self, data: [SingleReq]):
        """
        args:
            data: 2 pb in SingleReq
        
        参数:
            data: SingleReq存在两个pb
        """
        single_req_list = data
        if self.first_construct:
            # self.logger.debug("run_handle act_que为空, 构造rsp")
            for id, pb in enumerate(single_req_list):
                aisrv_reqpb = pb.ai_req
                self._init_hero_info(aisrv_reqpb, id)

            self.first_construct = False

    def parse_feature(self, req: SingleReq):
        req_type = req.req_type
        seq_no = req.seq_no
        pid = req.pid
        ai_req = req.ai_req
        msg = ai_req.SerializeToString()
        obs = bytearray(len(msg) + 4)
        obs[:4] = struct.pack('I', pid)
        obs[4:] = msg
        length = len(obs)
        obs = bytes(obs)
        return length, req_type, seq_no, obs

    def _state_tuple2np(self, states):
        states = list(states)
        for j, state in enumerate(states):
            if state is None:
                continue
            for k in state:
                if isinstance(state[k], tuple) and k in ["legal_action"]:
                    state[k] = np.array(state[k])
                if isinstance(state[k], tuple) and k in ["observation"]:
                    state[k] = np.array(state[k])
                    hero_id = 123
                    hero_id_vec = np.zeros([20, ], dtype=np.float)
                    hero_id_vec[self.HERO_ID_INDEX_DICT[hero_id]] = 1
                    # print("-------hero_id-------",hero_id_vec)

                    state[k] = np.concatenate((state[k], hero_id_vec), axis=0)
                if isinstance(state[k], dict) and k in ["sub_action_mask"]:
                    for i in state[k]:
                        state[k][i] = np.array(state[k][i])

        return states

    def get_gameover_pb(self):
        rsp_pbs = []
        for i in range(self.player_num):
            rsp = self._constuct_rsp_pb(i)
            rsp.gameover_ai_server = 1
            rsp_pbs.append(rsp)

        return rsp_pbs

    def put_rsp_queue(self, s_msgs, ignore_none=True):
        rsp_pbs = []
        for id in range(self.player_num):
            if s_msgs[id] is None:
                if ignore_none:
                    rsp_pbs.append(self._constuct_rsp_pb(id))
                else:
                    assert False, "send None at {}".format(id)
            else:
                rsp_pbs.append(s_msgs[id])
        
        kaiwu_rsp = self._get_kaiwu_rsp_pb(rsp_pbs)
        
        self._act_que.put(kaiwu_rsp.SerializeToString())
        # self.logger.debug("run_handle 放入Step的Rsp")

    def _update_gameover(self, states, datas:[SingleReq]):
        single_req_list = datas
        really_done = False
        for i in range(self.player_num):
            req_pb: SingleReq = single_req_list[i]
            if req_pb.ai_req.gameover:
                self.logger.info("run_handle frame_no {} req_pb gameover".format(req_pb.ai_req.frame_no))
            states[i]["done"] = states[i]["done"] or req_pb.ai_req.gameover
            really_done = really_done or states[i]["done"]
        self.is_gameover = really_done
        if self.is_gameover:
            self.logger.info("run_handle GameOver in Update")

    def policy_mapping_fn(self, agent_id):
        return self.policy_name

