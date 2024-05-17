#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import multiprocessing
import time
import traceback
import datetime
import lz4.block
import schedule
import datetime
import numpy as np
import collections
import pickle
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.common_func import set_schedule_event
from framework.common.monitor.monitor_proxy import MonitorProxy

if CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
    from app.sgame_5v5.sample_processor.sgame_expr import SgameExpr as RLDataInfo
    from app.sgame_5v5.algo.config import Config, ModelConfig
from framework.interface.sample_processor import SampleProcessor


"""样本处理相关类"""
class SgameSampleProcessor(SampleProcessor):
    __slots__ = ("_data_shapes", "_LSTM_FRAME", "network_sample_info","gamma","lamda","agent_policy","tmp_data",
                 "last_data", "log_rewsum", "steps","logger", "game_id", "m_task_id", "m_task_uuid", "num_agents",
                 "rl_data_map", "m_replay_buffer")
    
    def __init__(self):

        #self.simu_ctx = simu_ctx

        self._data_shapes = ModelConfig.data_shapes_for_sample
        self._LSTM_FRAME = ModelConfig.LSTM_TIME_STEPS

        # 由于aisrv生产样本时需要network_sample_info
        self.network_sample_info = None

        # load config from config file
        self.gamma=np.array([0.997,0.997,0.997,0.997,0.99978,0.997])
        self.lamda= np.array([0.95,0.95,0.95,0.95,0.99856,0.95])
        self.agent_policy= []
        
        self.tmp_data = None
        self.last_data = None
        
        self.log_rewsum = 0.0
        
        self.steps = 0
    
    '''
    框架提供了日志接口, 业务直接使用即可
    '''
    def set_logger(self, logger):
        self.logger = logger

    '''
    sample manager init 处理
    '''
    def on_init(self, player_num, game_id):
        self.game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = player_num
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

        self.log_rewsum = 0.0
        self.logger.info(f"sample sample on_init success, game_id {self.game_id}, num_agents {self.num_agents}")
    
    def should_train(self):
        return True

    def gen_expr(self, must_need_sample_info, network_sample_info,sample_model_version=None, current_model_version=None):
        """
        生成一个样本
        Args:
            must_need_sample_info: 5v5中使用较少
            network_sample_info: Actor预测会返回action和网络参数, network_sample_info为样本需要的信息
        Returns:

        """
        self.network_sample_info = network_sample_info
        state_dict = must_need_sample_info
        frame_no = state_dict['frame_no']
        done = state_dict['done']

        for i in range(self.num_agents):
            feature_vec, lstm_hidden, lstm_cell, reward, value, legal_action, sub_action_mask, action, prob, is_trains = network_sample_info[i]

            # 将feature压缩存储释放内存
            feature_vec = pickle.dumps(feature_vec)
            feature_vec = lz4.block.compress(feature_vec, mode='fast', store_size=False)
            
            meta_is_trains = np.zeros(5)
            keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "value", "prob", "sub_action",
                    "lstm_cell", "lstm_hidden", "done", "is_train", "meta_is_train")
            #目前网络中没有增加大局观reward，在这个地方加入全0向量
            reward_mgg = np.array([[0], [0], [0], [0], [0]])
            reward = np.hstack((reward, reward_mgg))

            values = (frame_no, feature_vec, legal_action, action, reward, value, prob, sub_action_mask,
                      lstm_cell, lstm_hidden, done, is_trains, meta_is_trains)
            sample = dict(zip(keys, values))

            # TODO:只有最新的Model，才能产生Sample
            self.save_sample(**sample, agent_id=i, game_id=self.game_id, uuid=None)
        
        self.steps += 1
        self.logger.debug(f"game_id {self.game_id}, sample gen_expr success")


    def proc_exprs(self, del_last=False):
        """
        生成固定步数的非全量样本
        Returns: train_data_all
        """
        total_frame_cnt = len(self.network_sample_info)
        
        #异常情况需要删除最后保存的样本来保证样本的正确性
        if del_last:
            for i in range(self.num_agents):
                if len(list(self.rl_data_map[i].keys())) > 0:
                    last_key = list(self.rl_data_map[i].keys())[-1]
                    self.rl_data_map[i].pop(last_key)

        #else:
        #    for i in range(self.num_agents):
        #        feature_vec, lstm_hidden, lstm_cell, reward, value, legal_action, sub_action_mask, action, prob, is_trains = self.network_sample_info[i]
        #        # 目前网络中没有增加大局观reward，在这个地方加入全0向量，5个hero均为0
        #        reward_mgg = np.array([[0], [0], [0], [0], [0]])
        #        reward = np.hstack((reward, reward_mgg))
        #        self.save_last_sample(agent_id=i, reward=reward)
        
        #保存最后一个样本，作为reset后的rl_data_map的第一个
        if len(list(self.rl_data_map[0].keys())) > 0:
            self.last_data = [list(self.rl_data_map[i].items())[-1] for i in range(self.num_agents)]
        else:
            self.last_data = None
        #保存尾部16个样本，方便最后样本数量不足16个时生成lstm的完整样本
        if len(list(self.rl_data_map[0].keys())) > self._LSTM_FRAME:
            self.tmp_data = [list(self.rl_data_map[i].items())[-self._LSTM_FRAME:-1] for i in range(self.num_agents)]
        else:
            self.tmp_data = None

        #最后一个样本已经保存，则不参与样本生成，不确定不对正常结束对局的最后一帧进行训练是否会有影响，后面可以加个flag进行判断
        for i in range(self.num_agents):
            if len(list(self.rl_data_map[i].keys())) > 0:
                last_key = list(self.rl_data_map[i].keys())[-1]
                self.rl_data_map[i].pop(last_key)
        
        #样本生成
        train_data = self.send_samples()
        return_rew = self.log_rewsum
        #清理旧样本
        self.reset()
        
        # 对train_data进行压平处理
        train_data_all = []
        for agent_data in train_data:
            # agent_data:list[(frame_no,vec)]
            for sample in agent_data:
                train_data_all.append({
                    # 发送样本时, 强制转换成float16
                    'input_datas': np.array(sample[1], dtype=np.float16)
                })
        train_frame_cnt = len(train_data)
        drop_frame_cnt = total_frame_cnt - train_frame_cnt
        self.logger.info(f'game_id {self.game_id}, sample train_frame_cnt {train_frame_cnt},  drop_frame_cnt {drop_frame_cnt}, reward {return_rew}')
        return train_data_all, train_frame_cnt, return_rew

    def reset(self):
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        assert self.last_data, "last_data error!"
        for i in range(self.num_agents):
            self.rl_data_map[i][self.last_data[i][0]] = self.last_data[i][1]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]
        
        self.log_rewsum = 0.0
        self.logger.info(f"game_id {self.game_id}, sample already reset")

    def save_sample(self, frame_no,
                    vec_feature, legal_action, action, reward, value, prob, sub_action,
                    lstm_cell, lstm_hidden,
                    done, agent_id, is_train=True, meta_is_train=False,
                    game_id=None, uuid=None):
        """
        samples must saved by frame_no order
        """
        reward = self._clip_reward(reward)
        #只需要上报id=0的reward sum
        if agent_id==0:
            self.log_rewsum += np.sum(reward)
        rl_data_info = RLDataInfo()

        #value = value.flatten()[0]
        #lstm_cell = lstm_cell.flatten()
        #lstm_hidden = lstm_hidden.flatten()

        # update last frame's next_value
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = value
            last_rl_data_info.reward = reward

        # save current sample

        rl_data_info.frame_no = frame_no
        rl_data_info.feature = vec_feature
        rl_data_info.legal_action = legal_action
        rl_data_info.reward = np.zeros([5, 6])
        rl_data_info.value = value
        rl_data_info.done = done
        rl_data_info.lstm_info = np.concatenate([lstm_cell, lstm_hidden], axis=-1)

        # np: (5， 14+25+42+42+3+61)
        rl_data_info.prob = prob
        # np: (5,6)
        # rl_data_info.action = 0 if action < 0 else action
        rl_data_info.action = action
        # np: (5,6)
        rl_data_info.sub_action = sub_action
        # np: (5)
        rl_data_info.is_train = is_train
        # np: (5)
        rl_data_info.meta_is_train = meta_is_train

        self.rl_data_map[agent_id][frame_no] = rl_data_info
        
    def save_last_sample(self, reward, agent_id):
        self.logger.info(f"game_id {self.game_id}, sample save last sample")
        if len(self.rl_data_map[agent_id]) > 0:
            # TODO: is_action_executed, last_gamecore_act
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = np.zeros([5, 6])
            last_rl_data_info.reward = reward

    def send_samples(self):
        self._calc_reward()
        self._format_data()

        return self._send_game_data()

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = np.zeros([5, 6])
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        sample_one_size = sample_batch.shape[1]
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0]:] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx: s_idx + split_shape[0]] = sample_batch[:, idx: idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample

    def common_sample_process(self, cnt, rl_info, sample_batch, sample_one_size):
        # serilize one frames
        idx, dlen = 0, 0
        # vec_data
        vec_feature = rl_info.feature
        vec_feature = lz4.block.decompress(vec_feature, uncompressed_size=CONFIG.lz4_uncompressed_size)
        vec_feature = pickle.loads(vec_feature, encoding='bytes')
        dlen = vec_feature.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = vec_feature
        idx += dlen

        # legal_action
        dlen = rl_info.legal_action.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = rl_info.legal_action
        idx += dlen

        # reward_sum & advantage
        dlen = rl_info.reward.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = rl_info.reward_sum
        idx += dlen
        #advantage最后一维是大局观，不参与policy训练才可以
        #多头相加得到advantage
        sample_batch[:, cnt, idx] = np.sum(rl_info.advantage[:, :-1], axis=-1)
        idx += 1

        # labels
        dlen = rl_info.action.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = rl_info.action
        idx += dlen

        # probs (neg log pi->prob)
        dlen = rl_info.prob.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = rl_info.prob
        idx += dlen

        # is_train
        sample_batch[:, cnt, idx] = rl_info.is_train
        idx += 1

        sample_batch[:, cnt, idx] = rl_info.meta_is_train
        idx += 1

        # sub_action
        dlen = rl_info.sub_action.shape[1]
        sample_batch[:, cnt, idx:idx + dlen] = rl_info.sub_action
        idx += dlen

        assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)
        return sample_batch

    def _format_data(self):
        # feature   legal_action    reward  advantage action_list  prob_list   frame_is_train meta_is_train  weight_list
        #self._data_shapes = [12667, 187, 6, 1, 6, 187, 1, 1, 6]

        sample_one_size = np.sum(self._data_shapes[:-2])
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([5, self._LSTM_FRAME, sample_one_size])
        sample_lstm = np.zeros([5, sample_lstm_size])
        first_frame_no = -1

        #本次发送样本数量小于lstm_step时,利用tmp_data进行样本拼凑保留尾部样本
        if len(list(self.rl_data_map[0].keys())) < self._LSTM_FRAME and self.tmp_data:
            data_len = len(list(self.rl_data_map[0].keys()))
            tmp_len = self._LSTM_FRAME - data_len
            for i in range(self.num_agents):
                cnt = 0
                for j in range(-tmp_len, 0):
                    rl_info = self.tmp_data[i][j][1]

                    if cnt == 0:
                        # lstm cell & hidden
                        first_frame_no = rl_info.frame_no
                        sample_lstm = rl_info.lstm_info
                        
                    sample_batch = self.common_sample_process(cnt, rl_info, sample_batch, sample_one_size)

                    cnt += 1
                    
                for j in self.rl_data_map[i]:
                    rl_info = self.rl_data_map[i][j]
                    sample_batch = self.common_sample_process(cnt, rl_info, sample_batch, sample_one_size)
                    cnt += 1
                    
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample = sample_batch.reshape((5, -1))
                    sample = np.concatenate([sample, sample_lstm], axis=-1)
                    sample = sample.reshape((-1))
                    self.m_replay_buffer[i].append((first_frame_no, sample))
                    # self.logger.debug(f'sample first_frame_no {first_frame_no} add sample success')
                else:
                    self.logger.debug(f'game_id {self.game_id}, sample nums is wrong!')
                    raise NotImplementedError
            return

        #正常处理样本逻辑
        for i in range(self.num_agents):
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]

                if cnt == 0:
                    # lstm cell & hidden
                    first_frame_no = rl_info.frame_no
                    sample_lstm = rl_info.lstm_info

                sample_batch = self.common_sample_process(cnt, rl_info, sample_batch, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample = sample_batch.reshape((5, -1))
                    sample = np.concatenate([sample, sample_lstm], axis=-1)
                    sample = sample.reshape((-1))
                    self.m_replay_buffer[i].append((first_frame_no, sample))
                    self.logger.debug(f'game_id {self.game_id}, sample first_frame_no {first_frame_no} add sample success')
            ''' 
            #尾部样本不丢弃
            if cnt > 0:
                a_ins = len(list(self.rl_data_map[i].keys()))-self._LSTM_FRAME
                e_ins = len(list(self.rl_data_map[i].keys()))
                cnt = 0
                if a_ins >= 0:
                    for k in range(a_ins, e_ins):
                        rl_info = self.rl_data_map[i][list(self.rl_data_map[i].keys())[k]]

                        if cnt == 0:
                            # lstm cell & hidden
                            first_frame_no = rl_info.frame_no
                            sample_lstm = rl_info.lstm_info

                            sample_batch = self.common_sample_process(cnt, rl_info, sample_batch, sample_one_size)

                        cnt += 1
                        if cnt == self._LSTM_FRAME:
                            cnt = 0
                            sample = sample_batch.reshape((5, -1))
                            sample = np.concatenate([sample, sample_lstm], axis=-1)
                            sample = sample.reshape((-1))
                            self.m_replay_buffer[i].append((first_frame_no, sample))
            '''

    def _clip_reward(self, reward, max=100, min=-100):
        reward = np.clip(reward, min, max)
        return reward

    # send game info like: ((size, data))*5:
    # [task_id: int, task_uuid: str, game_id: str, frame_no: int, real_data: data in str]

    def _send_game_data(self):
        all_samples = []
        
        # 不保存self-play时用旧模型训练的样本
        for i in range(self.num_agents):
            if not (CONFIG.self_play and self.agent_policy[i] == CONFIG.self_play_old_policy):
                all_samples.append(self.m_replay_buffer[i])
        
        # self.logger.info(f"sample send game data {all_samples}")
        return all_samples


class SampleServer(multiprocessing.Process):
    def __init__(self, learner_proxy_list) -> None:
        super(SampleServer, self).__init__()

        '''
        数据流:
        1. kaiwu_rl_helper线程放入该队列
        2. SampleServer将队列里的数据发送给learner_proxy
        '''
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        # sample_server需要learner_proxy_list
        self.learner_proxy_list = learner_proxy_list

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)
        
        # 对局样本管理: slot_id, sample_processor
        self.game_manager = {}
        
        self.log_rewsum = multiprocessing.Value('f', 0.0)
    
    def gen_frame_sample(self, slot_id, sample_info_list, must_need_sample_info):
        while self.msg_queue.full():
            time.sleep(0.01)
        
        self.msg_queue.put((slot_id, sample_info_list, must_need_sample_info))
    
    def sample_server_gameover(self, slot_id):
        
        while self.msg_queue.full():
            time.sleep(0.01)
        
        self.msg_queue.put((slot_id, None, None))

        return self.log_rewsum.value
    
    def add_policy_to_sample_server(self, slot_id, main_id):
        while self.msg_queue.full():
            time.sleep(0.01)
        
        self.msg_queue.put((slot_id, main_id))
    
    def sample_server_stat_reset(self):
        self.send_to_learner_proxy_suc_cnt = 0
        self.send_to_learner_proxy_err_cnt = 0
    
    def sample_server_stat(self):
        if int(CONFIG.use_prometheus):
            monitor_data  = {
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT : self.send_to_learner_proxy_suc_cnt,
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT : self.send_to_learner_proxy_err_cnt, 
            }

            self.monitor_proxy.put_data(monitor_data)

            # 指标重置
            self.sample_server_stat_reset()

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/sample_server_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'sample_server')
        self.logger.info(f'sample_server start at pid {pid}', g_not_server_label)

        self.process_run_count = 0

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.sample_server_stat)

        self.send_to_learner_proxy_suc_cnt = 0
        self.send_to_learner_proxy_err_cnt = 0
        self.recv_from_kaiwu_rl_helper_suc_cnt = 0
        self.recv_from_kaiwu_rl_helper_err_cnt = 0

    def send_to_learner_proxy(self, train_data, slot_id):

        # 发送给learner_proxy
        learn_index = slot_id % len(self.learner_proxy_list)
        self.learner_proxy_list[learn_index].put_data(slot_id, train_data)
        self.send_to_learner_proxy_suc_cnt += 1

    def run_once(self):
        # 从队列里获取数据进行样本保存处理
        try:
            data = self.msg_queue.get()
            if not data:
                return
        
            # add_policy
            if len(data)==2:
                slot_id, policy_id = data
                if slot_id not in self.game_manager:
                    self.game_manager[slot_id] = SgameSampleProcessor()
                    self.game_manager[slot_id].set_logger(self.logger)
                    # 5v5只支持self-play，因此agent为2
                    self.game_manager[slot_id].on_init(2, slot_id)
                self.game_manager[slot_id].agent_policy.append(policy_id)
                
            elif len(data)==3:
                slot_id, sample_info_list, must_need_sample_info = data
                # 处理gameover的情况
                if sample_info_list==None and must_need_sample_info==None:
                    train_data, train_frame_cnt, _ = self.game_manager[slot_id].proc_exprs()
                    if train_frame_cnt > 0:
                        self.send_to_learner_proxy(train_data, slot_id)            
                    del self.game_manager[slot_id]
                else:
                    # 保存样本
                    self.game_manager[slot_id].gen_expr(must_need_sample_info, sample_info_list)
            else:
                raise NotImplementedError
        
            # 轮询进行leanrer样本发送
            for i in self.game_manager:
                if self.game_manager[i].steps>0 and self.game_manager[i].steps%int(CONFIG.send_sample_size) == 0:
                    train_data, train_frame_cnt, _ = self.game_manager[i].proc_exprs()
                    if train_frame_cnt > 0:
                        self.send_to_learner_proxy(train_data, i)

            if 0 in self.game_manager:
                self.log_rewsum.value = self.game_manager[0].log_rewsum

        except Exception as e:
            pass
        
        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()

    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(f'sample_server run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)

    '''
    进程停止函数
    '''
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('sample_server SampleServer stop success', g_not_server_label)
        
        
        
