#!/usr/bin/env python
# -*- coding: utf-8 -*-


import datetime
import random
import numpy as np
from framework.common.logging.kaiwu_logger import KaiwuLogger
from framework.common.algorithms.expr_processor import RLExprProcessor
from framework.common.config.config_control import CONFIG

def split_seq(seq, time_steps=int(CONFIG.time_steps), pad_value=0, only_keep_first_time_step=False):
    if len(seq) % time_steps > 0:
        pad_width = [(0, 0)] * len(seq.shape)
        pad_width[0] = (0, time_steps - len(seq) % time_steps)
        seq = np.pad(seq, pad_width, mode='constant', constant_values=pad_value)
    seq = seq.reshape((-1, time_steps, *seq.shape[1:]))
    if only_keep_first_time_step:
        seq = seq[:, 0:1]
    return seq

class PPOExpr:
    def __init__(self, state, action, reward, done, vpred, neg_logp, step):
        """
        :param state: State对象
        :param action: Action对象
        :param reward: Reward对象
        :param done: bool
        :param vpred: float
        :param neg_logp: dict类型, key是动作名, value是对应的neg_logp值
        :param step: int
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.vpred = vpred
        self.neg_logp = neg_logp
        self.step = step


class PPOProcessor(RLExprProcessor):
    def __init__(self, simu_ctx, agent_ctx, policy_id):
        super(PPOProcessor, self).__init__(simu_ctx, agent_ctx, policy_id)

        self.logger = KaiwuLogger()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/sample_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'sample')

    def _gen_expr(self):
        pred_output = self._agent_ctx.pred_output[self._policy_id]
        neg_logprobs = {f"old_neg_logp_{action_name}": pred_output[self._agent_ctx.agent_id][f"neg_logprob_{action_name}"]
                        for action_name in self._agent_ctx.action.action_space()}

        return PPOExpr(
            self._agent_ctx.state[self._policy_id],
            self._agent_ctx.action,
            self._agent_ctx.reward,
            self._agent_ctx.done,
            pred_output[self._agent_ctx.agent_id]["v"],
            neg_logprobs,
            pred_output[self._agent_ctx.agent_id]["s"]
        )
    
    '''
    下面是样本生产后的格式:
    {
        'x': array([], shape=(0, 4, 2), dtype=float32), 
        'a': array([], shape=(0, 4, 1), dtype=float32), 
        'old_neg_logp_a': array([], shape=(0, 4), dtype=float32), 
        'y_r': array([], shape=(0, 4), dtype=float32), 
        'old_vpred': array([], shape=(0, 4), dtype=float32), 
        'm': array([], shape=(0, 4), dtype=float32), 
        's': array([], shape=(0, 1), dtype=int64)
    }

    建表语句：
    (TensorSpec(shape=(4,), dtype=tf.int32, name='a'), 
    TensorSpec(shape=(1,), dtype=tf.float32, name='m'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='old_neg_logp_a'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='old_vpred'), 
    TensorSpec(shape=(4, 4), dtype=tf.float32, name='x'), 
    TensorSpec(shape=(4,), dtype=tf.float32, name='y_r'))
    '''
    def _proc_exprs(self):
        local_step = self._agent_ctx.pred_output[self._policy_id][self._agent_ctx.agent_id]["s"]
        policy_conf = self._agent_ctx.policy_conf[self._policy_id]

        total_frame_cnt = len(self._exprs)
        self._exprs = [expr for expr in self._exprs if
                       expr.step >= local_step or random.random() <= CONFIG.expr_skip_rate]
        valid_frame_cnt = len(self._exprs)
        skip_frame_cnt = total_frame_cnt - valid_frame_cnt
        if valid_frame_cnt == 0:
            self.logger.warning(f"sample Got 0 valid frame, maybe all expr.step is lower than local_step({local_step}), "
                                 f"or FLAGS.expr_skip_rate({CONFIG.expr_skip_rate}) is an unexpected value.")
            return {}, valid_frame_cnt, skip_frame_cnt

        x, a, neg_logp, vpred, s, r = {}, {}, {}, [], [], []
        for expr in self._exprs:
            # state
            state_dict = expr.state.get_state()
            for k in state_dict:
                x.setdefault(k, []).append(state_dict[k])

            # action
            action_dict = expr.action.get_action()
            for k in action_dict:
                a.setdefault(k, []).append(action_dict[k])

            # neg_logp
            for k in expr.neg_logp:
                neg_logp.setdefault(k, []).append(expr.neg_logp[k])

            vpred.append(expr.vpred)
            s.append(expr.step)
            r.append(expr.reward.get_reward())

        # convert to numpy array
        for k, v in x.items():
            x[k] = np.array(v, dtype=policy_conf.state.state_space()[k].dtype)

        for k, v in a.items():
            a[k] = np.array(v, dtype=policy_conf.action.action_space()[k].dtype)

        for k, v in neg_logp.items():
            neg_logp[k] = np.array(v, dtype=np.float32)

        vpred = np.array(vpred, dtype=np.float32)
        s = np.array(s, dtype=np.int64)
        r = np.array(r, dtype=np.float32)
        adv = np.zeros_like(r)
        y_r = np.zeros_like(r)
        m = np.ones_like(r)  # mask

        done = self._agent_ctx.done
        if done:
            vpred[-1] = r[-1]

        lastgaelam = 0.0
        for i in reversed(range(len(self._exprs) - 1)):
            delta = r[i] + float(CONFIG.ppo_gamma) * vpred[i + 1] - vpred[i]
            adv[i] = lastgaelam = delta + float(CONFIG.ppo_gamma) * float(CONFIG.ppo_lam) * lastgaelam
        y_r = adv + vpred

        train_data = {}
        if not CONFIG.use_rnn:
            train_data.update({k: v[:-1] for k, v in x.items()})
        else:
            for k, v in x.items():
                only_keep_first = True if k in CONFIG.rnn_states else False
                new_v = split_seq(v[:-1], only_keep_first_time_step=only_keep_first)
                train_data[k] = new_v

        train_data.update({k: v[:-1] if not CONFIG.use_rnn else split_seq(v[:-1])
                           for k, v in a.items()})
        train_data.update({k: v[:-1] if not CONFIG.use_rnn else split_seq(v[:-1])
                           for k, v in neg_logp.items()})
        train_data.update({
            "y_r": y_r[:-1] if not CONFIG.use_rnn else split_seq(y_r[:-1]),
            "old_vpred": vpred[:-1] if not CONFIG.use_rnn else split_seq(vpred[:-1]),
            "m": m[:-1] if not CONFIG.use_rnn else split_seq(m[:-1]),
            "s": s[:-1].reshape(-1) if not CONFIG.use_rnn else np.amin(split_seq(s[:-1], pad_value=local_step),
                                                                      axis=1).reshape(-1, 1),
        })


        return train_data, valid_frame_cnt - 1, skip_frame_cnt + 1
