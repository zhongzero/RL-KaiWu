#!/usr/bin/env python
# -*- coding: utf-8 -*-


from framework.common.utils.tf_utils import *
from framework.common.algorithms.model import ModeKeys
from framework.common.algorithms.model import Model
from framework.common.algorithms.model import ModelSpec
from framework.common.config.config_control import CONFIG
from framework.common.algorithms.ckpt_saver_hook import CheckpointSaverHook, CkptSaverListener

PPODefaultConfig = {
    'ppo_gamma': 0.99,
    'ppo_lam': 0.95,
    'ppo_ent_coef': 0.01,
    'ppo_vf_coef': 0.5,
    'ppo_pg_coef': 1,
    'ppo_epsilon': 1e-5,
    'ppo_mini_batch_count': 96,
    'ppo_clip_range': 0.2,
    'ppo_end_clip_range': 0.1
}

'''
默认采用的PPO算法, 业务需要自己定制则业务侧自己实现, 再修改下配置文件, 框架加载配置文件里的类
'''
class PPO(Model):
    def __init__(self, network, name='ppo', srv_name=None):
        super().__init__(network, name)

    def build_model(self, mode, input_tensors):
        # 使用绝对偏移计算decay
        relative_step = False
        self.step = self.current_step(
            CONFIG.restore_dir + f"/{CONFIG.app}_{CONFIG.algo}",
            relative_step
        )

        self.state_tensors = {state_name: input_tensors[state_name]
                              for state_name in self.network.state_space}

        self._build_policy(input_tensors, mode)

        # actor上使用的
        if mode == ModeKeys.PREDICT:
            return ModelSpec(
                predict_input=self.state_tensors,
                predict_output={
                    **self.sample_action,
                    **self.sample_neg_logprob_action,
                    'v': self.logits_v,
                    **self.extra_tensors
                }
            )

        self._build_loss(input_tensors)

        self._build_optimizer()

        self._build_summary()

        # learner上使用的
        if mode == ModeKeys.TRAIN:
            return ModelSpec(
                train_input={
                    **self.state_tensors,
                    **self.action_dict,
                    **self.old_neg_logp_dict,
                    'y_r': self.y_r,
                    'adv': self.adv,
                    'old_vpred': self.old_vpred,
                },
                loss=self.cost_all,
                optimizer=self.opt
            )

        raise ValueError("invalid mode keys %s" % mode)
    
    '''
    采用hook方式, 保存ckpt文件
    '''
    def ckpt_saver_hook(self, var_list=None):
        extra_info = {'action_space': self.network.action_space}
        if hasattr(self.network, 'config'):
            extra_info['config'] = self.network.config
        
        listener = CkptSaverListener(
            inputs={
                **self.state_tensors,
                'play_mode': self.play_mode
            },
            outputs={
                **self.sample_action,
                **self.sample_neg_logprob_action,
                'v': self.logits_v,
                **self.extra_tensors
            },
            extra_info=extra_info
        )

        return CheckpointSaverHook(
            checkpoint_dir=f'{CONFIG.restore_dir}/{self.name}/',
            summary_dir=f'{CONFIG.summary_dir}/{self.name}/',
            save_secs=CONFIG.save_checkpoint_secs,
            saver=tf.compat.v1.train.Saver(max_to_keep=CONFIG.save_model_num),
            listeners=[listener, ],
            checkpoint_basename="model"
        )

    def _build_policy(self, input_tensors, mode):
        with tf.compat.v1.variable_scope("network"):
            self.network.build_network(input_tensors)
            self.logits_p = self.network.as_p()
            self.logits_v = self.network.as_v()
            self.extra_tensors = self.network.extra_tensors()

        with tf.compat.v1.variable_scope("sample_action"):
            # CONFIG.play_mode 设置为False, 需要看是否做成配置项目
            self.play_mode = tf.compat.v1.placeholder_with_default([False], [None], 'play_mode')

            self.dist = {action_name: self.network.action_space[action_name].pdclass(logits_p)
                         for action_name, logits_p in self.logits_p.items()}

            self.sample_action = {action_name: dist.sample(self.play_mode)
                                  for action_name, dist in self.dist.items()}
            self.sample_neg_logprob_action = {
                f'neg_logprob_{action_name}': dist.neg_logprob(self.sample_action[action_name])
                for action_name, dist in self.dist.items()
            }

    def _build_loss(self, input_tensors):

        with tf.compat.v1.variable_scope("loss"):
            self.y_r = tf.reshape(input_tensors['y_r'], [-1])
            self.m = tf.reshape(input_tensors['m'], [-1])
            self.old_vpred = tf.reshape(input_tensors['old_vpred'], [-1])

            self.action_dict = {action_name: tf.reshape(input_tensors[action_name], [-1])
                                for action_name in self.network.action_space}

            self.old_neg_logp_dict = {action_name: tf.reshape(input_tensors[f'old_neg_logp_{action_name}'], [-1])
                                      for action_name in self.network.action_space}

            logits_v = tf.reshape(self.logits_v, [-1])

            #self.moving_return.attach(self.y_r)

            advs = self.y_r - self.old_vpred
            mean = tf.math.reduce_mean(advs)
            stdv = tf.math.reduce_std(advs)
            self.adv = (advs - mean) / (stdv + 1e-8)

            self.clip_range = tf.compat.v1.train.exponential_decay(float(CONFIG.ppo_clip_range),
                                                                   self.step,
                                                                   int(CONFIG.decay_steps),
                                                                   float(CONFIG.decay_rate))
            if callable(self.clip_range):
                self.clip_range = self.clip_range()
            self.clip_range = tf.maximum(float(CONFIG.ppo_end_clip_range), self.clip_range)

            # value loss
            vpredclipped = self.old_vpred + tf.clip_by_value(
                logits_v - self.old_vpred, -1.0 * self.clip_range, self.clip_range)
            self.vpred_clipfrac = tf.reduce_mean(tf.cast(
                tf.greater(tf.abs(logits_v - self.old_vpred), self.clip_range), tf.float32))
            vf_losses1 = tf.square(logits_v - self.y_r)  # Unclipped value
            vf_losses2 = tf.square(vpredclipped - self.y_r)  # Clipped value
            self.vf_loss = .5 * tf.reduce_mean(
                tf.maximum(vf_losses1, vf_losses2) * self.m)

            # policy loss
            self.cost_p_all = tf.constant(0.0, dtype=tf.float32)
            self.policy_loss, self.policy_entropy, self.apporx_kl, self.ratio_clipfrac = {}, {}, {}, {}

            for action_name, dist in self.dist.items():
                action = self.action_dict[action_name]

                neg_logp_action = dist.neg_logprob(action)
                old_neg_logp_action = self.old_neg_logp_dict[action_name]

                # entropy
                entropy = tf.reduce_mean(dist.entropy() * self.m)
                self.policy_entropy[action_name] = entropy

                # policy loss
                ratio = tf.exp(old_neg_logp_action - neg_logp_action)
                pg_losses = - self.adv * ratio
                pg_losses2 = - self.adv * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = tf.reduce_mean(
                    tf.maximum(pg_losses, pg_losses2) * self.m)
                self.policy_loss[action_name] = pg_loss

                # approxkl
                approxkl = .5 * tf.reduce_mean(
                    tf.square(neg_logp_action - old_neg_logp_action) * self.m)

                self.apporx_kl[action_name] = approxkl

                # ratio_clipfrac
                ratio_clipfrac = tf.reduce_mean(
                    tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_range), tf.float32) * self.m)
                self.ratio_clipfrac[action_name] = ratio_clipfrac

                # all
                self.cost_p_all += float(CONFIG.ppo_pg_coef) * pg_loss - float(CONFIG.ppo_ent_coef) * entropy

            # total loss
            self.cost_all = self.cost_p_all + self.vf_loss * float(CONFIG.ppo_vf_coef)

    def _build_optimizer(self):
        with tf.compat.v1.variable_scope("optimizer"):
            self.lr = tf.compat.v1.train.exponential_decay(float(CONFIG.learning_rate),
                                                           self.step,
                                                           float(CONFIG.decay_steps),
                                                           float(CONFIG.decay_rate))
            if callable(self.lr):
                self.lr = self.lr()
            self.lr = tf.maximum(float(CONFIG.end_lr), self.lr)

            self.opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.lr,
                epsilon=float(CONFIG.ppo_epsilon))

    def _build_summary(self):

        with tf.compat.v1.variable_scope("summary"):
            for action_name, _policy_loss in self.policy_loss.items():
                tf.compat.v1.summary.scalar("policy_loss_%s" % action_name, _policy_loss)

            for action_name, _policy_entropy in self.policy_entropy.items():
                tf.compat.v1.summary.scalar("policy_entropy_%s" % action_name, _policy_entropy)

            for action_name, _apporx_kl in self.apporx_kl.items():
                tf.compat.v1.summary.scalar("apporx_kl_%s" % action_name, _apporx_kl)

            for action_name, _clipfrac in self.ratio_clipfrac.items():
                tf.compat.v1.summary.scalar("ratio_clipfrac_%s" % action_name, _clipfrac)

            tf.compat.v1.summary.scalar("ACost", self.cost_all)
            tf.compat.v1.summary.scalar("VCost", self.vf_loss)
            tf.compat.v1.summary.scalar("LR", self.lr)
            tf.compat.v1.summary.scalar("clip_range", self.clip_range)
            tf.compat.v1.summary.scalar("vpred_clipfrac", self.vpred_clipfrac)