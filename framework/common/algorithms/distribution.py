#!/usr/bin/env python
# -*- coding: utf-8 -*-


from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG


class Pd:
    def sample(self, play_mode_tensor):
        """
        :param play_mode_tensor: shape=(). dyte=bool. indicate train or predict mode
        :return: Generate samples.
        """
        raise NotImplementedError

    def selected_prob(self, actions):
        """
        :param actions:
        :return: Probability density/mass function.
        """
        raise NotImplementedError

    def neg_logprob(self, actions):
        """
        :param actions:
        :return: Negative Log probability density/mass function.
        """
        raise NotImplementedError

    def selected_logprob(self, actions):
        """
        :param actions:
        :return: Log probability density/mass function.
        """
        raise NotImplementedError

    def entropy(self):
        """
        :return: Shannon entropy in nats.
        """
        raise NotImplementedError

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        """
        :param old_dist:
        :param new_dist:
        :return: Computes the Kullback--Leibler divergence.
        """
        raise NotImplementedError


class CategoricalDist(Pd):
    def __init__(self, logits):
        """
        :param logits: 2-d+ tensor with shape [batch_size, ..., num_classes]
        """
        self.logits = logits
        self.num_actions = tf.shape(logits)[-1]

        self.action_logprobs = tf.cast(tf.nn.log_softmax(logits), dtype=tf.float32)
        self.action_probs = tf.cast(tf.nn.softmax(logits), dtype=tf.float32)

    def sample(self, play_mode_tensor):
        """
        :param play_mode_tensor: a 1-d tensor with shape [batch_size], all have same value(True or False)
        :return:
        """
        logits = tf.reshape(self.logits, [-1, self.num_actions])
        sampled_action = tf.cast(tf.compat.v2.where(
            tf.tile(play_mode_tensor[:1], tf.shape(logits)[:1]),
            tf.math.argmax(logits, axis=-1, name='argmax'),
            tf.squeeze(tf.random.categorical(logits, 1), axis=-1, name='categorical')
        ), dtype=tf.int32)
        sampled_action = tf.reshape(sampled_action, tf.shape(self.logits)[:-1])
        return sampled_action

    def selected_prob(self, actions):
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        selected_prob = tf.reduce_sum(self.action_probs * one_hot_actions, axis=-1)

        return selected_prob

    def neg_logprob(self, actions):
        return -1.0 * self.selected_logprob(actions)

    def selected_logprob(self, actions):
        selected_logprob = -1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=actions)
            
        return selected_logprob

    def entropy(self):
        ent_sy = -1.0 * tf.reduce_sum(self.action_probs * self.action_logprobs, axis=-1)
        # ent_sy = tf.reduce_mean(ent_sy)
        return ent_sy

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        assert isinstance(old_dist, CategoricalDist) and isinstance(new_dist, CategoricalDist)
        kl_sy = tf.reduce_sum(old_dist.action_probs
                              * (old_dist.action_logprobs - new_dist.action_logprobs), axis=-1)
        # kl_sy = tf.reduce_mean(kl_sy)
        return kl_sy


class GaussianDist(Pd):
    def __init__(self, flat):
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = tf.squeeze(mean, axis=1)
        self.logstd = tf.squeeze(logstd, axis=1)
        self.std = tf.exp(self.logstd)
        self.dist = tf.compat.v1.distributions.Normal(loc=self.mean, scale=self.std)
        self.action_probs = tf.zeros_like(mean)

    def sample(self, play_mode_tensor):
        sample = self.dist.sample()
        if CONFIG.gaussian_dist_add_noise:
            return tf.cond(play_mode_tensor,
                           lambda: sample,
                           lambda: sample + tf.distributions.Normal(loc=0.0, scale=CONFIG.gaussian_dist_noise_stddev). \
                           sample(sample_shape=tf.shape(self.mean))
                           )
        return sample

    def neg_logprob(self, actions):
        return -1 * self.selected_logprob(actions)

    def selected_prob(self, actions):
        selected_prob = self.dist.prob(actions)
        return selected_prob

    def selected_logprob(self, actions):
        selected_logprob = self.dist.log_prob(actions)
        return selected_logprob

    def entropy(self):
        return self.dist.entropy()

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        assert isinstance(old_dist, GaussianDist) and isinstance(new_dist, GaussianDist)
        return old_dist.kl_divergence(new_dist)
