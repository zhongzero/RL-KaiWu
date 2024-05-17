#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from framework.common.utils.tf_utils import *

from framework.common.algorithms.distribution import CategoricalDist


try:
    tf.enable_eager_execution()
except Exception: 
    pass


class TestCategoricalDist(unittest.TestCase):
    def test_2d(self):
        logits = tf.convert_to_tensor([[0.1, 0.9]])

        dist = CategoricalDist(logits)
        tensor = dist.sample(tf.convert_to_tensor([True]))
        self.assertEqual(tensor.numpy(), np.array([1]))

        prob = dist.selected_prob([1])
        self.assertEqual(prob.numpy(), np.array([0.6899744], dtype=np.float32))

        neg_logprob = dist.neg_logprob([1])
        self.assertEqual(neg_logprob.numpy(), np.array([0.3711007], dtype=np.float32))

        logprob = dist.selected_logprob([1])
        self.assertEqual(logprob.numpy(), np.array([-0.3711007], dtype=np.float32))

        entropy = dist.entropy()
        self.assertEqual(entropy.numpy(), np.array([0.6191211], dtype=np.float32))

    def test_2d_kl(self):
        old_dist = CategoricalDist(tf.convert_to_tensor([[0.1, 0.1]]))
        new_dist = CategoricalDist(tf.convert_to_tensor([[0.1, 0.1]]))

        kl = CategoricalDist.kl_divergence(old_dist, new_dist)
        self.assertEqual(kl.numpy(), np.array([0.0], dtype=np.float32))

    def test_3d(self):
        logits = tf.convert_to_tensor([[[0.1, 0.9], [0.1, 0.9]], [[0.1, 0.9], [0.1, 0.9]]])

        dist = CategoricalDist(logits)
        tensor = dist.sample(tf.convert_to_tensor([True, True]))
        self.assertTrue(np.allclose(tensor.numpy(), np.array([[1, 1], [1, 1]])))

        prob = dist.selected_prob([[1, 1], [1, 1]])
        self.assertTrue(np.allclose(prob.numpy(), np.array([[0.6899744, 0.6899744], [0.6899744, 0.6899744]])))

        neg_logprob = dist.neg_logprob([[1, 1], [1, 1]])
        self.assertTrue(np.allclose(neg_logprob.numpy(), np.array([[0.3711007, 0.3711007], [0.3711007, 0.3711007]])))

        logprob = dist.selected_logprob([[1, 1], [1, 1]])
        self.assertTrue(np.allclose(logprob.numpy(), np.array([[-0.3711007, -0.3711007], [-0.3711007, -0.3711007]])))

        entropy = dist.entropy()
        self.assertTrue(np.allclose(entropy.numpy(), np.array([[0.6191211, 0.6191211], [0.6191211, 0.6191211]])))

    def test_3d_kl(self):
        old_dist = CategoricalDist(tf.convert_to_tensor([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]]))
        new_dist = CategoricalDist(tf.convert_to_tensor([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]]))

        kl = CategoricalDist.kl_divergence(old_dist, new_dist)
        self.assertTrue(np.allclose(kl.numpy(), np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)))


if __name__ == '__main__':
    unittest.main()
