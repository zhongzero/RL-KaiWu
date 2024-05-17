#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.common.utils.tf_utils import *
from framework.common.algorithms.ckpt_saver_hook import CkptSaverListener

class CkptSaverHookTest(unittest.TestCase):
    def test_save_model(self):
        tensor_x = tf.convert_to_tensor(1)
        listener = CkptSaverListener(
            inputs={'x': tensor_x},
            outputs={'y': tf.identity(tensor_x)}
        )
        ckpt_saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir="/tmp",
            save_secs=30,
            listeners=[listener, ]
        )


if __name__ == '__main__':
    unittest.main()