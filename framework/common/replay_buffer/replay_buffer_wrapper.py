#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import threading
import datetime
from framework.common.utils.tf_utils import *
from framework.common.config.config_control import CONFIG
from framework.common.utils.tf_utils import TF_VERSION_MAJOR
from framework.common.replay_buffer.reverb_replay_buffer import ReverbReplayBuffer

class ReplayBufferWrapper(object):
    def __init__(self, tensor_names, tensor_dtypes, tensor_shapes, logger=None):
        self._tensor_names = tensor_names
        self._tensor_dtypes = tensor_dtypes
        self._tensor_shapes = tensor_shapes
        self._sorted_names = None
        self._sorted_dtypes = None
        self._sorted_shapes = None


        # 针对replaybuffer 统计信息
        self.proc_sample_cnt = 0
        self.skip_sample_cnt = 0

        self.logger = logger
        self.logger.info(f"train replaybuff, use {CONFIG.replay_buffer_type}")
    
    def init(self):
        if CONFIG.use_reverb():
            self._replay_buffer = ReverbReplayBuffer(
                tuple([tf.TensorSpec(shape, dtype, name) for name, dtype, shape in zip(*self.sorted_tensor_spec())]))
            self._replay_buffer.init()
        elif CONFIG.use_tf_uniform():
            self._replay_buffer = NotImplemented
        elif CONFIG.use_mempool():
            self._replay_buffer = NotImplemented
        else:
            raise ValueError('ReplayBuffer currently only support reverb or tf_uniform or mempool!')

    def train_hooks(self, local_step_tensor=None):
        return []

    def sorted_tensor_spec(self):
        if self._sorted_names is None or self._sorted_dtypes is None or self._sorted_shapes is None:
            shapes = [tf.TensorShape(([int(CONFIG.rnn_time_steps)] + shape.dims[2:]) if CONFIG.use_rnn else shape.dims[1:])
                      for shape in self._tensor_shapes]
            tensor_infos = list(zip(self._tensor_names, self._tensor_dtypes, shapes))
            sorted_tensors_infos = sorted(tensor_infos, key=lambda x: x[0])
            tmp_uniq_names, names, dtypes, shapes = set(), [], [], []
            # uniq
            for item in sorted_tensors_infos:
                if item[0] not in tmp_uniq_names:
                    tmp_uniq_names.add(item[0])
                    names.append(item[0])
                    dtypes.append(item[1])
                    shapes.append(item[2])
            shapes = [tf.TensorShape([1, ]) if s.ndims == 0 else s for s in shapes]
            for i, (name, shape) in enumerate(list(zip(names, shapes))):
                only_keep_first = True if CONFIG.use_rnn and name in CONFIG.rnn_states else False
                if only_keep_first:
                    shape = tf.TensorShape([1] + shape.dims[1:])
                    shapes[i] = shape
                self.logger.info(f"train tensor spec: {name}, {shape}")

            # Replay buffer hooker needs `step` to filter expired samples.
            if CONFIG.use_tf_uniform():
                names += ['s']
                dtypes += [tf.int64]
                shapes += [tf.TensorShape([1, ])]

            self._sorted_dtypes = dtypes
            self._sorted_names = names
            self._sorted_shapes = shapes

        return self._sorted_names, self._sorted_dtypes, self._sorted_shapes

    '''
    该方案采用dataset.from_generator来进行构造数据, 获取到具体数据, 再进行run_session
    '''
    def dataset_from_generator(self):
        dataset = self._replay_buffer.as_dataset()
        self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)

        if CONFIG.use_reverb():
            next_tensors = self._dataset_iter.get_next()[1]
        elif CONFIG.use_tf_uniform():
            next_tensors = self._dataset_iter.get_next()
        else:
            assert False
        
        return next_tensors

    '''
    该方案是采用tf.compat.v1.placeholder_with_default占位符 + 业务自定义网络结构生成的流水线设计, 推荐
    '''

    def input_tensors(self):
        if CONFIG.use_tf_uniform():
            pass

        dataset = self._replay_buffer.as_dataset()

        self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        if CONFIG.use_reverb():
            next_tensors = self._dataset_iter.get_next()[1]
        elif CONFIG.use_tf_uniform():
            next_tensors = self._dataset_iter.get_next()
        else:
            assert False

        tensors = [
            tf.compat.v1.placeholder_with_default(d, shape=[None, None] + d.get_shape().as_list()[2:])
            if CONFIG.use_rnn else
            tf.compat.v1.placeholder_with_default(d, shape=[None] + d.get_shape().as_list()[1:])
            for d in next_tensors
        ]

        return dict(zip(self._sorted_names, tensors))

    def extra_initializer_ops(self):
        return [self._dataset_iter.initializer]

    def extra_threads(self):
        if CONFIG.use_reverb():
            self._reverb_server = self._replay_buffer.build_reverb_server()
            self._reverb_client = self._replay_buffer.build_reverb_client()

            '''
            server的wait是常驻线程
            '''
            def start_reverb_server():
                self._reverb_server.wait()

            thread = threading.Thread(target=start_reverb_server)
            thread.daemon = True
            thread.start()

            def start_reverb_update_stats():
                has_inserted = False
                while True:
                    if TF_VERSION_MAJOR == 1:
                        if has_inserted and self.proc_sample_cnt == 0:
                            __ = self._reverb_client.server_stats_info(True)
                            has_inserted = False
                        else:
                            server_stats_info = self._reverb_client.server_stats_info(False)
                            ps_cnt, ss_cnt = 0, 0
                            for table_name in self._replay_buffer.table_names:
                                ps_cnt += server_stats_info[table_name].proc_frame_cnt
                                ss_cnt += server_stats_info[table_name].skip_frame_cnt
                            self.proc_sample_cnt = ps_cnt
                            self.skip_sample_cnt = ss_cnt
                            if not has_inserted and ps_cnt > 0:
                                has_inserted = True
                    
                    # 复用 idle_sleep_second 参数
                    time.sleep(CONFIG.idle_sleep_second)

            #thread = threading.Thread(target=start_reverb_update_stats)
            #thread.daemon = True
            #thread.start()

    def reset(self, step, tf_sess):
        if CONFIG.use_reverb():
            self._replay_buffer.clear(self._reverb_client, step)
        elif CONFIG.use_tf_uniform():
            tf_sess.run(self._tf_replay_buffer_clear)
        else:
            assert False

    def input_ready(self, tf_sess):
        if CONFIG.use_reverb():
            current_size = self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.use_tf_uniform():
            # 这里暂时没有实现
            current_size = tf_sess.run(self._tf_replay_buffer_total_size)
        else:
            assert False

        # self.logger.debug(f"train current_size: {current_size}, CONFIG.train_batch_size: {CONFIG.train_batch_size}")
        return current_size >= int(CONFIG.train_batch_size)
    
    '''
    获取样本接收速度
    '''
    def get_recv_speed(self):
        if CONFIG.use_reverb():
            return 0
        elif CONFIG.use_tf_uniform():
            return 0
        else:
            assert False
    
    '''
    获取目前样本池里的数目
    '''
    def get_current_size(self):
        if CONFIG.use_reverb():
            return self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.use_tf_uniform():
            # 这里暂时没有实现
            return 0
        else:
            assert False
        
    '''
    获取目前样本池里插入的数目
    '''
    def get_insert_stats(self):
        if CONFIG.use_reverb():
            return self._replay_buffer.insert_stats(self._reverb_client)
        elif CONFIG.use_tf_uniform():
            # 这里暂时没有实现
            return 0
        else:
            assert False
