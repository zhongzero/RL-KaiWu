#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from pydoc import locate
from framework.common.utils.tf_utils import *
import reverb

from framework.common.config.config_control import CONFIG
from framework.common.replay_buffer.replay_buffer_base import ReplayBufferBase
from framework.common.utils.tf_utils import *

# ref https://github.com/deepmind/reverb, need pip install dm-reverb

class ReverbReplayBuffer(ReplayBufferBase):
    def __init__(self, data_spec, capacity = 4096):

        # 以配置文件里为准
        capacity = CONFIG.replay_buffer_capacity
        super().__init__(data_spec, capacity)

        self._table_names = ""

        # 优先级里默认的0.5
        self.priority_exponent = 0.5
        
    def init(self):
        self._table_names = ['{}_{}'.format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]
        self._svr_port = CONFIG.reverb_svr_port

    @property
    def table_names(self):
        return self._table_names

    def _build_sampler(self):
        """several selectors that can be used for item sampling or removal
            reverb.selectors.Uniform: Sample uniformly among all items.
            reverb.selectors.Prioritized: Samples proportional to stored priorities.
            reverb.selectors.FIFO: Selects the oldest data.
            reverb.selectors.LIFO: Selects the newest data.
            reverb.selectors.MinHeap: Selects data with the lowest priority.
            reverb.selectors.MaxHeap: Selects data with the highest priority
        """
        reverb_sampler = None
        if CONFIG.reverb_sampler == 'reverb.selectors.Uniform' or CONFIG.reverb_sampler == 'reverb.selectors.Fifo':
            reverb_sampler = locate(CONFIG.reverb_sampler)()
        elif CONFIG.reverb_sampler == 'reverb.selectors.Prioritized':
            reverb_sampler =locate(CONFIG.reverb_sampler)(self.priority_exponent)
        else:
            pass

        return reverb_sampler

    def _build_remover(self):
        """several selectors that can be used for item sampling or removal
            reverb.selectors.Uniform: Sample uniformly among all items.
            reverb.selectors.Prioritized: Samples proportional to stored priorities.
            reverb.selectors.FIFO: Selects the oldest data.
            reverb.selectors.LIFO: Selects the newest data.
            reverb.selectors.MinHeap: Selects data with the lowest priority.
            reverb.selectors.MaxHeap: Selects data with the highest priority
        """
        reverb_remover = None
        if CONFIG.reverb_remover == 'reverb.selectors.Fifo' or CONFIG.reverb_remover == 'reverb.selectors.Lifo':
            reverb_remover = locate(CONFIG.reverb_remover)()
        elif CONFIG.reverb_remover == 'reverb.selectors.Prioritized':
            reverb_remover = locate(CONFIG.reverb_remover)(self.priority_exponent)
        else:
            pass

        return reverb_remover

    def _build_limiter(self):
        """Rate limiters allow users to enforce conditions on when items can be inserted and/or sampled from a Table.
        MinSize:
         Sets a minimum number of items that must be in the Table before anything can be sampled.
         This limiter blocks all sample calls when the replay contains less than `min_size_to_sample` items,
         and accepts all sample calls otherwise.
        SampleToInsertRatio:
          Sets that the average ratio of inserts to samples by blocking insert and/or sample requests.
          This SampleToInsertRatio limiter works in two stages:
          Stage 1. Size of table is lt `min_size_to_sample`.
          Stage 2. Size of table is ge `min_size_to_sample`.
          During stage 1 the limiter works exactly like MinSize,
          i.e. it allows  all insert calls and blocks all sample calls. Note that it is possible to
          transition into stage 1 from stage 2 when items are removed from the table.
          During stage 2,
           the limiter attempts to maintain the ratio  `samples_per_inserts` between the samples and inserts.
          This is done by  measuring the "error" in this ratio, calculated as:
          (number_of_inserts - min_size_to_sample) * samples_per_insert - number_of_samples
          If this quantity is within the range (-error_buffer, error_buffer) then no limiting occurs.
          If the error is larger than `error_buffer` then insert calls
          will be blocked; sampling will be blocked for error less than -error_buffer.
        """
        if CONFIG.reverb_rate_limiter == 'MinSize':
            rate_limiter = reverb.rate_limiters.MinSize(int(int(CONFIG.train_batch_size) / int(CONFIG.reverb_table_size)))
        else:
            rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
                CONFIG.reverb_samples_per_insert, int(int(CONFIG.train_batch_size) / int(CONFIG.reverb_table_size)),
                CONFIG.reverb_error_buffer)
        
        return rate_limiter

    def build_reverb_server(self):
        # Support multi tables.

        sampler = self._build_sampler()
        remover = self._build_remover()
        rate_limiter = self._build_limiter()

        tables = [reverb.Table(
            name = table_name,
            max_size = int(max(int(CONFIG.replay_buffer_capacity), int(CONFIG.train_batch_size)) / int(CONFIG.reverb_table_size)),
            sampler=sampler,
            remover=remover,
            rate_limiter=rate_limiter,
            signature=self._data_spec,
        ) for table_name in self._table_names]
        
        server =  reverb.Server(tables=tables, port=int(self._svr_port))
        return server

    def build_reverb_client(self):
        return reverb.Client(f'localhost:{self._svr_port}')

    def clear(self, client, step=None):
        # return的不是op
        for table_name in self._table_names:
            if TF_VERSION_MAJOR == 1 and step is not None:
                client.set_local_step(table_name, step)
            client.reset(table_name)
        return None

    def as_dataset(self, batch_size=128, prefetch_size=tf.data.experimental.AUTOTUNE):
        # 以配置为主
        batch_size = int(CONFIG.train_batch_size)
        get_dtype = lambda x: x.dtype
        get_shape = lambda x: x.shape
        shapes = tf.nest.map_structure(get_shape, self._data_spec)
        dtypes = tf.nest.map_structure(get_dtype, self._data_spec)

        # 设置为自动并行, 提高CPU使用率
        num_parallel_calls = tf.data.experimental.AUTOTUNE
        
        dataset = tf.data.Dataset.from_tensor_slices(self._table_names).interleave(
            lambda name: reverb.dataset.ReplayDataset(
                f'localhost:{CONFIG.reverb_svr_port}',
                table=name,
                dtypes=dtypes,
                shapes=shapes,
                max_in_flight_samples_per_worker=2 * batch_size,
                num_workers_per_iterator=CONFIG.reverb_num_workers_per_iterator),
            cycle_length=CONFIG.reverb_table_size,
            block_length=int(1.0 / int(CONFIG.reverb_table_size) * int(CONFIG.train_batch_size)),
            num_parallel_calls=num_parallel_calls,
        )

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        if prefetch_size is not None:
            dataset = dataset.prefetch(prefetch_size)
        
        return dataset
    
    # 获取insert次数
    def insert_stats(self, client=None):
        return sum([client.server_info()[table_name].rate_limiter_info.insert_stats.completed for table_name in self._table_names])

    def total_size(self, client=None):
        return sum([client.server_info()[table_name].current_size for table_name in self._table_names])

    def add_batch(self):
        raise NotImplementedError('ReverbReplayBuffer does not support `add_batch`.')

    def gather_all(self):
        raise NotImplementedError('ReverbReplayBuffer does not support `gather_all`.')
