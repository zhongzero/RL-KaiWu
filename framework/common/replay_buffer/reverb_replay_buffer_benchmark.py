#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import time
import numpy as np
from framework.common.utils.tf_utils import *
import reverb
import argparse
from multiprocessing import Process
from threading import Thread
from numpy import float32, int32, float16

from framework.common.replay_buffer.replay_buffer_benchmark_common import BenchmarkData
from framework.common.replay_buffer.reverb_replay_buffer import ReverbReplayBuffer
from framework.common.replay_buffer.replay_buffer_wrapper import ReplayBufferWrapper
from framework.common.config.config_control import CONFIG
from framework.common.utils.tf_utils import *
from framework.common.logging.kaiwu_logger import KaiwuLogger

parser = argparse.ArgumentParser()
parser.add_argument('--data_scale', default='small', type=str, help='data_scale must be small or big')
parser.add_argument('--svr_host', default='0.0.0.0', type=str)
parser.add_argument('--svr_port', default='6666', type=str)
parser.add_argument('--send_batch_size', default=1, type=int)
parser.add_argument('--max_sequence_length', default=1, type=int)
parser.add_argument('--chunk_length', default=1, type=int)
parser.add_argument('--num_workers_per_iterator', default=4, type=int)
parser.add_argument('--writer_process_num', default=1, type=int)
parser.add_argument('--reverb_version', default='0.1.0', type=str)
parser.add_argument('--table_name', default='reverb_table', type=str)
parser.add_argument('--table_num', default=4, type=int)
parser.add_argument('--sampler_name', default='Uniform', type=str)
parser.add_argument('--remover_name', default='Fifo', type=str)
parser.add_argument('--limiter_name', default='MinSize', type=str)
parser.add_argument('--sample_batch_size', default=512, type=int)
parser.add_argument('--reverb_capacity', default=1024, type=int)

args = parser.parse_args()

benchmark_data = BenchmarkData(args.data_scale)

def send_data_proc():
    client = reverb.client.Client(f'{args.svr_host}:{args.svr_port}')
    data = benchmark_data.sample(args.send_batch_size)
    data = {k: np.squeeze(v, axis=0) for k, v in data.items()}
    with client.writer(max_sequence_length=args.max_sequence_length,
                       chunk_length=args.chunk_length) as writer:
        bgn_time = time.time()
        cnt = 0
        while True:
            for i in range(args.table_num):
                for _ in range(args.chunk_length):
                    writer.append(data)
                cnt += args.chunk_length
                if args.reverb_version == '0.1.0':
                    writer.create_item(args.table_name + str(i), 1, 1, 1.0)
                else:
                    writer.create_item(args.table_name + str(i), args.chunk_length, 1.0)
            cur_time = time.time()
            if cur_time - bgn_time > 30:
                print(f'================ #### writer sample to reverb, cnt {cnt}/30s')
                bgn_time = cur_time
                cnt = 0


def build_limiter(name, batch_size):
    supports = {"MinSize", "SampleToInsertRatio"}
    assert name in supports
    if name == 'MinSize':
        rate_limiter = reverb.rate_limiters.MinSize(batch_size)
    elif name == "SampleToInsertRatio":
        samples_per_insert = 20  # 每个样本的可采样次数
        error_buffer = 1000
        rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
            samples_per_insert, batch_size, error_buffer)
    else:
        rate_limiter = None
    return rate_limiter


def build_remover(name):
    supports = {"Fifo"}
    assert name in supports
    if name == "Fifo":
        remover = reverb.selectors.Fifo()
    else:
        remover = None
    return remover


def build_sampler(name):
    supports = {"Uniform"}
    assert name in supports
    if name == "Uniform":
        sampler = reverb.selectors.Uniform()
    else:
        sampler = None
    return sampler


def build_tables(tensor_spec):
    table_names = [f'{args.table_name}{i}' for i in range(args.table_num)]
    tables = [
        reverb.Table(
            name=table_name,
            max_times_sampled=0,
            sampler=build_sampler(args.sampler_name),
            remover=build_remover(args.remover_name),
            max_size=int(max(args.reverb_capacity, args.sample_batch_size) / args.table_num),
            rate_limiter=build_limiter(args.limiter_name, int(args.sample_batch_size / args.table_num)),
            signature=tensor_spec
        )

        for table_name in table_names
    ]
    return table_names, tables


def make_dataset(table_names):
    def generate_reverb_dataset(table_name):
        return reverb.dataset.ReplayDataset(
            f'{args.svr_host}:{args.svr_port}',
            table=table_name,
            dtypes=benchmark_data.tensor_dtypes(),
            shapes=[tf.TensorShape(shape) for shape in benchmark_data.tensor_shapes()],
            max_in_flight_samples_per_worker=2 * args.sample_batch_size,
            num_workers_per_iterator=args.num_workers_per_iterator,
        )

    dataset = tf.data.Dataset.from_tensor_slices(table_names).interleave(
        generate_reverb_dataset,
        cycle_length=args.table_num,
        block_length=int(1.0 / args.table_num * args.sample_batch_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(args.sample_batch_size
            ).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def start_reverb_server(tables):
    server = reverb.server.Server(tables=tables, port=int(args.svr_port))
    server.wait()


def sample_data_proc():
    tensor_spec = tuple([tf.TensorSpec(shape, dtype, name) for name, dtype, shape in
                         zip(benchmark_data.tensor_names(),
                             benchmark_data.tensor_dtypes(),
                             benchmark_data.tensor_shapes())])
    table_names, tables = build_tables(tensor_spec)
    dataset = make_dataset(table_names)
    s = Thread(target=start_reverb_server, args=(tables,))
    s.start()
    it = tf.compat.v1.data.make_initializable_iterator(dataset)
    next_op = it.get_next()[1]
    sess = tf.compat.v1.train.MonitoredTrainingSession()
    sess.run(it.initializer)
    cnt = 0
    # warm up
    time.sleep(2)
    sess.run(next_op)

    bgn_time = time.time()
    while True:
        data = sess.run(next_op)
        cnt += data[0].shape[0]
        cur_time = time.time()
        if cur_time - bgn_time > 30:
            print(f'================ #### read sample from reverb, cnt {cnt}/30s')
            bgn_time = cur_time
            cnt = 0

def test_reverb_replay_buffer():

    # 加载配置文件
    CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/learner.toml")
    CONFIG.parse_learner_configure()

    tensor_names =  ['x', 'a', 'old_neg_logp_a', 'y_r', 'old_vpred', 'm']
    tensor_dtypes = [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32]
    tensor_shapes = [tf.TensorShape([4, 4]), tf.TensorShape([4]), tf.TensorShape([4]), 
                    tf.TensorShape([4]), tf.TensorShape([4]), tf.TensorShape([1])]

    replay_buffer_wrapper = ReplayBufferWrapper(tensor_names, tensor_dtypes, tensor_shapes, KaiwuLogger())
    replay_buffer_wrapper.init()
    replay_buffer_wrapper.extra_threads()

    print("reverb server is success start up")

    print("获取数据")
    #sess = tf.compat.v1.train.MonitoredTrainingSession()
    sess = tf.Session()
    next_tensors = replay_buffer_wrapper.dataset_from_generator()
    sess.run(replay_buffer_wrapper.extra_initializer_ops())
    while True:
        print(next_tensors)
        data = sess.run(next_tensors)
        print(data)


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
def nump_array_use_rnn():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=float32)
    x = x.reshape(4, 4)

    a = np.array([1, 2, 3, 4], dtype=int32)
    a = a.reshape(4)
    
    old_neg_logp_a = np.array([1, 2, 3, 4], dtype=float32)
    old_neg_logp_a = old_neg_logp_a.reshape(4)
    
    y_r = np.array([1, 2, 3, 4], dtype=float32)
    y_r = y_r.reshape(4)

    old_vpred = np.array([1, 2, 3, 4], dtype=float32)
    old_vpred = old_vpred.reshape(4)

    m = np.array([1], dtype=float32)
    m = m.reshape(1)
    
    # step
    s = np.random.rand(1, 1)

    data = {
        'x' : x,
        'a' : a,
        'old_neg_logp_a' : old_neg_logp_a,
        'y_r' : y_r,
        'old_vpred' : old_vpred,
        'm' : m,
    }

    return data


def nump_array_not_use_rnn():
    # x = np.array([0.1,] * 15552, dtype=float16)
    # x = x.reshape(15552)
    x = np.array([0.1,] * 12, dtype=float16)
    x = x.reshape(12)
    xx = np.array(100, dtype=float16)
    xx = xx.reshape(1)

    a = np.array([1], dtype=int32)
    a = a.reshape(1)
    
    old_neg_logp_a = np.array([1], dtype=float32)
    old_neg_logp_a = old_neg_logp_a.reshape(1)
    
    y_r = np.array([1], dtype=float32)
    y_r = y_r.reshape(1)

    old_vpred = np.array([1], dtype=float32)
    old_vpred = old_vpred.reshape(1)

    m = np.array([1], dtype=float32)
    m = m.reshape(1)
    
    # step
    s = np.random.rand(1, 1)

    data = {
        'input_datas': x, 
        # '_obs': xx
        # 'x' : x,
        # 'a' : a,
        # 'old_neg_logp_a' : old_neg_logp_a,
        # 'y_r' : y_r,
        # 'old_vpred' : old_vpred,
        # 'm' : m,
    }

    return data

def send_data_proc_new():
    client = reverb.client.Client("127.0.0.1:9999")
    if False: # CONFIG.use_rnn:
        data = nump_array_use_rnn()
    else:
        data = nump_array_not_use_rnn()
    #print(data)

    #data = benchmark_data.sample(1)
    #data = {k: np.squeeze(v, axis=0) for k, v in data.items()}
    #print(data)

    table_name = 'reverb_replay_buffer_table_0'

    with client.writer(max_sequence_length=1, chunk_length=1) as writer:
        bgn_time = time.time()
        cnt = 0
        while True:
            print("data send to reverb server" + str(data))
            writer.append(data)
            cnt += 1
            writer.create_item(table_name, 1, 1.0)
            writer.flush()
            cur_time = time.time()
            if cur_time - bgn_time > 1:
                print(f'================ #### writer sample to reverb, cnt {cnt}/30s')
                bgn_time = cur_time
                cnt = 0

def test_inset_replay_buff():
    # 写入数据, 采用线程方式
    print("插入数据")
    p = Process(target = send_data_proc_new)
    p.daemon = True
    p.start()
    p.join()


if __name__ == '__main__':

    #nump_array_use_rnn()

    #nump_array_not_use_rnn()

    test_inset_replay_buff()

    # test_reverb_replay_buffer()
    
    '''
    proc_list = []
    for _ in range(args.writer_process_num):
        p = Process(target=send_data_proc)
        p.start()
        proc_list.append(p)

    p = Process(target=sample_data_proc)
    p.start()
    proc_list.append(p)
    for proc in proc_list:
        proc.join()
    '''
