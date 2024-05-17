#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import reverb
from framework.common.config.config_control import CONFIG
import os
from numpy import float32, int32, array
from framework.common.logging import getLogger
import numpy as np
import lz4.block
import multiprocessing


class LearnerProxyTest(multiprocessing.Process):
    def __init__(self) -> None:

        super().__init__()

        # 解析aisrv进程的配置
        CONFIG.set_configure_file('/data/projects/kaiwu-fwk/conf/framework/aisrv.toml')
        CONFIG.parse_aisrv_configure()

        self.reverb_client = None
    
    def insert_func_a(self, train_data):
        reverb_table_names = ['reverb_replay_buffer_table_0']
        try: 
            with self.reverb_client.writer(max_sequence_length=1, chunk_length=1) as writer:
                # 需要拆分成多个样本
                temp_train_data = {}
                keys = list(train_data.keys())
                value_len  = len(train_data[keys[0]])
                for j in range(value_len):
                    for key in keys:
                        temp_train_data[key] = train_data[key][j]
                
                    print(temp_train_data)
                    writer.append(temp_train_data)
                    temp_train_data.clear()
                
                for table in reverb_table_names:
                    writer.create_item(table, 1, 1.0)
                    writer.flush()
                    print(f'send one data use writer to reverb server table {table}')

        except Exception as e:
            print(f'send one data to reverb server error as {str(e)}')
    
    def insert_fun_b(self, train_data):
        reverb_table_names = ['reverb_replay_buffer_table_0']
        try: 
            writer = self.reverb_client.writer(max_sequence_length=1, chunk_length=1) 

            # 需要拆分成多个样本
            temp_train_data = {}
            keys = list(train_data.keys())
            value_len  = len(train_data[keys[0]])
            for j in range(value_len):
                for key in keys:
                    temp_train_data[key] = train_data[key][j]
                
                print(temp_train_data)
                writer.append(temp_train_data)
                temp_train_data.clear()
                
            for table in reverb_table_names:
                writer.create_item(table, 1, 1.0)
                writer.flush()
                print(f'send one data use writer to reverb server table {table}')

        except Exception as e:
            print(f'send one data to reverb server error as {str(e)}')
        finally:
            writer.close()
    
    def insert_func_c(self, train_data):
        reverb_table_names = ['reverb_replay_buffer_table_0']
        '''
        使用trajectory_writer形式
        '''
        try: 
            with self.reverb_client.trajectory_writer(num_keep_alive_refs = len(train_data)) as writer:
                # 需要拆分成多个样本
                temp_train_data = {}
                keys = list(train_data.keys())

                value_len  = len(train_data[keys[0]])
                for j in range(value_len):
                    for key in keys:
                        temp_train_data[key] = train_data[key][j]
                
                    print(temp_train_data)
                    writer.append(temp_train_data)
                    temp_train_data.clear()
                
                trajectory_list = []
                for key in keys:
                    trajectory_list.append(writer.history[key][:])
                
                for table in reverb_table_names:
                    writer.create_item(table, 1.0, trajectory = trajectory_list)
                    writer.end_episode()
                    print(f'send one data use trajectory_writer to reverb server table {table}')

        except Exception as e:
            print(f'send one data to reverb server error as {str(e)}')

    def befor_run(self):
        self.reverb_client = reverb.Client(f'127.0.0.1:9999')


    def run(self):
        self.befor_run()

        while True:
            self.test_reverb_write()

    def test_reverb_write(self):
        train_data = {
            'x': array([[-2.0738866e-02,  5.5687334e-03,  6.3040312e-03, -5.9341458e-03],
       [-2.0627493e-02,  2.0059972e-01,  6.1853481e-03, -2.9662141e-01],
       [-1.6615499e-02,  3.9563295e-01,  2.5291980e-04, -5.8734721e-01],
       [-8.7028388e-03,  2.0050745e-01, -1.1494024e-02, -2.9458460e-01],
       [-4.6926900e-03,  3.9579138e-01, -1.7385716e-02, -5.9087032e-01],
       [ 3.2231372e-03,  5.9115237e-01, -2.9203122e-02, -8.8897866e-01],
       [ 1.5046185e-02,  7.8665817e-01, -4.6982694e-02, -1.1906968e+00],
       [ 3.0779349e-02,  9.8235637e-01, -7.0796631e-02, -1.4977281e+00],
       [ 5.0426476e-02,  7.8816265e-01, -1.0075119e-01, -1.2279640e+00],
       [ 6.6189729e-02,  9.8442656e-01, -1.2531048e-01, -1.5504377e+00],
       [ 8.5878260e-02,  1.1808093e+00, -1.5631923e-01, -1.8794470e+00],
       [ 1.0949445e-01,  9.8769879e-01, -1.9390817e-01, -1.6390840e+00]],
        dtype=float32),

        'a': array([[1], [1], [0], [1], [1], [1], [1], [0], [1], [1], [0], [0]], dtype=int32), 

            'old_neg_logp_a': array([[0.6980009 ],
       [0.70184183],
       [0.6805177 ],
       [0.70106095],
       [0.70511955],
       [0.7089584 ],
       [0.7125829 ],
       [0.6711563 ],
       [0.71160614],
       [0.715009  ],
       [0.6680723 ],
       [0.6706911 ]], dtype=float32),

            'y_r': array([[8.78271  ],
       [8.272708 ],
       [7.730111 ],
       [7.153529 ],
       [6.540121 ],
       [5.887482 ],
       [5.193087 ],
       [4.4542437],
       [3.669058 ],
       [2.8336408],
       [1.944784 ],
       [1.       ]], dtype=float32), 
       
            'old_vpred': array([[0.03259209],
       [0.04502943],
       [0.05128409],
       [0.04478063],
       [0.051423  ],
       [0.05947725],
       [0.06835523],
       [0.07820268],
       [0.0705913 ],
       [0.08119134],
       [0.09235431],
       [0.08654644]], dtype=float32), 

            'm': array([[1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.]], dtype=float32), 

            's': array([[0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0]])
            }

        # 删除step
        del train_data['s']

        self.insert_func_a(train_data)

        #self.insert_fun_b(train_data)

        #self.insert_func_c(train_data)

if __name__ == '__main__':
    LearnerProxyTest = LearnerProxyTest()
    LearnerProxyTest.start()
    LearnerProxyTest.join()