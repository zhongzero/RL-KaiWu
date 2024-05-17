#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import traceback
import reverb
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import get_random

class RevervbUtil():
    def __init__(self, learner_addr, logger) -> None:

        self.logger = logger

        self.reverb_client = reverb.Client(learner_addr)
        self.logger.info(f'learner_proxy reverb client {self.reverb_client} {learner_addr} connect to reverb server')

        # 统计情况, 因为该类是高频调用, 不建议打印太多日志, 采用统计项来跟进
        self.send_to_reverb_server_succ_cnt = 0
        self.send_to_reverb_server_error_cnt = 0

        # 如果采样的是reverb_sampler = reverb.selectors.Prioritized, 则需要自动随机生成每条样本的优先级
        self.has_prioritezed = (CONFIG.reverb_sampler == 'reverb.selectors.Prioritized')

        self.max_sequence_length = int(CONFIG.reverb_client_max_sequence_length)
        self.chunk_length = int(CONFIG.reverb_client_chunk_length)

    def get_send_to_reverb_server_stat(self):
        tmp_send_to_reverb_server_succ_cnt = self.send_to_reverb_server_succ_cnt
        tmp_send_to_reverb_server_error_cnt = self.send_to_reverb_server_error_cnt
        
        # 指标周期性的复原
        self.send_to_reverb_server_succ_cnt = 0
        self.send_to_reverb_server_error_cnt = 0

        return tmp_send_to_reverb_server_succ_cnt, tmp_send_to_reverb_server_error_cnt

    '''
    如果设置了reverb.selectors.Prioritized, 则优先级是业务自定义的prioritezeds是个数组
    prioritezeds其长度需要和train_data的长度一致
    '''
    def write_to_reverb_server_simple(self, reverb_table_names, train_data, prioritezeds=None):
        if not train_data:
            return
        
        if self.has_prioritezed:
            if not prioritezeds or len(train_data) != len(prioritezeds):
                return
        
        try: 
            with self.reverb_client.writer(max_sequence_length=self.max_sequence_length, chunk_length=self.chunk_length) as writer:
                count = 0
                # 每条样本进行发送reverb server
                for sample in train_data:
                    for table in reverb_table_names:
                        writer.append(sample)

                        prioritezed = 1.0
                        if self.has_prioritezed:
                            prioritezed = prioritezeds[count]
                        
                        writer.create_item(table, 1, prioritezed)
                    
                    count += 1
                
                # 批量处理减少交互
                writer.flush()

                self.send_to_reverb_server_succ_cnt += len(train_data)
        
        except Exception as e:
            self.send_to_reverb_server_error_cnt += 1
            self.logger.error(f'learner_proxy send one data to reverb server error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}')
    
    '''
    如果设置了reverb.selectors.Prioritized, 则优先级是业务自定义的prioritezeds是个数组
    prioritezeds其长度需要和train_data的长度一致
    '''
    def write_to_reverb_server(self, reverb_table_names, train_data, prioritezeds=None):
        if not train_data:
            return
        
        if self.has_prioritezed:
            if not prioritezeds or len(train_data) != len(prioritezeds):
                return
        
        try: 
            with self.reverb_client.writer(max_sequence_length=self.max_sequence_length, chunk_length=self.chunk_length) as writer:

                # 需要拆分成多个样本, 多少维度就拆分为多少维度, 多次append, 1次creeate_item和flush
                temp_train_data = {}
                keys = list(train_data.keys())
                value_len  = len(train_data[keys[0]])

                count = 0
                for j in range(value_len):
                    for key in keys:
                        temp_train_data[key] = train_data[key][j]
                
                    writer.append(temp_train_data)
                    temp_train_data.clear()

                    for table in reverb_table_names:

                        prioritezed = 1.0
                        if self.has_prioritezed:
                            prioritezed = prioritezeds[count]

                        writer.create_item(table, 1, prioritezed)
                    
                    count += 1
                
                writer.flush()

                self.send_to_reverb_server_succ_cnt += len(train_data)

                # 该日志打印频繁, DEBUG不建议开启
                # self.logger.debug(f'learner_proxy send one data to reverb server success on table {table}')

        except Exception as e:
            self.send_to_reverb_server_error_cnt += 1
            self.logger.error(f'learner_proxy send one data to reverb server error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}')