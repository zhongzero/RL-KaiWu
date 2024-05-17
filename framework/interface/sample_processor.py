#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
样本相关的接口
'''
class SampleProcessor:
    def __init__(self) -> None:
        pass

    '''
    是否需要训练
    '''
    def should_train(self):
        raise NotImplementedError

    '''
    生产样本函数, 返回train_data, train_frame_cnt, drop_frame_cnt
    '''
    def proc_exprs(self):
        raise NotImplementedError