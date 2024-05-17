#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project :1v1
@File    :config.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

import os
from framework.common.config.config_control import CONFIG


class DimConfig:
    '''
    Dimension configuration file, called by model.py, passes the specific descriptions of each segment of the features. 
    For detailed information, please refer to the document: "Honor of King 1v1 Experimental Development Guide - Environment Introduction - Observation Space."

    维度配置文件, model.py调用，传入特征每段的具体描述，详细描述可以查看文档: “王者荣耀1v1实验开发指南 - 环境介绍 - 观测空间“
    '''

    # own soldiers
    # 己方小兵
    DIM_OF_SOLDIER_1_10 = [
        18,
        18,
        18,
        18]

    # enemy soldiers
    # 敌方小兵
    DIM_OF_SOLDIER_11_20 = [
        18,
        18,
        18,
        18]

    # own defense tower
    # 己方防御塔
    DIM_OF_ORGAN_1_2 = [
        18,
        18]

    # enemy defense tower
    # 敌方防御塔
    DIM_OF_ORGAN_3_4 = [
        18,
        18]

    # own hero
    # 己方英雄
    DIM_OF_HERO_FRD = [
        235]

    # enemy hero
    # 敌方英雄
    DIM_OF_HERO_EMY = [
        235]

    # main hero
    # 主英雄
    DIM_OF_HERO_MAIN = [14] 

    # global information of map
    # 地图的全局信息
    DIM_OF_GLOBAL_INFO = [25]


class ModelConfig:
    '''
    Configuration file, called by model.py

    模型的配置文件, model.py调用
    '''
    NETWORK_NAME = "network"
    # step and unit information of LSTM
    # LSTM的step和unit信息
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    # slicing dimension of the input data for the model
    # 传入模型的数据的切片维度
    DATA_SPLIT_SHAPE = [809, 1, 1, 1,1,1,1,1,1, 12, 16, 16, 16, 16, 8, 1, 1, 1, 1, 1, 1, 1, 512, 512]
    # 
    # 向量特征的切片维度
    SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
    # 0.0001
    INIT_LEARNING_RATE_START = CONFIG.learning_rate
    # 0.025
    BETA_START = CONFIG.var_beta

    LOG_EPSILON = 1e-6
    # The number and dimension description of labels output by the model, for specific meanings,
    # please refer to the document: "Honor of King 1v1 Experimental Development Guide - Environment Introduction - Action Space".
    # 模型输出的label头数和维度描述，具体含义可以查看文档: “王者荣耀1v1实验开发指南 - 环境介绍 - 动作空间“
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]

    # The expanded dimensions of legal actions, for specific meanings,
    # please refer to the document: "Honor of King 1v1 Experimental Development Guide - Environment Introduction - Action Space".
    # legal action的拓展维度，具体含义可以查看文档: “王者荣耀1v1实验开发指南 - 环境介绍 - 动作空间“
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1]*LEGAL_ACTION_SIZE_LIST[0]

    # Determine which label headers participate in RL learning, True indicates participation, False indicates non-participation.
    # 确定是否哪些label头的输出参与RL学习，True表示参与，False表示不参与
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True] 

    # 0.2
    CLIP_PARAM = CONFIG.ppo_clip_range

    # minimum probability value for constraining the policy.
    # 限制策略的最小概率值
    MIN_POLICY = 0.00001

    BATCH_SIZE = CONFIG.train_batch_size
    TARGET_EMBED_DIM = 32

    # Meaning, shape, and type of the sliced input features.
    # 传入特征分片后对应的含义，shape和type
    data_keys = "observation,reward,advantage,label0,label1,label2,label3,label4,label5,prob0,prob1,prob2,prob3,prob4,prob5,weight0,weight1,weight2,weight3,weight4,weight5,is_train, lstm_cell, lstm_hidden_state"
    data_shapes = [[12944], [16], [16], [16], [16], [16], [16], [16], [16], [192], [256], [256], [256], [256], [128], [16], [16], [16], [16], [16], [16], [16], [512], [512]]
    key_types = "tf.float32,tf.float32,tf.float32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32"


class Config:
    '''
    Configuration file, called by sgame_sample_processor.py and agent.py.

    配置文件, sgame_sample_processor.py和agent.py调用
    '''
    GAMMA = 0.995
    LAMDA = 0.95
    
    SINGLE_TEST = False
    IS_CHECK = False

    ENEMY_TYPE = "network"
    if os.getenv("ENEMY_TYPE") is not None:
        enemy_type = int(os.getenv("ENEMY_TYPE"))
        if enemy_type == 0:
            ENEMY_TYPE = "random"
        elif enemy_type == 1:
            ENEMY_TYPE = "common_ai"
        elif enemy_type == 2:
            ENEMY_TYPE = "network"

    ACTION_DIM = 79
    INPUT_DIM = [2823]
