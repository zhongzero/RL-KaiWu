#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
普罗米修斯的官网见:https://prometheus.io/
需要安装prometheus_client, 采用pip install prometheus_client, 见: https://github.com/prometheus/client_python
'''

'''
下面是KaiwuDRL上报到普罗米修斯的指标, 容器方面的采集指标复用k8s自带的

采用Counter、Gauge、Histogram、Summary

aisrv
1. aisrv --> actor --> aisrv的平均时延, 最大时延
2. aisrv的QPS
3. aisrv进程的CPU, 内存占用

actor
1. actor的单次预测平均时延, 最大时延
2. actor的GPU使用率

learner
1. learner的GPU使用率
2. learner的单次预测平均时延, 最大时延

'''

import os

MONITOR_ITEMS = {
    'aisrv' : [], 
    'actor' : [], 
    'learner' : [], 
}

import random
from prometheus_client import Counter, Histogram, Summary, Gauge, push_to_gateway, CollectorRegistry
from framework.common.config.config_control import CONFIG
from prometheus_client.exposition import basic_auth_handler
from framework.common.utils.common_func import get_host_ip
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine

# 普罗米修斯监控方面
class PrometheusUtils(object):
    def __init__(self, logger) -> None:

        # 下面是参数配置
        self.prometheus_pwd = CONFIG.prometheus_pwd
        self.prometheus_user = CONFIG.prometheus_user
        self.prometheus_pushgateway = CONFIG.prometheus_pushgateway
        self.prometheus_instance = CONFIG.prometheus_instance
        self.prometheus_db = CONFIG.prometheus_db

        # job名, 格式形如kaiwu_job_1
        self.job = f'kaiwu_pid_{os.getpid()}'
        self.random_low = 0
        self.random_high = 10000

        # 本机IP名
        self.host = get_host_ip()

        # task_id
        self.task_id = CONFIG.task_id

        self.logger = logger

        # 注意每次push后复用问题
        self.registry = CollectorRegistry()

        # 注意每次定义时不能重复, 格式是{srv_name}_{item_name}, 确保每一项指标有对应的数据结构, 故采用map形式
        self.g_maps = {}
        self.c_maps = {}
        self.h_maps = {}
        self.s_maps = {}

        # 默认的lables, ip、task_id属性
        self.default_lables = ['ip', 'task_id']

    '''
    检测进程名是否属于范围内
    '''
    def check_server_name(self, server_name):
        if server_name not in [KaiwuDRLDefine.SERVER_AISRV, KaiwuDRLDefine.SERVER_ACTOR, KaiwuDRLDefine.SERVER_LEARNER, KaiwuDRLDefine.SERVER_CLIENT]:
            return False
        
        return True
    
    '''
    认证
    '''
    def auth_handler(self, url, method, timeout, headers, data):
         return basic_auth_handler(url, method, timeout, headers, data, self.prometheus_user, self.prometheus_pwd)
    
    # Counter使用, 只是增加不减少
    def counter_use(self, server_name, item_name, item_help, value):
        if not self.check_server_name(server_name):
            return
        
        regist_name = f'{server_name}_{item_name}'
        if regist_name not in self.c_maps:
            self.c_maps[regist_name] = Counter(item_name, item_help, registry=self.registry, labelnames=self.default_lables)

        self.c_maps[regist_name].labels(self.host, self.task_id).inc(value)

    # Histogram使用, 直方图
    def histogram_use(self, server_name, item_name, item_help, value):
        if not self.check_server_name(server_name):
            return
        
        regist_name = f'{server_name}_{item_name}'
        if regist_name not in self.h_maps:
            self.h_maps[regist_name] = Histogram(item_name, item_help, registry=self.registry, labelnames=self.default_lables)

        self.h_maps[regist_name].labels(self.host, self.task_id).observe(value)

    # Summary使用
    def summay_use(self, server_name, item_name, item_help, value):
        if not self.check_server_name(server_name):
            return

        regist_name = f'{server_name}_{item_name}'
        if regist_name not in self.s_maps:
            self.s_maps[regist_name] = Summary(item_name, item_help, registry=self.registry, labelnames=self.default_lables)

        self.s_maps[regist_name].labels(self.host, self.task_id).observe(value)

    # Gauge使用, 可增可减
    def gauge_use(self, server_name, item_name, item_help, item_value):
        if not self.check_server_name(server_name):
            return
        
        # 需要保证是调用第一次来定义Gauge, 并且lablenames不能带上item_name
        regist_name = f'{server_name}_{item_name}'
        if regist_name not in self.g_maps:
            self.g_maps[regist_name] = Gauge(item_name, item_help, registry=self.registry, labelnames=self.default_lables)

        self.g_maps[regist_name].labels(self.host, self.task_id).set(item_value)
    
    '''
    由于每次push_to_gateway需要和网络调用, 
    故需要调用N次gauge_use或者summay_use或者histogram_use或者counter_use后再调用push_to_prometheus_gateway, 减少网络耗时
    不能每次就调用push_to_gateway
    '''
    def push_to_prometheus_gateway(self):
        try:
            push_to_gateway(self.prometheus_pushgateway, job = self.job, registry=self.registry, handler=self.auth_handler)
        except Exception as e:
            self.logger.error(f'push_to_gateway faild, error is {str(e)}')
