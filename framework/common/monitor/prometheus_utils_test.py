#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.common.monitor.prometheus_utils import PrometheusUtils
from framework.common.config.config_control import CONFIG

class AllocUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/aisrv.toml")
        CONFIG.parse_aisrv_configure()

        self.prometheus_utils = PrometheusUtils(None)

    def test_counter_use(self):
        self.prometheus_utils.counter_use('aisrv', 'qps', 'qps help', 1)

    def test_histogram_use(self):
        pass

    def test_summay_use(self):
        pass

    def test_gauge_use(self):
        self.prometheus_utils.gauge_use('aisrv', 'qps', 'qps help', 1)
    
    def test_all(self):
        from prometheus_client.exposition import basic_auth_handler
        from prometheus_client import Counter, Histogram, Summary, Gauge, push_to_gateway, CollectorRegistry
        
        def auth_handler(url, method, timeout, headers, data):
            return basic_auth_handler(url, method, timeout, headers, data, '1258344700', 'rIBVP&&Be28TxS+PuAHu44(evLC')


        registry = CollectorRegistry()
        g = Gauge('qps', 'qps help', registry=registry, labelnames=['qps'])
        for i in range(100):
            g.labels('qps').inc(1)

        try:
            push_to_gateway('11.177.89.153:9090', job='job_1', registry=registry, handler=auth_handler)
        except Exception as e:
            raise e

if __name__ == '__main__':
    unittest.main()