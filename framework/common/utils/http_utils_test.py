#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
from framework.common.utils.http_utils import http_utils_post, http_utils_request
import warnings


class HttpUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.url = 'sgameai.alloc-proxy-test.odpcldev.woa.com'
        warnings.simplefilter('ignore', ResourceWarning)

    def test_http_get_with_field(self):
        url = f'http://{self.url}/api/get'
        param = {
            "addr":"xxxx", 
            "target_role" : 1
        }

        r = http_utils_request(url, param)
        print(r)
    
    def test_http_get(self):
        url = f'http://{self.url}/api/get'
        r = http_utils_request(url)
        print(r)

    def test_http_post_with_field(self):
        url = f'http://{self.url}/api/registry'
        parm = {
            "set":"testset1",
            "role":0,
            "addr":"6.6.6.6:66",
            "assign_limit":2
        }

        r = http_utils_post(url, parm)
        print(r)
    
    def test_http_post(self):
        url = f'http://{self.url}/api/registry'
        r = http_utils_post(url)
        print(r)
    
    def test_error_url(self):
        url = 'http://123'
        r = http_utils_post(url)
        print(r)

if __name__ == '__main__':
    unittest.main()
