#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# need install urllib3
import json
import traceback
from urllib3 import *
import warnings
warnings.simplefilter('ignore', ResourceWarning)

# http request请求
def http_utils_request(url, fields=None):
    if not url:
        return 

    data = None
    try:
        http = PoolManager(timeout=Timeout(connect=2.0, read=2.0))

        if fields:
            r = http.request('GET', url, fields=fields)
        else:
            r = http.request('GET', url)

        # 提前判断status的值
        if r.status != 200:
            return None
        
        data = json.loads(r.data.decode('utf-8'), strict=False)
    
    # urllib3.exceptions的比较多, 故采用Exception作为兜底的, 并且不做处理, filnally返回data
    except Exception  as e:
        print(f'http_utils_request error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}')
    finally:
        return data

# http post请求, fields支持json格式
def http_utils_post(url, fields=None):
    if not url:
        return
    
    data = None
    try:
        http = PoolManager(timeout=Timeout(connect=2.0, read=2.0))

        if fields:
            encode_data = json.dumps(fields).encode('utf-8')
            r = http.request('POST', url, body=encode_data, headers={'Content-Type' : 'application/json'})
        else:
            r = http.request('POST', url)
        
        # 提前判断status的值
        if r.status != 200:
            return None

        data = json.loads(r.data.decode('utf-8'), strict=False)
    
    # urllib3.exceptions的比较多, 故采用Exception作为兜底的, 并且不做处理, filnally返回data
    except Exception  as e:
        print(f'http_utils_post error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}')
    finally:
        return data