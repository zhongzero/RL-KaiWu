#!/usr/bin/env bash

# 注意这里的flatc的版本, 必须和代码里使用的flatbuffer版本一致
./flatc -p  -o kaiwu_msg/ kaiwu_rsp.fbs 
./flatc -p  -o kaiwu_msg/ kaiwu_req.fbs 
