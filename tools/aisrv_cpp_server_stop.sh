#!/bin/bash


# aisrv_cpp_server_stop.sh 主要用于关闭C++常驻进程

chmod +x tools/common.sh
. tools/common.sh

# 删除以前的共享内存
rm -rf /dev/shm/*

judge_process_exist_and_kill "aisrv_cpp_server"
judge_succ_or_fail $? "aisrv_cpp_server stop"
