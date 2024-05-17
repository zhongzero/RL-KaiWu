#!/bin/bash

# C++内存泄漏检测, 确保已经安装了asan, yum install -y libasan


if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/memory_asan.sh cmd, \n such as sh tools/memory_asan.sh ./aisrv_cpp_server --flagfile /data/projects/kaiwu-fwk/framework/server/cpp/conf/aisrv_server.ini \033[0m"

    exit -1
fi

cmd=$1

yum install -y libasan

export LD_PRELOAD=/usr/local/lib64/libasan.so.4

$cmd

