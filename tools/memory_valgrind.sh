#!/bin/bash

# C++内存泄漏检测, 确保已经安装了valgrind, yum install -y valgrind

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/memory_valgrind.sh cmd, \n such as sh tools/memory_valgrind.sh ./aisrv_cpp_server --flagfile /data/projects/kaiwu-fwk/framework/server/cpp/conf/aisrv_server.ini \033[0m"

    exit -1
fi

cmd=$1

yum install -y valgrind

valgrind --leak-check=full --show-leak-kinds=all --log-file=log.txt $cmd 

