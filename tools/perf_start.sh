#!/bin/bash


# perf采集C++性能数据, 在perf界面出来后, 关闭时输入q即可

chmod +x tools/common.sh
. tools/common.sh


# 参数如下:
# pid, 进程ID
if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/perf_start.sh pid, such as sh tools/perf_start.sh 1 \033[0m"

    exit -1
fi

pid=$1

tools/perf top -p $pid -g