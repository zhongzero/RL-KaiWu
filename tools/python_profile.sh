#!/bin/bash


# python里查看性能, 类似perf的工具, 参照https://github.com/benfred/py-spy#how-do-i-run-py-spy-in-docker, 需要pip3 install py-spy

chmod +x tools/common.sh
. tools/common.sh


# 参数如下:
# pid, 进程ID
if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/python_profile.sh pid, such as sh tools/python_profile.sh 1 \033[0m"

    exit -1
fi

pid=$1

/usr/local/python3/bin/py-spy top --pid $pid

 /usr/local/python3/bin/py-spy record -o profile.svg --pid pid