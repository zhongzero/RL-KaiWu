#!/bin/bash
# 进程查看脚本, 用于在启动进程前或者关闭进程后确认使用

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ] && [ $# -ne 0 ];
then
    echo -e "\033[31m useage: sh tools/check.sh [process_name], such as: sh tools/check.sh or sh tools/check.sh aisrv \033[0m"
    exit -1
fi

process_name=$1
if [ -z $process_name ]
then
    array=("aisrv" "actor" "learner" "modelpool" "alloc" "job_master" "client" "interface_server")
    for element in ${array[@]}
    do
        judge_process_exist $element
    done
else
    judge_process_exist $process_name
fi