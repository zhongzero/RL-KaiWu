#!/bin/bash


# 检测进程是否存在, 如果不存在则拉起来进程

process_name=aisrv_cpp_server

while true
do 
    # 休息120s, 规避aisrv进程没有启动成功, 如果aisrv进程出现core则因为1分钟的监控上报间隔, 能看出来是哪些aisrv进程异常
    sleep 120

    # 检测aisrv_cpp_server进程是否存在, 如果存在则跳过, 不存在则直接kill掉aisrv便于容器重新启动
    process_num=`ps -ef | grep $process_name | grep -v grep | wc -l`
    if [ $process_num -eq 0 ];
    then
        ps -ef | grep aisrv | grep -v "grep" | awk '{print $2}' | xargs kill -9
    fi

done