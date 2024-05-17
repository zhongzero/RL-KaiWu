#!/bin/bash
# 更新配置文件里的进程个数的配置



chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/change_alloc_process_count.sh aisrv|actor|learner|arena count \
    such as: sh tools/change_alloc_process_count.sh aisrv 1  \033[0m"
    
    exit -1
fi

server_name=$1
count=$2

configure_file=conf/framework/aisrv.toml
if [ $server_name == "aisrv" ];
then
    echo -e "\033[32m aisrv not need to change alloc process count \033[0m"
    exit -1

elif [ $server_name == "actor" ];
then
    sed -i "s/aisrv_connect_to_actor_count = .*/aisrv_connect_to_actor_count = $count/g" $configure_file

elif [ $server_name == "learner" ];
then
    sed -i "s/aisrv_connect_to_learner_count = .*/aisrv_connect_to_learner_count = $count/g" $configure_file

elif [ $server_name == "arena" ];
then
    sed -i "s/aisrv_connect_to_arena_count = .*/aisrv_connect_to_arena_count = $count/g" $configure_file

else
    echo -e "\033[31m useage: sh tools/change_alloc_process_count.sh aisrv|actor|learner|arena count \
    such as: sh tools/change_alloc_process_count.sh aisrv 1  \033[0m"
    
    exit -1
fi

judge_succ_or_fail $? "$server_name change alloc process count $count"
