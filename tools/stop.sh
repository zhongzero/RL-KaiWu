#!/bin/bash
# 进程停止脚本

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/stop.sh all|actor|learner|aisrv|interface_server|battlesrv|modelpool, such as: sh tools/stop.sh all \033[0m"
    
    exit -1
fi

server_type=$1
if [ $server_type == "aisrv" ] || [ $server_type == "actor" ] || [ $server_type == "learner" ] || [ $server_type == "interface_server" ];
then

    # KaiwuDRL的组件安全关闭
    judge_process_exist_and_kill $server_type
    judge_succ_or_fail $? "$server_type stop"

elif [ $server_type == "all" ];
then
    # 依赖的第三方组件modelpool是需要独立部署的, 如果是开发测试环境可以单独手动启动
    sh thirdparty/model_pool_go/op/stop.sh
    judge_succ_or_fail $? "modelpool stop"

    # KaiwuDRL的组件安全关闭
    array=("aisrv" "actor" "learner" "battlesrv" "interface_server")
    for element in ${array[@]}
    do
        judge_process_exist_and_kill $element
        judge_succ_or_fail $? "$element stop"
    done

elif [ $server_type == "client" ];
then
    judge_process_exist_and_kill "sgame_client"
    judge_succ_or_fail $? "sgame_client stop"

elif [ $server_type == "battlesrv" ];
then
    judge_process_exist_and_kill "battlesrv"
    judge_succ_or_fail $? "battlesrv stop"

elif [ $server_type == "modelpool" ];
then
    # modelpool是调用第三方组件安全关闭
    sh thirdparty/model_pool_go/op/stop.sh
    judge_succ_or_fail $? "modelpool stop"

else
    echo -e "\033[31m useage: sh tools/stop.sh all|actor|learner|aisrv|interface_server|battlesrv|modelpool, such as: sh tools/stop.sh all \033[0m"

    exit -1
fi
