#!/bin/bash
# 登录容器脚本

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/k8s/k8s_process_login.sh pod_name, \n such as: sh tools/k8s/k8s_process_login.sh kaiwu-aisrv-nj-sgame-5v5-56b4dfd9df-zjc42 \033[0m"
    
    exit -1
fi

pod_name=$1

kubectl -n kaiwu-drl-prod exec $pod_name -it /bin/bash