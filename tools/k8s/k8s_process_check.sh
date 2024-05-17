#!/bin/bash
# 进程查看脚本, 在k8s上用于查看进程的数目, 比如battlesrv和aisrv数量

if [ $# -ne 1 ] && [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/k8s/check_k8s_process.sh process_name[aisrv|battlesrv|all] [detail], \n such as: sh tools/k8s/check_k8s_process.sh aisrv \n or sh tools/k8s/check_k8s_process.sh battlesrv detail\033[0m"
    
    exit -1
fi

process_name=$1
detail=$2
if [ $process_name == "aisrv" ];
then
    echo -e "\033[32m $process_name Running count:  \033[0m"
    if [ ! -n "$detail" ];
    then
        kubectl get pod -o wide -n kaiwu-drl-prod | grep aisrv | grep sgame-5v5 | grep Running  | wc -l
    else
        kubectl get pod -o wide -n kaiwu-drl-prod | grep aisrv | grep sgame-5v5 | grep Running
    fi
    
elif [ $process_name == "battlesrv" ];
then
    echo -e "\033[32m $process_name Running count:  \033[0m"
    if [ ! -n "$detail" ];
    then
        kubectl get pod -o wide -n kaiwu-drl-prod | grep battlesrv | grep sgame-5v5 | grep Running  | wc -l
    else
        kubectl get pod -o wide -n kaiwu-drl-prod | grep battlesrv | grep sgame-5v5 | grep Running
    fi
elif [ $process_name == "all" ];
then
    echo -e "\033[32m aisrv Running count:  \033[0m"
    if [ ! -n "$detail" ];
    then
        kubectl get pod -o wide -n kaiwu-drl-prod | grep aisrv | grep sgame-5v5 | grep Running  | wc -l
    else
        kubectl get pod -o wide -n kaiwu-drl-prod | grep aisrv | grep sgame-5v5 | grep Running
    fi

    echo -e "\033[32m battlesrv Running count:  \033[0m"
    if [ ! -n "$detail" ];
    then
        kubectl get pod -o wide -n kaiwu-drl-prod | grep battlesrv | grep sgame-5v5 | grep Running  | wc -l
    else
        kubectl get pod -o wide -n kaiwu-drl-prod | grep battlesrv | grep sgame-5v5 | grep Running
    fi
else
    echo -e "\033[31m useage: sh tools/k8s/check_k8s_process.sh process_name[aisrv|battlesrv|all] [detail], \n such as: sh tools/k8s/check_k8s_process.sh aisrv \n or sh tools/k8s/check_k8s_process.sh battlesrv detail\033[0m"
    
    exit -1
fi
