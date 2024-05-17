#!/bin/bash

# 下面是modelpool部署方式:
# 1. 角色分为actor和learner上容器的进程 
# 2. 主learner(其他learner不需要启动)进程, 启动sh start.sh learner 
# 3. 每个actor进程, 启动sh start.sh actor

if [ $# -ne 1 ];
then
   echo "usage sh start.sh actor|learner, such as: sh start.sh learner"
   exit -1
fi

role=$1

if [ -d "../log" ]; 
then
    rm -r ../log
fi
mkdir ../log

# actor进程
if [ $role == "actor" ];
then
   # 获取master_ip, 注意是从config下的gpu.iplist获取的, 而gpu.iplist 是需要配置的, 每行一个IP
   master_ip=`head -n 1 ../config/gpu.iplist | awk '{print $1}'`
   bash set_actor_config.sh $master_ip
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > ../log/cpu.log 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=./model > ../log/proxy.log 2>&1 &
# learner进程
elif [ $role == "learner" ];
then
   bash set_learner_config.sh
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > ../log/gpu.log 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=./model > ../log/proxy.log 2>&1 &
else
   echo "usage sh start.sh actor|learner, such as: sh start.sh learner"
   exit -1
fi
