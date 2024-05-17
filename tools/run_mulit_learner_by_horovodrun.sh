#!/bin/bash

# 采用horovodrun运行learner进程, 执行梯度参数同步, 集群版本才使用
# 在多台机器上执行的前提，包括：
# 容器环境启动, 需要各个容器的配置一样
# 容器之间采用ssh key免密登录

if [ $# -ne 1 ];
then
    echo "useage: sh run_mulit_learner_by_horovodrun.sh debug|release, such as: sh run_mulit_learner_by_horovodrun.sh release"
    exit
fi
Debug_release=$1

# 格式机器IP:卡数目
Nodelist=9.134.253.89:1,9.134.253.27:1
# 进程数目, 主要看learner进程数量
Num_process=2
Work_dir=/data/projects/1v1
Ssh_key=~/.ssh/id_rsa.pub
# 需要按照实地环境来进行配置
Ssh_port=36000

# 生成horovodrun的日志文件目录
if [ ! -x "$Work_dir/log/learner" ];
then
    mkdir $Work_dir/log/
    mkdir $Work_dir/log/learner
fi

# 获取网卡, 在不同的机型上需要按照实地情况来获取
net_card_name=eth1
result=`cat /etc/os-release | grep Ubuntu`
if [[ "$result" != "" ]]
then
    net_card_name=eth0
    echo "Ubuntu system, net_card_name is" $net_card_name
else
    echo "tlinux system, net_card_name is" $net_card_name
fi

if [ $Debug_release == "release" ];
then
    # 正式线上环境使用, 后台进程, 不打印日志打屏幕
    nohup horovodrun -np ${Num_process} -H ${Nodelist} --ssh-port 36000 --ssh-identity-file $Ssh_key \
    python3 $Work_dir/framework/server/learner/learner.py --conf=$Work_dir/conf/framework/learner.toml >$Work_dir/log/learner/horovodrun.log 2>&1 &
else
    # 开发测试环境使用, 非后台进程, 打印日志到屏幕
    horovodrun -np ${Num_process} -H ${Nodelist} --ssh-port 36000 --ssh-identity-file $Ssh_key \
    python3 $Work_dir/framework/server/learner/learner.py --conf=$Work_dir/conf/framework/learner.toml 
fi

if [ $? -ne 0 ]; 
then
    echo -e "\033[31m run_mulit_learner_by_horovodrun $Debug_release start fail \033[0m"
else
    echo -e "\033[32m run_mulit_learner_by_horovodrun $Debug_release start success \033[0m"
fi