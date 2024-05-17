#!/bin/bash

# GPU机器拉起tensorboard功能



chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/start_tensorboard.sh ip summay_dir \
    \n such as: sh tools/start_tensorboard.sh 127.0.0.1 /data/summary/sgame_5v5_ppo/  \033[0m"

    exit -1
fi

ip=$1
summary_dir=$2

# 注意tensorboard文件路径
/usr/local/python-3.7/bin/tensorboard serve --host $ip --logdir $summary_dir
judge_succ_or_fail $? "start tensorboard, if success, please visit http://$ip:6006/"