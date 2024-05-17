#!/bin/bash

# modelpool进程启动脚本

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/start_modelpool.sh actor|learner, such as sh tools/start_modelpool.sh learner
    如果是actor和learner进程在同一个容器上, 只需要传递learner即可 \033[0m"

    exit -1
fi

# 启动modelpool进程
# 参数process, 主要区分actor和learner
# 启动前, 删除前次遗留的文件
process=$1

# 删除掉上次运行后遗留的文件
cd thirdparty/model_pool_go/bin/
rm -rf files/*
rm -rf model/*
judge_succ_or_fail $? "modelpool $process old file delete"

# 注意文件目录路径位置
cd ../op/
sh start.sh $process
judge_succ_or_fail $? "modelpool $process start"
cd /data/projects/1v1
