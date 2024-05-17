#!/bin/bash

# 进程部署脚本, 主要内容:
# 1. 拉取docker镜像
# 2. 启动dokcer

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 5 ] && [ $# -ne 6 ];
then
    echo -e "\033[31m useage: sh tools/deploy.sh all|cpu|gpu version kaiwu-1v1|kaiwu-5v5 gpu_machine_type process_name [time], such as: sh tools/deploy.sh gpu 1 kaiwu-1v1 A100 learner [20220830]\033[0m"
    exit -1
fi

deploy_type=$1
version=$2
sgame_type=$3
gpu_machine=$4
process_name=$5
time=$6

if [ $deploy_type != "cpu" ] && [ $deploy_type != "gpu" ] && [ $deploy_type != "all" ];
then
    echo -e "\033[31m useage: sh tools/deploy.sh all|cpu|gpu version kaiwu-1v1|kaiwu-5v5 gpu_machine_type process_name [time], such as: sh tools/deploy.sh gpu 1 kaiwu-1v1 A100 learner [20220830]\033[0m"
    exit -1
fi

# 修改为这里的用户名和密码即可
#docker login --username username --password password mirrors.tencent.com
docker login --username fengjunyang --password e0456f6e00e911edb8a36225058ed3dc mirrors.tencent.com
judge_succ_or_fail $? "docker login "

if [ -z $time ]
then
    # 如果没有设置time, 默认拉取当天的
    time=$(date "+%Y%m%d")
fi

if [ $deploy_type == "gpu" ] || [ $deploy_type == "all" ];
then
    image_name="mirrors.tencent.com/kaiwudrl/${sgame_type}:${deploy_type}_${gpu_machine}_${time}_v${version}"
else
    image_name="mirrors.tencent.com/kaiwudrl/${sgame_type}:${deploy_type}_${time}_v${version}"
fi

docker pull $image_name
judge_succ_or_fail $? "docker pull $image_name "

if [ $deploy_type == "gpu" ] || [ $deploy_type == "all" ];
then
    docker_image_id=`docker images | grep -w ${deploy_type}_${gpu_machine}_${time}_v${version} | awk '{print $3}'`
else
    docker_image_id=`docker images | grep -w ${deploy_type}_${time}_v${version} | awk '{print $3}'`
fi

if [ -z "$docker_image_id" ]
then
    echo -e "\033[31m docker image id is null \033[0m"

    exit -1
fi

# 因为有些GPU环境调试时可能会采用all镜像, 故这里加上--gpus all
if [ $deploy_type == "gpu" ] || [ $deploy_type == "all" ];
then
    docker run --privileged --gpus all --shm-size="4g" --net=host --name $process_name -it $docker_image_id /bin/bash
else
    docker run --privileged --net=host --name $process_name -it $docker_image_id /bin/bash
fi

judge_succ_or_fail $? "docker run $docker_image_id "
