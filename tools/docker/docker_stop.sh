#!/bin/bash

# 删除机器上docker stop 和docker rm 

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_stop.sh [container_id]\033[0m"
    exit -1
fi

container_id=$1

docker stop $container_id
docker rm $container_id