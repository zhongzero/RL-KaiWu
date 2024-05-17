#!/bin/bash

# 删除机器上docker image rm

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_rm_image.sh [image_id]\033[0m"
    exit -1
fi

image_id=$1

docker image rm $image_id
