#!/bin/bash

# 查看机器上docker image

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 0 ] && [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_search_image.sh [keyword]\033[0m"
    exit -1
fi

keyword=$1

if [ -n "$keyword" ]; 
then
    docker images | grep $keyword
else
    docker images
fi