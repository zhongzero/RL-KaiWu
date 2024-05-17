#!/bin/bash
# 查看机器上指定端口下的IP连接, 哪些是有请求和响应包的

if [ $# -ne 1 ] && [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/machine_ip_port_check.sh port, such as: sh tools/machine_ip_port_check.sh 8888 [show_valid_message]\033[0m"
    
    exit -1
fi

port=$1
if [ $# -eq 2 ];
then
    show_valid_message=$2
else
    show_valid_message=""
fi

if [ "$show_valid_message" == "show_valid_message" ];
then
    netstat -anp | grep $port | grep -v "0      0"
    count=`netstat -anp | grep $port | grep -v "0      0" | wc -l`

    echo -e "\033[32m vaild message count: $count  \033[0m" 
else
    netstat -anp | grep $port
fi
