#!/bin/bash


# 免密登录机器间批量传输文件/执行命令

chmod +x tools/common.sh
. tools/common.sh

# 参数如下:
# IP列表, 采用逗号分割
# 传输的文件列表, 采用逗号分割, 将该文件从A机器传递到B机器上, 文件位置是一样的
# 传输文件或者执行命令, 如果是传输文件则文件列表file_list有效, 如果是执行命令, 则执行命令cmd_list有效, cmd_list是按照;号分割
if [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/auto_scp_and_cmd.sh scp|cmd ip_list file_list|cmd_list, \
    \n such as sh tools/auto_scp_and_cmd.sh scp 127.0.0.1,127.0.0.2 file1,file2 \
    \n or sh tools/auto_scp_and_cmd.sh cmd 127.0.0.1,127.0.0.2 ls;pwd  \033[0m"

    exit -1
fi

# 传输文件或者执行命令
scp_or_cmd=$1

# 读取IP配置文件
ip_list=$2

ip_lists=(`echo $ip_list | tr ',' ' '`)

count=0
# 读取传输文件列表文件
if [ $scp_or_cmd == "scp" ];
then
    file_list=$3
    file_lists=(`echo $file_list | tr ',' ' '`)

    # 遍历IP列表和文件进行传输
    for ip in ${ip_lists[@]}
    do
        for file in  ${file_lists[@]}
        do 
            scp -P 36000 -r $file root@$ip:$file
            judge_succ_or_fail $? "scp to $ip of $file "
            let count++
        done
    done

    judge_succ_or_fail 0 "scp $count, please see the log"
elif  [ $scp_or_cmd == "cmd" ];
then
    cmd_list=$3

    # 遍历IP列表和命令执行
    for ip in ${ip_lists[@]}
    do
        echo $(ssh -o ConnectTimeout=10 -t root@$ip "$cmd_list")
        judge_succ_or_fail $? "cmd to $ip of $cmd_list"
        let count++
    done

    judge_succ_or_fail 0 "cmd $count, please see the log"
else
    echo -e "\033[31m useage: sh tools/auto_scp_and_cmd.sh scp|cmd ip_list file_list|cmd_list, \
    \n such as sh tools/auto_scp_and_cmd.sh scp 127.0.0.1,127.0.0.2 file1,file2 \
    \n or sh tools/auto_scp_and_cmd.sh cmd 127.0.0.1,127.0.0.2 ls;pwd  \033[0m"

    exit -1
fi
