#!/bin/bash


# interface_server_start.sh 主要用于拉起C++常驻进程

chmod +x tools/common.sh
. tools/common.sh

cd /data/projects/kaiwu-fwk/framework/server/cpp/dist/interface_server

# C++需要和python端的设置一致才可以采用共享内存通信
export G6SHMNAME=KaiwuDRL 

#c++调用python函数需要配置环境变量
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/framework/server/cpp/dist/interface_server

# 默认绑在第一个核上, 但是需要根据机器具体情况进行, 推荐绑在最后开始数的CPU核上
myarray=()
args=$(echo ${myarray[*]})
# CPU机器采用get_cpu_ids_by_lxcfs, cgroup分配
cpu_ids=($(get_cpu_ids_by_lxcfs $args))

array_len=${#cpu_ids[@]}
interface_server_bind_cpu_idx=${cpu_ids[0]}

if [ $array_len -ge 1 ];
then
    count=0
    for cpu in ${cpu_ids[@]}
    do
        if [ $count -eq 0 ];
        then
            interface_server_bind_cpu_idx=$cpu
        else
            interface_server_bind_cpu_idx="$interface_server_bind_cpu_idx, $cpu"
        fi

        let count++
    done
fi

# 确保日志文件存在
interface_server_log_dir=/data/projects/kaiwu-fwk/
if [ ! -x "$interface_server_log_dir/log/" ];
then
    mkdir $interface_server_log_dir/log/
fi
if [ ! -x "$interface_server_log_dir/log/interface_server" ];
then
    mkdir $interface_server_log_dir/log/interface_server
fi

# 修改配置文件里的值, 主要是修改绑核逻辑
interface_server_conf=/data/projects/kaiwu-fwk/framework/server/cpp/conf/interface_server.ini
sed -i '/--interface_server_bind_cpu_idx/d' $interface_server_conf

# 注意不要和已经有的配置项格式冲突
echo -e "\n--interface_server_bind_cpu_idx=$interface_server_bind_cpu_idx" >> $interface_server_conf
echo -e "\033[32m interface_server_bind_cpu_idx  is $interface_server_bind_cpu_idx \033[0m"

# 删除以前的unix_ipc_path的相关文件
unix_ipc_path=`grep -oP '(?<=--unix_ipc_path=).*' $interface_server_conf`
if [ ! -n "$unix_ipc_path" ];
then
    echo -e "\033[31m unix_ipc_path is empty\033[0m"
    exit -1
fi
rm -rf $unix_ipc_path*

# 这里注意日志输出量的估算问题, 以免将磁盘空间占用完
./interface_server --flagfile /data/projects/kaiwu-fwk/framework/server/cpp/conf/interface_server.ini >$interface_server_log_dir/log/interface_server/interface_server.log 2>&1 &
judge_succ_or_fail $? "interface_server start"
