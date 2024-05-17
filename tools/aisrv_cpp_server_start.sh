#!/bin/bash


# aisrv_cpp_server_start.sh 主要用于拉起C++常驻进程

chmod +x tools/common.sh
. tools/common.sh

# 删除以前的共享内存
rm -rf /dev/shm/*

cd /data/projects/kaiwu-fwk/framework/server/cpp/dist/aisrv

# C++需要和python端的设置一致才可以采用共享内存通信
export G6SHMNAME=KaiwuDRL 

#c++调用python函数需要配置环境变量
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/framework/server/cpp/dist/aisrv

# 默认绑在第一个核上, 但是需要根据机器具体情况进行, 推荐绑在最后开始数的CPU核上
myarray=()
args=$(echo ${myarray[*]})
# CPU机器采用get_cpu_ids_by_lxcfs, cgroup分配
cpu_ids=($(get_cpu_ids_by_lxcfs $args))

array_len=${#cpu_ids[@]}
actor_proxy_bind_cpu_idx=${cpu_ids[0]}
learner_proxy_bind_cpu_idx=${cpu_ids[0]}
aisrv_socket_server_bind_cpu_idx=${cpu_ids[0]}
sample_server_bind_cpu_idx=${cpu_ids[0]}
kaiwu_rl_helper_bind_cpu_idx=${cpu_ids[0]}

if [ $array_len -ge 5 ];
then
    count=0
    for cpu in ${cpu_ids[@]}
    do
        if [ $count -eq 0 ];
        then
            actor_proxy_bind_cpu_idx=$cpu
            learner_proxy_bind_cpu_idx=$cpu
            aisrv_socket_server_bind_cpu_idx=$cpu
            sample_server_bind_cpu_idx=$cpu
            kaiwu_rl_helper_bind_cpu_idx=$cpu
        else
            actor_proxy_bind_cpu_idx="$actor_proxy_bind_cpu_idx, $cpu"
            learner_proxy_bind_cpu_idx="$learner_proxy_bind_cpu_idx, $cpu"
            aisrv_socket_server_bind_cpu_idx="$aisrv_socket_server_bind_cpu_idx, $cpu"
            sample_server_bind_cpu_idx="$sample_server_bind_cpu_idx, $cpu"
            kaiwu_rl_helper_bind_cpu_idx="$kaiwu_rl_helper_bind_cpu_idx, $cpu"
        fi

        let count++
    done
fi

# 确保日志文件存在
aisrv_cpp_server_log_dir=/data/projects/kaiwu-fwk/
if [ ! -x "$aisrv_cpp_server_log_dir/log/" ];
then
    mkdir $aisrv_cpp_server_log_dir/log/
fi
if [ ! -x "$aisrv_cpp_server_log_dir/log/aisrv" ];
then
    mkdir $aisrv_cpp_server_log_dir/log/aisrv
fi

# 修改配置文件里的值, 主要是修改绑核逻辑
aisrv_cpp_server_conf=/data/projects/kaiwu-fwk/framework/server/cpp/conf/aisrv_server.ini
sed -i '/--actor_proxy_bind_cpu_idx/d' $aisrv_cpp_server_conf
sed -i '/--learner_proxy_bind_cpu_idx/d' $aisrv_cpp_server_conf
sed -i '/--aisrv_socket_server_bind_cpu_idx/d' $aisrv_cpp_server_conf
sed -i '/--sample_server_bind_cpu_idx/d' $aisrv_cpp_server_conf
sed -i '/--kaiwu_rl_helper_bind_cpu_idx/d' $aisrv_cpp_server_conf

# 注意不要和已经有的配置项格式冲突
echo -e "\n--actor_proxy_bind_cpu_idx=$actor_proxy_bind_cpu_idx" >> $aisrv_cpp_server_conf
echo -e "\033[32m actor_proxy_bind_cpu_idx  is $actor_proxy_bind_cpu_idx \033[0m"
echo "--learner_proxy_bind_cpu_idx=$learner_proxy_bind_cpu_idx" >> $aisrv_cpp_server_conf
echo -e "\033[32m learner_proxy_bind_cpu_idx  is $learner_proxy_bind_cpu_idx \033[0m"
echo "--aisrv_socket_server_bind_cpu_idx=$aisrv_socket_server_bind_cpu_idx" >> $aisrv_cpp_server_conf
echo -e "\033[32m aisrv_socket_server_bind_cpu_idx  is $aisrv_socket_server_bind_cpu_idx \033[0m"
echo "--sample_server_bind_cpu_idx=$sample_server_bind_cpu_idx" >> $aisrv_cpp_server_conf
echo -e "\033[32m sample_server_bind_cpu_idx  is $sample_server_bind_cpu_idx \033[0m"
echo "--kaiwu_rl_helper_bind_cpu_idx=$kaiwu_rl_helper_bind_cpu_idx" >> $aisrv_cpp_server_conf
echo -e "\033[32m kaiwu_rl_helper_bind_cpu_idx  is $kaiwu_rl_helper_bind_cpu_idx \033[0m"

# 注意判断interface_server_cnt值是否相同, aisrv和interface_server必须一致, 否则会导致部分特征值处理没有interface_server发送
# 注意判断unix_ipc_path值是否相同, aisrv和interface_server必须一致, 否则无法通信
interface_server_conf=/data/projects/kaiwu-fwk/framework/server/cpp/conf/interface_server.ini

tmpfile1=$(mktemp)
tmpfile2=$(mktemp)
grep -E -- "--interface_server_cnt|--unix_ipc_path" $aisrv_cpp_server_conf > $tmpfile1
grep -E -- "--interface_server_cnt|--unix_ipc_path" $interface_server_conf > $tmpfile2
diff=`diff $tmpfile1 $tmpfile2`
rm -f $tmpfile1 $tmpfile2

if [ -n "$diff" ]; 
then
    echo -e "\033[31m $aisrv_cpp_server_conf and $interface_server_conf's interface_server_cnt is not same, please check \033[0m"
    exit -1
else
    echo -e "\033[32m $aisrv_cpp_server_conf and $interface_server_conf's interface_server_cnt is same \033[0m"
fi

cd /data/projects/kaiwu-fwk/
# aisrv启动前先启动interface_server
. tools/interface_server_start.sh

# 随机休息5-10秒, 规避interface_server没有启动成功导致aisrv无法连接interface_server
sleep_time=$((RANDOM%6+5))
sleep $sleep_time

cd /data/projects/kaiwu-fwk/framework/server/cpp/dist/aisrv
# 这里注意日志输出量的估算问题, 以免将磁盘空间占用完
./aisrv_cpp_server --flagfile /data/projects/kaiwu-fwk/framework/server/cpp/conf/aisrv_server.ini >$aisrv_cpp_server_log_dir/log/aisrv/aisrv_cpp_server.log 2>&1 &
judge_succ_or_fail $? "aisrv_cpp_server start"
