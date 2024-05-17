#!/bin/bash

# learner_cpp_server_start.sh 主要用于拉起C++常驻进程

chmod +x tools/common.sh
. tools/common.sh

# 删除以前的共享内存
rm -rf /dev/shm/*

cd /data/projects/kaiwu-fwk/framework/server/cpp/dist/learner

# 默认绑在第一个核上, 但是需要根据机器具体情况进行, 推荐绑在最后开始数的CPU核上
myarray=()
args=$(echo ${myarray[*]})
# GPU机器采用get_cpu_ids_by_lxcfs
cpu_ids=($(get_cpu_ids_by_lxcfs $args))

array_len=${#cpu_ids[@]}
read_zmq_work_bind_cpu_idx=${cpu_ids[0]}
write_zmq_work_bind_cpu_idx=${cpu_ids[0]}
predict_bind_cpu_idx=${cpu_ids[0]}
pre_data_bind_cpu_idx=${cpu_ids[0]}
post_data_bind_cpu_idx=${cpu_ids[0]}

if [ $array_len -ge 5 ];
then
    count=0
    for cpu in ${cpu_ids[@]}
    do
        if [ $count -eq 0 ];
        then
            read_zmq_work_bind_cpu_idx=$cpu
            write_zmq_work_bind_cpu_idx=$cpu
            predict_bind_cpu_idx=$cpu
            pre_data_bind_cpu_idx=$cpu
            post_data_bind_cpu_idx=$cpu
        else
            read_zmq_work_bind_cpu_idx="$read_zmq_work_bind_cpu_idx, $cpu"
            write_zmq_work_bind_cpu_idx="$write_zmq_work_bind_cpu_idx, $cpu"
            predict_bind_cpu_idx="$predict_bind_cpu_idx, $cpu"
            pre_data_bind_cpu_idx="$pre_data_bind_cpu_idx, $cpu"
            post_data_bind_cpu_idx="$post_data_bind_cpu_idx, $cpu"
        fi

        let count++
    done
fi

# 确保日志文件存在
learner_cpp_server_log_dir=/data/projects/kaiwu-fwk/
if [ ! -x "$actor_cpp_server_log_dir/log/" ];
then
    mkdir $actor_cpp_server_log_dir/log/
fi
if [ ! -x "$actor_cpp_server_log_dir/log/learner" ];
then
    mkdir $actor_cpp_server_log_dir/log/learner
fi