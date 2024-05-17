#!/bin/bash


# actor_cpp_server_start.sh 主要用于拉起C++常驻进程

chmod +x tools/common.sh
. tools/common.sh

# 根据不同的GPU机型到不同的文件目录
check_gpu_machine_type
gpu_machine_type=$result

cd /data/projects/kaiwu-fwk/framework/server/cpp/dist/actor/$gpu_machine_type

# C++需要和python端的设置一致才可以采用共享内存通信
export G6SHMNAME=KaiwuDRL 

#c++调用python函数需要配置环境变量
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/framework/server/cpp/dist/actor/
export PYTHONPATH=$PYTHONPATH:/data/projects/kaiwu-fwk/framework/server/cpp/dist/actor/$gpu_machine_type

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
actor_cpp_server_log_dir=/data/projects/kaiwu-fwk/
if [ ! -x "$actor_cpp_server_log_dir/log/" ];
then
    mkdir $actor_cpp_server_log_dir/log/
fi
if [ ! -x "$actor_cpp_server_log_dir/log/actor" ];
then
    mkdir $actor_cpp_server_log_dir/log/actor
fi

# 修改配置文件里的值, 主要是修改绑核逻辑
actor_cpp_server_conf=/data/projects/kaiwu-fwk/framework/server/cpp/conf/actor_server.ini
sed -i '/--read_zmq_work_bind_cpu_idx/d' $actor_cpp_server_conf
sed -i '/--write_zmq_work_bind_cpu_idx/d' $actor_cpp_server_conf
sed -i '/--predict_bind_cpu_idx/d' $actor_cpp_server_conf
sed -i '/--pre_data_bind_cpu_idx/d' $actor_cpp_server_conf
sed -i '/--post_data_bind_cpu_idx/d' $actor_cpp_server_conf

# 注意不要和已经有的配置项格式冲突
echo -e "\n--read_zmq_work_bind_cpu_idx=$read_zmq_work_bind_cpu_idx" >> $actor_cpp_server_conf
echo -e "\033[32m read_zmq_work_bind_cpu_idx  is $read_zmq_work_bind_cpu_idx \033[0m"
echo "--write_zmq_work_bind_cpu_idx=$write_zmq_work_bind_cpu_idx" >> $actor_cpp_server_conf
echo -e "\033[32m write_zmq_work_bind_cpu_idx  is $write_zmq_work_bind_cpu_idx \033[0m"
echo "--predict_bind_cpu_idx=$predict_bind_cpu_idx" >> $actor_cpp_server_conf
echo -e "\033[32m predict_bind_cpu_idx  is $predict_bind_cpu_idx \033[0m"
echo "--pre_data_bind_cpu_idx=$pre_data_bind_cpu_idx" >> $actor_cpp_server_conf
echo -e "\033[32m pre_data_bind_cpu_idx  is $pre_data_bind_cpu_idx \033[0m"
echo "--post_data_bind_cpu_idx=$post_data_bind_cpu_idx" >> $actor_cpp_server_conf
echo -e "\033[32m post_data_bind_cpu_idx  is $post_data_bind_cpu_idx \033[0m"

# 这里注意日志输出量的估算问题, 以免将磁盘空间占用完
./actor_cpp_server --flagfile /data/projects/kaiwu-fwk/framework/server/cpp/conf/actor_server.ini >$actor_cpp_server_log_dir/log/actor/actor_cpp_server.log 2>&1 &
judge_succ_or_fail $? "actor_cpp_server start"
