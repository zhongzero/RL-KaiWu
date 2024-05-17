#!/bin/bash

# 采用OpenMPI运行learner进程, 执行梯度参数同步, 集群版本才使用
# 在多台机器上执行的前提，包括：
# 容器环境启动, 需要各个容器的配置一样
# 容器之间采用ssh key免密登录

if [ $# -ne 1 ];
then
    echo "useage: sh run_mulit_learner_by_openmpirun.sh debug|release, such as: sh run_mulit_learner_by_openmpirun.sh release"
    exit
fi
Debug_release=$1

# 格式机器IP:卡数目
Nodelist="127.0.0.1:1"
# 进程数目, 主要看learner进程数量
Num_process=1
Work_dir=/data/projects/1v1
Ssh_key=~/.ssh/id_rsa.pub
# 需要按照实地环境来进行配置
Ssh_port=36000

# 安装openmpi后的环境配置, 注意需要和dockerfile.base里安装的实际情况配置
MPI_HOME=/usr/local/
export PATH=$PATH:${MPI_HOME}/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MPI_HOME}/lib
export MANPATH=${MANPATH}:${MPI_HOME}/share/man

# 生成mpirun的日志文件目录
if [ ! -x "$Work_dir/log/learner" ];
then
    mkdir $Work_dir/log/
    mkdir $Work_dir/log/learner
fi

# 获取网卡, 在不同的机型上需要按照实地情况来获取
net_card_name=eth1
result=`cat /etc/os-release | grep Ubuntu`
if [[ "$result" != "" ]]
then
    net_card_name=eth0
    echo "Ubuntu system, net_card_name is" $net_card_name
else
    echo "tlinux system, net_card_name is" $net_card_name
fi

if [ $Debug_release == "release" ];
then
# 正式线上环境使用, 后台进程, 不打印日志打屏幕, 不采集timeline
    mpirun --allow-run-as-root --prefix ${MPI_HOME} -np ${Num_process} -H ${Nodelist} -bind-to none -map-by slot \
    --mca btl_openib_want_cuda_gdr 1 -mca coll_fca_enable 0 \
    --report-bindings --display-map --mca btl_openib_rroce_enable 1 --mca pml ob1 --mca btl ^openib \
    --mca btl_openib_cpc_include rdmacm  --mca coll_hcoll_enable 0  --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args "-p ${Ssh_port}" \
    --mca orte_keep_fqdn_hostnames t \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_SOCKET_IFNAME=$net_card_name \
    -x NCCL_DEBUG=INFO -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_HCA=mlx5_2:1,mlx5_3:1 -x NCCL_IB_SL=3 -x NCCL_NET_GDR_READ=1 \
    -x NCCL_CHECK_DISABLE=1  -x NCCL_LL_THRESHOLD=16384 -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 -x HOROVOD_FUSION_THRESHOLD=1 -x HOROVOD_CYCLE_TIME=0.5  -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH  python3 $Work_dir/framework/server/learner/learner.py --conf=$Work_dir/conf/framework/learner.toml \
    --variable_update horovod \
    --custom_dataformat True >$Work_dir/log/learner/mpirun.log 2>&1 &
else
    # 开发测试环境使用, 非后台进程, 采集timeline
    HOROVOD_TIMELINE_MARK_CYCLES=1 HOROVOD_TIMELINE=$Work_dir/log/learner/horovod.timeline \ 
    mpirun --allow-run-as-root --prefix ${MPI_HOME} --verbose -np ${Num_process} -H ${Nodelist} -bind-to none -map-by slot \
    --mca btl_openib_want_cuda_gdr 1 -mca coll_fca_enable 0 \
    --report-bindings --display-map --mca btl_openib_rroce_enable 1 --mca pml ob1 --mca btl ^openib \
    --mca btl_openib_cpc_include rdmacm  --mca coll_hcoll_enable 0  --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args "-p ${Ssh_port}" \
    --mca orte_keep_fqdn_hostnames t \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_SOCKET_IFNAME=$net_card_name \
    -x NCCL_DEBUG=DEBUG -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_HCA=mlx5_2:1,mlx5_3:1 -x NCCL_IB_SL=3 -x NCCL_NET_GDR_READ=1 \
    -x NCCL_CHECK_DISABLE=1  -x NCCL_LL_THRESHOLD=16384 -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 -x HOROVOD_FUSION_THRESHOLD=1 -x HOROVOD_CYCLE_TIME=0.5  -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH  python3 $Work_dir/framework/server/learner/learner.py --conf=$Work_dir/conf/framework/learner.toml \
    --variable_update horovod \
    --custom_dataformat True >$Work_dir/log/learner/mpirun.log 2>&1 &
fi

if [ $? -ne 0 ]; 
then
    echo -e "\033[31m run_mulit_learner_by_openmpirun $Debug_release start fail \033[0m"
else
    echo -e "\033[32m run_mulit_learner_by_openmpirun $Debug_release start success \033[0m"
fi
