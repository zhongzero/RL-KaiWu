#!/bin/bash

# 进程启动脚本
# 1. aisrv, 如果传入的有actor和learner的IP列表, 则会生成配置文件
# 2. actor, 为了简化k8s启动流程, 会主动拉起来modelpool actor进程
# 3. learner, 分为cluster和single, cluster是正式线上环境使用, 为了简化k8s启动流程, 会主动拉起来modelpool learner进程
# 4. all, 这种场景是开发测试使用, 故会启动aisrv, actor, learner single, modelpool learner

chmod +x tools/common.sh
. tools/common.sh

chmod +x tools/produce_config.sh
. tools/produce_config.sh

chmod +x tools/change_linux_parameter.sh
. tools/change_linux_parameter.sh

# 参数的场景比较多, 主要分为:
# 进程名
# 是否需要生产配置文件, config即生成配置文件, 其他的字符串不用生成配置文件
# 1V1还是5V5模式, 即sgame_5v5、sgame_1v1
# 是否是self-play模式
# 生成配置文件, 即需要actor_list和learner_list参数
# 集群版本还是单机版本, learner单独的配置
# 生成版本还是测试版本, learner单独的配置
if [ $# -ne 2 ] && [ $# -ne 3 ] && [ $# -ne 4 ] && [ $# -ne 5 ] && [ $# -ne 6 ] && [ $# -ne 7 ] && [ $# -ne 8 ] && [ $# -ne 9 ];
then
    echo -e "\033[31m useage: sh tools/start.sh all|actor|aisrv config sgame_5v5 self_play actor_list learner_list [with_client|without_client], \
     \n or sh tools/start.sh learner cluster release config sgame_5v5 self_play actor_list learner_list \
     \n such as: sh tools/start.sh all config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh learner cluster release config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh aisrv no_config sgame_5v5 no_self_play \
     \n or sh tools/start.sh client no_config sgame_5v5 no_self_play \
     \n or sh tools/start.sh aisrv config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 with_client \
     \n or sh tools/start.sh learner cluster release no_config sgame_5v5 no_self_play \033[0m"

    exit -1
fi

# 启动modelpool进程
# 参数process, 主要区分actor和learner
# 启动前, 删除前次遗留的文件
start_modelpool()
{
    process=$1

    # 删除掉上次运行后遗留的文件
    cd thirdparty/model_pool_go/bin/
    rm -rf files/*
    rm -rf model/*
    judge_succ_or_fail $? "modelpool $process old file delete"

    # 注意文件目录路径
    cd ../op/
    sh start.sh $process
    judge_succ_or_fail $? "modelpool $process start"
    cd /data/projects/1v1
}

server_type=$1
produce_config=$2

# 无论哪种进程启动, 先修改下linux内核参数, 注意部分机器适配的情况
change_tcp_parameter
judge_succ_or_fail $? "change_tcp_parameter"

# 删除已经有的core文件
rm -rf /data/corefile/*

# 进程启动日志, 输入到日志文件里, 规避出现问题时不知道报错信息
Work_dir=/data/projects/1v1
aisrv_start_file=$Work_dir/log/aisrv.log
actor_start_file=$Work_dir/log/actor.log
learner_start_file=$Work_dir/log/learner.log
client_start_file=$Work_dir/log/client.log

# 配置protobuf的路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/protobuf/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/protobuf/lib
export PATH=$PATH:/usr/local/protobuf/bin 

make_log_dir()
{
    if [ ! -x "$Work_dir/log/" ];
    then
        mkdir $Work_dir/log/
    fi
}

# 下面日志文件, 如果日志目录没有在则新建, 在则清空日志目录, 规避日志多导致机器磁盘紧张
make_aisrv_log_dir()
{
    make_log_dir
    if [ ! -x "$Work_dir/log/aisrv" ];
    then
        mkdir $Work_dir/log/aisrv
    else
        find $Work_dir/log/aisrv/ -type f -exec rm -f {} +
    fi
}

make_actor_log_dir()
{
    make_log_dir
    if [ ! -x "$Work_dir/log/actor" ];
    then
        mkdir $Work_dir/log/actor
    else
        find $Work_dir/log/actor/ -type f -exec rm -f {} +
    fi
}

make_learner_log_dir()
{
    make_log_dir
    if [ ! -x "$Work_dir/log/learner" ];
    then
        mkdir $Work_dir/log/learner
    else
        find $Work_dir/log/learner/ -type f -exec rm -f {} +
    fi
}

make_client_log_dir()
{
    make_log_dir
    if [ ! -x "$Work_dir/log/client" ];
    then
        mkdir $Work_dir/log/client
    else
        find $Work_dir/log/client/ -type f -exec rm -f {} +
    fi
}


if [ $server_type == "aisrv" ];
then
    source /etc/profile
    ldconfig

    sgame_type=$3
    self_play=$4

    make_aisrv_log_dir

    # 如果是需要生成配置文件, 则开始生成
    if [ $produce_config == "config" ];
    then 
        actor_ips=$5
        learner_ips=$6
        check_param_is_null "$actor_ips" "actor ips is null"
        check_param_is_null "$learner_ips" "learner ips is null"

        produce_config_by_process_name aisrv /data/projects/1v1/conf/framework/aisrv.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "aisrv produce config"
    elif [ $produce_config == "no_config" ];
    then
        echo -e "\033[32m start aisrv no need to produce config file \033[0m"

        # 单独修改self-play值
        main_configure_file=conf/framework/configure.toml
        # 如果明确需要设置再开始设置
        if [ -n "$self_play" ]
        then
            if [ "$self_play" == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi
        fi

    else
        echo -e "\033[31m produce_config only support config or no_config \033[0m"\

        exit -1
    fi

    # 为了做到进程发送数据均衡, 这里会增加随机sleep
    sleep_time=$((RANDOM%10))
    sleep $sleep_time

    # 临时增加启动进程检测脚本
    aisrv_framework=`grep '^aisrv_framework' conf/framework/aisrv.toml | cut -d '=' -f2 | tr -d ' '`
    if [ "$aisrv_framework" = "kaiwudrl" ];
    then
        sh tools/deploy/check_process.sh &
    fi

    # 主要是为了解决aisrv启动时同时拉起来sgame_client
    with_client=$7
    if [ ! -n "$with_client" ];
    then
        # 如果不用启动sgame_client, 则需要设置下aisrv启动是阻塞的, 以免容器退出
        export G6SHMNAME=KaiwuDRL && python3 framework/server/aisrv/aisrv.py --conf=/data/projects/1v1/conf/framework/aisrv.toml >$aisrv_start_file 2>&1
        judge_succ_or_fail $? "aisrv start"
    else
        if [ $with_client == "with_client" ];
        then
            make_client_log_dir
            export G6SHMNAME=KaiwuDRL && python3 framework/server/aisrv/aisrv.py --conf=/data/projects/1v1/conf/framework/aisrv.toml >>$aisrv_start_file 2>&1 &
            judge_succ_or_fail $? "aisrv start"

            # 临时拉起来sgame_client
            sleep 10s

            cd /data/projects/1v1/app/sgame_5v5/tools

            sh start_multi_game.sh 1 /data/projects/1v1/conf/framework/client.toml >$client_start_file 2>&1
            judge_succ_or_fail $? "sgame_client start"

            cd /data/projects/1v1/
        else

            # 如果不用启动sgame_client, 则需要设置下aisrv启动是阻塞的, 以免容器退出
            export G6SHMNAME=KaiwuDRL && python3 framework/server/aisrv/aisrv.py --conf=/data/projects/1v1/conf/framework/aisrv.toml >$aisrv_start_file 2>&1
            judge_succ_or_fail $? "aisrv start"
        fi
    fi

elif [ $server_type == "actor" ];
then
    sgame_type=$3
    self_play=$4

    make_actor_log_dir

    source /etc/profile
    ldconfig

    # 如果是需要生成配置文件, 则开始生成
    if [ $produce_config == "config" ];
    then
        actor_ips=$5
        learner_ips=$6
        check_param_is_null "$actor_ips" "actor ips is null"
        check_param_is_null "$learner_ips" "learner ips is null"

        produce_config_by_process_name actor /data/projects/1v1/conf/framework/actor.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "actor produce config"
    elif [ $produce_config == "no_config" ];
    then
        echo -e "\033[32m start actor no need to produce config file \033[0m"

        # 单独修改self-play值
        main_configure_file=conf/framework/configure.toml
        # 如果明确需要设置再开始设置
        if [ -n "$self_play" ]
        then
            if [ "$self_play" == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi
        fi

    else
        echo -e "\033[31m produce_config only support config or no_config \033[0m"

        exit -1
    fi

    # 为了简化k8s启动流程, 会主动拉起来modelpool actor进程
    start_modelpool "actor"

    # 处理GPU异构的情况
    check_gpu_machine_type
    gpu_machine_type=$result

    echo -e "\033[32m gpu engine is $gpu_machine_type \033[0m"
    if [ -n "$gpu_machine_type" ];
    then
        sh tools/actor_cpp_copy.sh $gpu_machine_type
    fi

    # 删除/dev/shm
    rm -rf /dev/shm/*

    export G6SHMNAME=KaiwuDRL && python3 framework/server/actor/actor.py --conf=/data/projects/1v1/conf/framework/actor.toml >$actor_start_file 2>&1
    judge_succ_or_fail $? "actor start"

# 注意learner的参数顺序, 形如sh start.sh learner cluster release config 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2
elif [ $server_type == "learner" ];
then

    # 如果是需要生成配置文件, 则开始生成
    produce_config=$4
    sgame_type=$5
    self_play=$6

    make_learner_log_dir

    source /etc/profile
    ldconfig

    if [ $produce_config == "config" ];
    then
        actor_ips=$7
        learner_ips=$8
        check_param_is_null "$actor_ips" "actor ips is null"
        check_param_is_null "$learner_ips" "learner ips is null"

        produce_config_by_process_name learner /data/projects/1v1/conf/framework/learner.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "learner produce config"
    elif [ $produce_config == "no_config" ];
    then
        echo -e "\033[32m start learner no need to produce config file \033[0m"

        # 单独修改self-play值
        main_configure_file=conf/framework/configure.toml
        # 如果明确需要设置再开始设置
        if [ -n "$self_play" ]
        then
            if [ "$self_play" == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi
        fi

    else
        echo -e "\033[31m produce_config only support config or no_config \033[0m"
        exit -1
    fi

    # 为了简化k8s启动流程, 会主动拉起来modelpool learner进程
    start_modelpool "learner"

    cluster_or_single=$2
    if [ $cluster_or_single == "cluster" ];
    then
        debug_release=$3
        check_param_is_null $debug_release "debug or release is null"

        sh tools/run_mulit_learner_by_openmpirun.sh $debug_release >/dev/null 2>&1
        judge_succ_or_fail $? "learner cluster $debug_release start"
    else
        python3 framework/server/learner/learner.py --conf=/data/projects/1v1/conf/framework/learner.toml >$learner_start_file 2>&1
        judge_succ_or_fail $? "learner single start"
    fi

elif [ $server_type == "all" ];
then

    # all的场景是测试场景, 如果需要生成配置则把配置文件都生成出来
    if [ $produce_config == "config" ];
    then
        sgame_type=$3
        self_play=$4

        actor_ips=$5
        learner_ips=$6
        check_param_is_null "$actor_ips" "actor ips is null"
        check_param_is_null "$learner_ips" "learner ips is null"

        produce_config_by_process_name aisrv /data/projects/1v1/conf/framework/aisrv.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "aisrv produce config"

        produce_config_by_process_name actor /data/projects/1v1/conf/framework/actor.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "actor produce config"

        produce_config_by_process_name learner /data/projects/1v1/conf/framework/learner.toml $sgame_type $self_play $actor_ips $learner_ips
        judge_succ_or_fail $? "learner produce config"
    elif [ $produce_config == "no_config" ];
    then
        echo -e "\033[32m start all no need to produce config file \033[0m"

        # 单独修改self-play值
        main_configure_file=conf/framework/configure.toml
        # 如果明确需要设置再开始设置
        if [ -n "$self_play" ]
        then
            if [ "$self_play" == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi
        fi

    else
        echo -e "\033[31m produce_config only support config or no_config \033[0m"
        exit -1
    fi

    # 依赖的第三方组件modelpool是需要独立部署的, 如果是开发测试环境可以单独手动启动, 为了简化k8s启动流程, 会主动拉起来modelpool learner进程
    start_modelpool "learner"

    cd /data/projects/1v1

    # 下面是启动KaiwuDRL框架的组件
    source /etc/profile
    ldconfig

    make_aisrv_log_dir
    make_actor_log_dir
    make_learner_log_dir

    # 处理GPU异构的情况
    check_gpu_machine_type
    gpu_machine_type=$result

    echo -e "\033[32m gpu engine is $gpu_machine_type \033[0m"
    if [ -n "$gpu_machine_type" ];
    then
        sh tools/actor_cpp_copy.sh $gpu_machine_type
    fi

    export G6SHMNAME=KaiwuDRL && python3 framework/server/actor/actor.py --conf=/data/projects/1v1/conf/framework/actor.toml >$actor_start_file 2>&1 &
    judge_succ_or_fail $? "actor start"
    
    # 因为单个机器上部署多个进程, 默认就设置为learner的非集群模式
    python3 framework/server/learner/learner.py --conf=/data/projects/1v1/conf/framework/learner.toml >$learner_start_file 2>&1 &
    judge_succ_or_fail $? "learner single start"

    # 最后启动aisrv, 间隔10s
    sleep 10
    export G6SHMNAME=KaiwuDRL && python3 framework/server/aisrv/aisrv.py --conf=/data/projects/1v1/conf/framework/aisrv.toml >$aisrv_start_file 2>&1 &
    judge_succ_or_fail $? "aisrv start"

# 增加上sgame_client, 便于开发测试
elif [ $server_type == "client" ];
then
    sgame_type=$3
    self_play=$4

    make_client_log_dir

    if [ $sgame_type == "sgame_5v5" ];
    then
        cd /data/projects/1v1/app/sgame_5v5/tools
    elif [ $sgame_type == "sgame_1v1" ];
    then
        cd /data/projects/1v1/app/sgame_1v1/tools
    else
        echo -e "\033[31m sgame_type only support sgame_5v5 or sgame_1v1 \033[0m"
        exit -1
    fi

    # 启动sgame_client
    sh start_multi_game.sh 1 /data/projects/1v1/conf/framework/client.toml >$client_start_file 2>&1 &
    judge_succ_or_fail $? "sgame_client start"

    cd /data/projects/1v1/
    
else
    echo -e "\033[31m useage: sh tools/start.sh all|actor|aisrv config sgame_5v5 self_play actor_list learner_list [with_client|without_client], \
     \n or sh tools/start.sh learner cluster release config sgame_5v5 self_play actor_list learner_list \
     \n such as: sh tools/start.sh all config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh learner cluster release config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh aisrv no_config sgame_5v5 no_self_play \
     \n or sh tools/start.sh client no_config sgame_5v5 no_self_play \
     \n or sh tools/start.sh aisrv config sgame_5v5 self_play 127.0.0.1,127.0.0.2 127.0.0.1,127.0.0.2 with_client \
     \n or sh tools/start.sh learner cluster release no_config sgame_5v5 no_self_play \033[0m"

    exit -1
fi
