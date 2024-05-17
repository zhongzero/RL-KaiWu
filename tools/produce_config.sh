#!/bin/bash


# produce_config.sh 主要生成配置文件内容
# 目的是减少人工操作
# 如果job_master上线后, 则不需要采用该脚本来生成配置, 而是依赖job_master的生成配置和部署

chmod +x tools/common.sh
. tools/common.sh

main_configure_file=conf/framework/configure.toml

# 根据进程名来生成不同的配置
produce_config_by_process_name()
{
    process_name=$1
    configure_file=$2
    sgame_type=$3
    self_play=$4
    
    # 不同的进程对这些参数使用方法不同, 为了编程和运营方便, 这里参数形式和个数不按照进程名来区别
    # aisrv, 需要actor和learner列表, 需要区分self_play模式
    # actor, 需要learner列表
    # learner, 需要learner列表
    # actor_ip_list_str和learner_ip_list_str支持IP:端口形式的; 同时也支持IP形式的, 此时端口即采用默认端口
    actor_ip_list_str=$5
    learner_ip_list_str=$6

    # 不同的进程名, 生成不同的配置文件
    if [ $process_name == "aisrv" ];
    then

        if [ $sgame_type == "sgame_1v1" ];
        then
            # 1V1的self_play按照需要设置
            if [ $self_play == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi

            # 删除原来的actor_addrs, actor_proxy_num, self_play_actor_proxy_num, self_play_old_actor_proxy_num
            # learner_addrs, learner_proxy_num, self_play_learner_proxy_num, self_play_old_learner_proxy_num内容
            sed -i '/actor_addrs/d' $configure_file
            sed -i '/actor_proxy_num/d' $configure_file
            sed -i '/self_play_actor_proxy_num/d' $configure_file
            sed -i '/self_play_old_actor_proxy_num/d' $configure_file
            sed -i '/learner_addrs/d' $configure_file
            sed -i '/learner_proxy_num/d' $configure_file
            sed -i '/self_play_learner_proxy_num/d' $configure_file
            sed -i '/self_play_old_learner_proxy_num/d' $configure_file

            # 新增actor_addrs, actor_proxy_num, learner_addrs, learner_proxy_num内容
            actor_addrs_list=""

            # actor地址, 按照逗号,分割
            actor_ip_list=(`echo $actor_ip_list_str | tr ',' ' '`)
            actor_ip_list_num=${#actor_ip_list[*]}

            # learner地址, 按照逗号,分割
            learner_addrs_list=""
            learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
            learner_ip_list_num=${#learner_ip_list[*]}

            if [ $self_play == "self_play" ];
            then
                new_array_list_num=$actor_ip_list_num/2

                # 当小于等于1时需要特别处理
                if [ $actor_ip_list_num -eq 1 ];
                then
                    policy_one_actor_list=$actor_ip_list
                    policy_two_actor_list=$actor_ip_list
                else
                    policy_one_actor_list=${actor_ip_list[@]:0:new_array_list_num}
                    policy_two_actor_list=${actor_ip_list[@]:new_array_list_num:actor_ip_list_num}
                fi

                # 处理policy one和policy two的列表情况
                count=0
                for ip_port in ${policy_one_actor_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs_one="train_one = [$actor_addrs_list]"
                self_play_actor_proxy_num=$count

                count=0
                actor_addrs_list=""
                for ip_port in ${policy_two_actor_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs_two="train_two = [$actor_addrs_list]"
                self_play_old_actor_proxy_num=$count

                actor_addrs="{ "$actor_addrs_one","$actor_addrs_two" }"

                # 针对输入的IP列表, 直接切分为2段进行self_play
                new_array_list_num=$learner_ip_list_num/2

                # 当小于等于1时需要特别处理
                if [ $learner_ip_list_num -eq 1 ];
                then
                    policy_one_learner_list=$learner_ip_list
                    policy_two_learner_list=$learner_ip_list
                else
                    policy_one_learner_list=${learner_ip_list[@]:0:new_array_list_num}
                    policy_two_learner_list=${learner_ip_list[@]:new_array_list_num:learner_ip_list_num}
                fi

                count=0
                for ip_port in ${policy_one_learner_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs_one="train_one = [$learner_addrs_list]"
                self_play_learner_proxy_num=$count

                count=0
                learner_addrs_list=""
                for ip_port in ${policy_two_learner_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs_two="train_two = [$learner_addrs_list]"
                self_play_old_learner_proxy_num=$count

                learner_addrs="{ "$learner_addrs_one","$learner_addrs_two" }"

                # 不需要的都赋值默认值1
                learner_proxy_num=1
                actor_proxy_num=1
            else
                count=0
                for ip_port in ${actor_ip_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs="{train_one = [$actor_addrs_list]}"
                actor_proxy_num=$count

                count=0
                for ip_port in ${learner_ip_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs="{train_one = [$learner_addrs_list]}"
                learner_proxy_num=$count

                # 下面self_play_actor_proxy_num,self_play_old_actor_proxy_num,self_play_learner_proxy_num,self_play_old_learner_proxy_num赋值为1, 在1V1的场景下不使用
                self_play_actor_proxy_num=1
                self_play_old_actor_proxy_num=1
                self_play_learner_proxy_num=1
                self_play_old_learner_proxy_num=1
            fi

            # 换行, 规避和原来的内容覆盖
            echo -e "\nactor_addrs = $actor_addrs" >> $configure_file
            echo "actor_proxy_num = $actor_proxy_num" >> $configure_file
            echo "self_play_actor_proxy_num = $self_play_actor_proxy_num" >> $configure_file
            echo "self_play_old_actor_proxy_num = $self_play_old_actor_proxy_num" >> $configure_file
            echo "learner_addrs = $learner_addrs" >> $configure_file
            echo "learner_proxy_num = $learner_proxy_num" >> $configure_file
            echo "self_play_learner_proxy_num = $self_play_learner_proxy_num" >> $configure_file
            echo "self_play_old_learner_proxy_num = $self_play_old_learner_proxy_num" >> $configure_file

        elif [ $sgame_type == "sgame_5v5" ];
        then
            # 5V5的self_play按照需要设置
            if [ $self_play == "self_play" ];
            then
                sed -i "s/self_play = False/self_play = True/g" $main_configure_file
            else
                sed -i "s/self_play = True/self_play = False/g" $main_configure_file
            fi

            # 删除原来的actor_addrs, actor_proxy_num, self_play_actor_proxy_num, self_play_old_actor_proxy_num
            # learner_addrs, learner_proxy_num, self_play_learner_proxy_num, self_play_old_learner_proxy_num内容
            sed -i '/actor_addrs/d' $configure_file
            sed -i '/actor_proxy_num/d' $configure_file
            sed -i '/self_play_actor_proxy_num/d' $configure_file
            sed -i '/self_play_old_actor_proxy_num/d' $configure_file
            sed -i '/learner_addrs/d' $configure_file
            sed -i '/learner_proxy_num/d' $configure_file
            sed -i '/self_play_learner_proxy_num/d' $configure_file
            sed -i '/self_play_old_learner_proxy_num/d' $configure_file

            # 新增原来的actor_addrs, actor_proxy_num, self_play_actor_proxy_num, self_play_old_actor_proxy_num
            # learner_addrs, learner_proxy_num, self_play_learner_proxy_num, self_play_old_learner_proxy_num内容
            actor_addrs_list=""

            # actor地址, 按照逗号,分割
            actor_ip_list=(`echo $actor_ip_list_str | tr ',' ' '`)
            actor_ip_list_num=${#actor_ip_list[*]}

            # learner地址, 按照逗号,分割
            learner_addrs_list=""
            learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
            learner_ip_list_num=${#learner_ip_list[*]}

            # 针对输入的IP列表, 采用下面处理策略:
            # 如果是self_play, 分为2个数组
            # 如果是非self_play, 分为1个数组
            if [ $self_play == "self_play" ];
            then
                new_array_list_num=$actor_ip_list_num/2
                # 当小于等于1时需要特别处理
                if [ $actor_ip_list_num -eq 1 ];
                then
                    policy_one_actor_list=$actor_ip_list
                    policy_two_actor_list=$actor_ip_list
                else
                    policy_one_actor_list=${actor_ip_list[@]:0:new_array_list_num}
                    policy_two_actor_list=${actor_ip_list[@]:new_array_list_num:actor_ip_list_num}
                fi

                # 处理policy one和policy two的列表情况
                count=0
                for ip_port in ${policy_one_actor_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs_one="train_one = [$actor_addrs_list]"
                self_play_actor_proxy_num=$count

                count=0
                actor_addrs_list=""
                for ip_port in ${policy_two_actor_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs_two="train_two = [$actor_addrs_list]"
                self_play_old_actor_proxy_num=$count

                actor_addrs="{ "$actor_addrs_one","$actor_addrs_two" }"

                # 针对输入的IP列表, 直接切分为2段进行self_play
                new_array_list_num=$learner_ip_list_num/2
                # 当小于等于1时需要特别处理
                if [ $learner_ip_list_num -eq 1 ];
                then
                    policy_one_learner_list=$learner_ip_list
                    policy_two_learner_list=$learner_ip_list
                else
                    policy_one_learner_list=${learner_ip_list[@]:0:new_array_list_num}
                    policy_two_learner_list=${learner_ip_list[@]:new_array_list_num:learner_ip_list_num}
                fi

                count=0
                for ip_port in ${policy_one_learner_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs_one="train_one = [$learner_addrs_list]"
                self_play_learner_proxy_num=$count

                count=0
                learner_addrs_list=""
                for ip_port in ${policy_two_learner_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs_two="train_two = [$learner_addrs_list]"
                self_play_old_learner_proxy_num=$count

                learner_addrs="{ "$learner_addrs_one","$learner_addrs_two" }"

                # 不需要的都赋值默认值1
                learner_proxy_num=1
                actor_proxy_num=1

            else
                count=0
                for ip_port in ${actor_ip_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=8888
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        actor_addrs_list="\"$ip:$port\""
                    else
                        actor_addrs_list="$actor_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                actor_addrs="{train_one = [$actor_addrs_list]}"
                actor_proxy_num=$count

                count=0
                for ip_port in ${learner_ip_list[@]}
                do
                    # 字符串切割为IP和port的形式
                    arr=(`echo $ip_port | tr ':' ' '`)
                    len=${#arr[@]}
                    if [ $len -eq 2 ];
                    then
                        ip=${arr[0]}
                        port=${arr[1]}
                    elif [ $len -eq 1 ];
                    then
                        ip=${arr[0]}
                        port=9999
                    else
                        echo "$ip_port is not ip:port or ip format"
                        exit -1
                    fi

                    if [ $count -eq 0 ];
                    then
                        learner_addrs_list="\"$ip:$port\""
                    else
                        learner_addrs_list="$learner_addrs_list, "\"$ip:$port\"""
                    fi

                    let count++
                done

                learner_addrs="{train_one = [$learner_addrs_list]}"
                learner_proxy_num=$count

                # 不需要的都赋值默认值1
                self_play_actor_proxy_num=1
                self_play_old_actor_proxy_num=1
                self_play_learner_proxy_num=1
                self_play_old_learner_proxy_num=1

            fi

            # 换行, 规避和原来的内容覆盖, 无论哪种情况都需要设置
            echo -e "\nactor_addrs = $actor_addrs" >> $configure_file
            echo "actor_proxy_num = $actor_proxy_num" >> $configure_file
            echo "learner_addrs = $learner_addrs" >> $configure_file
            echo "learner_proxy_num = $learner_proxy_num" >> $configure_file
            echo "self_play_actor_proxy_num = $self_play_actor_proxy_num" >> $configure_file
            echo "self_play_old_actor_proxy_num = $self_play_old_actor_proxy_num" >> $configure_file
            echo "self_play_learner_proxy_num = $self_play_learner_proxy_num" >> $configure_file
            echo "self_play_old_learner_proxy_num = $self_play_old_learner_proxy_num" >> $configure_file

        else
            echo -e "\033[31m produce_config only support sgame_5v5 or sgame_1v1 \033[0m" 
            exit -1
        fi

    elif [ $process_name == "actor" ];
    then
        # 删除modelpool_remote_addrs
        sed -i '/modelpool_remote_addrs/d' $configure_file
        get_host_ip
        host_ip=$result

        # 新增modelpool_remote_addrs内容

        # 换行, 规避和原来的内容覆盖
        echo -e "\nmodelpool_remote_addrs = \"$host_ip:10014\"" >> $configure_file

        # 配置/data/projects/1v1/thirdparty/model_pool_go/config下面的gpu.iplist
        learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
        gpu_configure_file=/data/projects/1v1/thirdparty/model_pool_go/config/gpu.iplist
        cat /dev/null > $gpu_configure_file

        for ip_port in ${learner_ip_list[@]}
        do
            # 字符串切割为IP和port的形式
            arr=(`echo $ip_port | tr ':' ' '`)
            len=${#arr[@]}
            if [ $len -eq 2 ];
            then
                ip=${arr[0]}
                port=${arr[1]}
            elif [ $len -eq 1 ];
            then
                ip=${arr[0]}
                port=9999
            else
                echo "$ip_port is not ip:port or ip format"
                exit -1
            fi

            echo $ip >> $gpu_configure_file
        done

    elif [ $process_name == "learner" ];
    then
        # 删除modelpool_remote_addrs, ip_address
        sed -i '/modelpool_remote_addrs/d' $configure_file
        sed -i '/ip_address/d' $configure_file

        # 新增modelpool_remote_addrs, ip_address内容
        get_host_ip
        host_ip=$result

        # 换行, 规避和原来的内容覆盖
        echo -e "\nmodelpool_remote_addrs = \"$host_ip:10014\"" >> $configure_file
        echo -e "ip_address = \"$host_ip\"" >> $configure_file

        # 配置/data/projects/1v1/tools下的run_mulit_learner_by_openmpirun.sh里的Nodelist和Num_process
        openmpi_congiure_file=/data/projects/1v1/tools/run_mulit_learner_by_openmpirun.sh
        
        learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
        count=0
        for ip_port in ${learner_ip_list[@]}
        do
            # 字符串切割为IP和port的形式
            arr=(`echo $ip_port | tr ':' ' '`)
            len=${#arr[@]}
            if [ $len -eq 2 ];
            then
                ip=${arr[0]}
                port=${arr[1]}
            elif [ $len -eq 1 ];
            then
                ip=${arr[0]}
                port=9999
            else
                echo "$ip_port is not ip:port or ip format"
                exit -1
            fi

            if [ $count -eq 0 ];
            then
                learner_addrs_openmpi_list="\"$ip:1\""
            else
                learner_addrs_openmpi_list="$learner_addrs_openmpi_list,"\"$ip:1\"""
            fi

            let count++
        done

        # 修改以某个字符串开头的某行的配置项
        sed -i '/^Nodelist=/c'Nodelist=$learner_addrs_openmpi_list'' $openmpi_congiure_file
        sed -i '/^Num_process=/c'Num_process=$count'' $openmpi_congiure_file
    fi
}
