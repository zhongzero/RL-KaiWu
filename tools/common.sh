#!/bin/bash


# common.sh 主要放公共函数

# 根据命令行上次执行的结果进行处理
# 参数result, 上次执行命令的结果
# 参数message, 提示信息
judge_succ_or_fail()
{
    result=$1
    message=$2

    if [ $result -ne 0 ];
    then
        echo -e "\033[31m $message failed \033[0m"

        exit -1
    else
        echo -e "\033[32m $message success \033[0m"
    fi
}

# 判断进程是否存在, 不存在不做操作，存在则进行kill
# 参数process_name, 进程名
judge_process_exist_and_kill()
{
    process_name=$1
    process_num=`ps -ef | grep $process_name | grep -v grep | grep -v stop.sh | wc -l`
    if [ $process_num -gt 0 ];
    then
        ps -ef | grep $process_name | grep -v "grep" | awk '{print $2}' | xargs kill -9
    fi

    echo -e "\033[31m $process_name kill success \033[0m"
}

# 判断进程是否存在, 展示到屏幕
# 参数process_name, 进程名
judge_process_exist()
{
    process_name=$1
    process_num=`ps -ef | grep $process_name | grep -v grep | grep -v 'start.sh' | grep -v 'check.sh' | wc -l`

    echo -e "\033[32m $process_name "check success, process num is "$process_num \033[0m"
    ps -ef | grep $process_name | grep -v "grep" | grep -v 'start.sh' | grep -v 'check.sh'
}


# 获取本机的IP, 注意需要查看本机的网卡地址才行, 重试eth0和eth1的网卡
get_host_ip()
{
    ip=`ifconfig eth1 | grep 'inet ' | awk '{print $2}'`
    if [ -z $ip ]
    then
        ip=`ifconfig eth0 | grep 'inet ' | awk '{print $2}'`
    fi

    result=$ip
}


# 判断某个字符串是不是空串, 主要是做参数检测
# 参数param, 需要判断的参数名称
# 参数msg, 如果param为空时, 展示的报错信息
check_param_is_null()
{
    # 需要传入2个参数才行, 否则返回错误
    if [ $# != 2 ]; 
    then
        echo -e "\033[31m param is null or msg is null \033[0m"
        
        exit -1
    fi

    param=$1
    msg=$2

    if [ -z $param ]
    then
        echo -e "\033[31m $msg \033[0m"

        exit -1
    fi
}


# 返回GPU机器类型
check_gpu_machine_type()
{
    # 判断是否支持nvidia命令, 如果不支持会提前返回
    gpu_machine_type="CPU"
    result=`lspci |grep -i nvidia`
    if [ ! -n "$result" ];
    then
        result=$gpu_machine_type
        return
    fi

    # 获取具体的GPU卡号
    result=`nvidia-smi -L`
    if [[ "$(echo $result | grep A100)" != "" ]];
    then
        gpu_machine_type=A100
    elif [[ "$(echo $result | grep V100)" != "" ]];
    then
        gpu_machine_type=V100
    elif [[ "$(echo $result | grep T4)" != "" ]];
    then
        gpu_machine_type=T4
    else

        # 实在找不到就设置为CPU
        gpu_machine_type=CPU
    fi

    result=$gpu_machine_type
}

# 返回机器上CPU ID的集合, 从/proc/stat文件里获取
get_cpu_ids()
{
    cpu_result=()
    cpus=`cat /proc/stat | grep cpu | awk '{print $1}'`
    array=(${cpus//,/ })
    for var in ${array[@]}
    do
        cpu=`echo $var|tr -d 'a-z'`
        if [ -n "$cpu" ]
        then
            cpu_result[${#cpu_result[*]}]=$cpu
        fi
    done

    echo ${cpu_result[*]}
}

# 返回机器上的CPU ID集合, 适配lxcfs, 其从/sys/fs/cgroup/cpuset/cpuset.cpus文件里获取, 不是从/proc/stat文件获取
get_cpu_ids_by_lxcfs()
{
    cpuset_cpus=$(cat /sys/fs/cgroup/cpuset/cpuset.cpus)
    cpu_info=$(echo ${cpuset_cpus} | tr "," "\n")
    for cpu_core in ${cpu_info};
    do
        echo ${cpu_core} | grep "-" > /dev/null 2>&1
        if [ $? -eq 0 ];
        then
            first_cpu=$(echo ${cpu_core} | awk -F"-" '{print $1}')
            last_cpu=$(echo ${cpu_core} | awk -F"-" '{print $2}')
            cpu_modify=$(seq -s "," ${first_cpu} ${last_cpu})
            cpuset_cpus=$(echo ${cpuset_cpus} | sed "s/${first_cpu}-${last_cpu}/${cpu_modify}/g")
        fi
    done

    cpu_result=()
    array=(${cpuset_cpus//,/ })
    for var in ${array[@]}
    do
        cpu=`echo $var|tr -d 'a-z'`
        if [ -n "$cpu" ]
        then
            cpu_result[${#cpu_result[*]}]=$cpu
        fi
    done

    echo ${cpu_result[*]}
}

# 判断源文件是否存在, 如果存在则拷贝到目的端, 否则不做操作
file_exist_then_copy()
{
    # 需要传入2个参数才行, 否则返回错误
    if [ $# != 2 ]; 
    then
        echo -e "\033[31m  file_name is null or dest_dit is null \033[0m"
        exit -1
    fi

    file_name=$1
    dest_dit=$2

    if [ -f "$file_name" ]; 
    then
        cp -p $file_name $dest_dit

        echo -e "\033[32m $file_name copy to $dest_dit success \033[0m"
    fi

}
