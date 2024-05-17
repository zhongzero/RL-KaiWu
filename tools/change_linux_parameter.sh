#!/bin/bash
# 修改linux的内核参数, 便于提升数据传输效率


# TCP相关参数修改
change_tcp_parameter()
{
    tcp_config="/proc/sys/net/ipv4"

    # TCP窗口扩大因子
    if [ -f "$tcp_config/tcp_window_scaling" ] && [ -w "$tcp_config/tcp_window_scaling" ]; 
    then
        echo "1" > $tcp_config/tcp_window_scaling
    fi

    # 调整TCP发送缓冲区大小
    if [ -f "$tcp_config/tcp_wmem" ] && [ -w "$tcp_config/tcp_wmem" ]; 
    then
        echo "4096    16384   4194304" > $tcp_config/tcp_wmem
    fi

    # 调整TCP接收缓冲区大小
    if [ -f "$tcp_config/tcp_rmem" ] && [ -w "$tcp_config/tcp_rmem" ]; 
    then
        echo "4096    87380   6291456" > $tcp_config/tcp_rmem
    fi

    # 启动接收缓冲区的调节功能, 发送缓冲区调节功能是自动打开的
    if [ -f "$tcp_config/tcp_moderate_rcvbuf" ] && [ -w "$tcp_config/tcp_moderate_rcvbuf" ]; 
    then
        echo "1" > $tcp_config/tcp_moderate_rcvbuf
    fi

    # 调整TCP内存范围
    if [ -f "$tcp_config/tcp_rmem" ] && [ -w "$tcp_config/tcp_rmem" ]; 
    then
        echo "182514  243352  365028" > $tcp_config/tcp_rmem
    fi
    if [ -f "$tcp_config/tcp_wmem" ] && [ -w "$tcp_config/tcp_wmem" ]; 
    then
        echo "182514  243352  365028" > $tcp_config/tcp_wmem
    fi

    # 调整somaxconn值
    #current_value=$(sysctl -n net.core.somaxconn 2>/dev/null)
    #if [[ $current_value != 10240 ]]; 
    #then
    #    sudo sysctl -w net.core.somaxconn=10240 2>/dev/null
    #fi

}
