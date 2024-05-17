#!/bin/bash

# 进程安装脚本, 在开发测试阶段使用较多, 线上部署采用容器方式后不需要使用

if [ $# -ne 1 ] && [ $# -ne 2 ] && [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/setup_whl.sh debug|release [set_name] [set_self_play_name]\033[0m"
    exit -1
fi

debug_release=$1
# 如果第二个参数不空则为set_name, 如果是self_play模式下, 则为self_play_set_name
set_name=$2
# 如果第三个参数不为空则为self_play_set_old_name
self_play_set_old_name=$3

if [[ $debug_release != 'debug' ]] && [[ $debug_release != 'release' ]];
then
    echo -e "\033[31m useage: sh tools/setup_whl.sh debug|release [set_name] [set_self_play_name]\033[0m"
    exit -1
fi

# 处理下pybind11找不到连接的情况
rm -rf framework/common/pybind11/zmq_ops/pybind11
ln -s /usr/local/python-3.7/lib/python3.7/site-packages/pybind11/include/ framework/common/pybind11/zmq_ops/pybind11

pip uninstall -y dist/*.whl
sh build_wheel.sh $debug_release $set_name $self_play_set_old_name
pip install dist/*.whl

echo -e "\033[32m tools/setup_whl.sh $debug_release $set_name $self_play_set_old_name success \033[0m"