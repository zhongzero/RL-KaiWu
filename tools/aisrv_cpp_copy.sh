#!/bin/bash


# aisrv_cpp_copy.sh 主要用于处理aisrv采用C++常驻进程时文件拷贝

chmod +x tools/common.sh
. tools/common.sh

# 拷贝aisrv的C++文件, 包括aisrv_server.so, aisrv_cpp_server, 从build文件夹里赋值到dist文件夹下
aisrv_cpp_dir=/data/projects/kaiwu-fwk/framework/server/cpp/src/aisrv/build
aisrv_cpp_dist_dir=/data/projects/kaiwu-fwk/framework/server/cpp/dist/aisrv/

file_exist_then_copy ${aisrv_cpp_dir}/aisrv_server.cpython-37m-x86_64-linux-gnu.so ${aisrv_cpp_dist_dir};
file_exist_then_copy ${aisrv_cpp_dir}/aisrv_cpp_server ${aisrv_cpp_dist_dir};
judge_succ_or_fail $? "cp aisrv cpp file"

# 其他操作
