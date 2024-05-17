#!/bin/bash


# learner_cpp_copy.sh 主要用于处理learner采用C++常驻进程时文件拷贝

chmod +x tools/common.sh
. tools/common.sh

# 拷贝learner的C++文件, 包括learner_server.so, learner_cpp_server
learner_cpp_dir=/data/projects/kaiwu-fwk/framework/server/cpp/src/learner/
learner_cpp_dist_dir=/data/projects/kaiwu-fwk/framework/server/cpp/dist/learner/

file_exist_then_copy ${learner_cpp_dir}/learner_server.cpython-37m-x86_64-linux-gnu.so ${learner_cpp_dist_dir};
file_exist_then_copy ${learner_cpp_dir}/learner_cpp_server ${learner_cpp_dist_dir};
judge_succ_or_fail $? "cp learner cpp file"

# 其他操作
