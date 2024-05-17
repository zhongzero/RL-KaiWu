#!/bin/bash

# 清理机器上learner进程的tensorflow的checkpoint文件的功能

chmod +x tools/common.sh
. tools/common.sh

# 直接删除对应的文件夹
ckpt_dir_str=`grep '^restore_dir' conf/framework/configure.toml | cut -d '=' -f2 | tr -d ' '`
ckpt_dir=$(echo "$ckpt_dir_str" | tr -d '"')

rm -rf $ckpt_dir/*
judge_succ_or_fail $? "$ckpt_dir clear"

rm -rf /data/summary/*
judge_succ_or_fail $? "/data/summary/ clear"

rm -rf /data/pb_model/*
judge_succ_or_fail $? "/data/pb_model/ clear"