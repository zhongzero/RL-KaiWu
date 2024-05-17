#!/bin/bash

# eval模式下, 修改eval_model_dir路径

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/eval_model_dir_change.sh eval_model_dir such as: sh tools/eval_model_dir_change.sh /data/projects/kaiwu-fwk/ckpt/model.ckpt-0 \033[0m"
    exit -1
fi

configure_ini=/data/projects/kaiwu-fwk/conf/framework/configure.toml
eval_model_dir=$1

sed -i '/eval_model_dir/d' $configure_ini

echo -e "\neval_model_dir = $eval_model_dir" >> $configure_ini

judge_succ_or_fail $? "$configure_system_file change eval_model_dir $eval_model_dir"