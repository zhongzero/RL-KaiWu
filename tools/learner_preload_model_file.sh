#!/bin/bash
# learner上预先加载训练好的model文件
# 参数: 需要加载的model文件, 注意如果是tensorflow, 则是文件目录; 如果是pytorch, 是具体的文件
# 主要步骤:
# 1. 获取learner引擎文件目录
# 2. 将需要替换的引擎文件放置到1中
# 3. 修改checkpoint里的内容
# 4. learner进程启动, 不需要在本脚本里处理


chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/learner_preload_model_file.sh tensorflow|pytorch dir|file, \n such as: sh tools/learner_preload_model_file.sh tensorflow /data/ckpt/gorge_walk_dqn/ \n or sh tools/learner_preload_model_file.sh pytorch /data/ckpt/gorge_walk_dqn/model.ckpt-0.pkl \033[0m"
    
    exit -1
fi

frame_work=$1
preload_dir_or_file=$2

# 获取learner引擎文件目录, 形如/data/ckpt/sgame_5v5_ppo/
ckpt_dir=`grep '^restore_dir' conf/framework/configure.toml | cut -d '=' -f2 | tr -d ' '`
app=`grep '^app' conf/framework/configure.toml | cut -d '=' -f2 | tr -d ' '`
algo=`grep '^algo' conf/framework/learner.toml | cut -d '=' -f2 | tr -d ' '`
dist_dir=${ckpt_dir}/${app}_${algo}

# 将需要替换的引擎文件放置到1中
if [ $frame_work == "tensorflow" ];
then
    if [ ! -d "$preload_dir_or_file" ]; 
    then
        echo -e "\033[31m useage: sh tools/learner_preload_model_file.sh tensorflow|pytorch dir|file, \n such as: sh tools/learner_preload_model_file.sh tensorflow /data/ckpt/gorge_walk_dqn/ \n or sh tools/learner_preload_model_file.sh pytorch /data/ckpt/gorge_walk_dqn/model.ckpt-0.pkl \033[0m"
        
        exit -1
    fi

    cp -r $preload_dir_or_file/* $dist_dir

elif [ $frame_work == "pytorch" ];
then
    if [ ! -f "$preload_dir_or_file" ]; 
    then
        echo -e "\033[31m useage: sh tools/learner_preload_model_file.sh tensorflow|pytorch dir|file, \n such as: sh tools/learner_preload_model_file.sh tensorflow /data/ckpt/gorge_walk_dqn/ \n or sh tools/learner_preload_model_file.sh pytorch /data/ckpt/gorge_walk_dqn/model.ckpt-0.pkl \033[0m"
        
        exit -1
    fi

    cp -r $preload_dir_or_file $dist_dir
else
    echo -e "\033[31m useage: sh tools/learner_preload_model_file.sh tensorflow|pytorch dir|file, \n such as: sh tools/learner_preload_model_file.sh tensorflow /data/ckpt/gorge_walk_dqn/ \n or sh tools/learner_preload_model_file.sh pytorch /data/ckpt/gorge_walk_dqn/model.ckpt-0.pkl \033[0m"
    
    exit -1
fi

# 修改checkpoint里的内容
checkpoint_file=$dist_dir/checkpoint
echo " " > $checkpoint_file

echo "checkpoints list" > $checkpoint_file
checkpoint_id=0
if [ $frame_work == "tensorflow" ];
then
    echo "please see "
elif [ $frame_work == "pytorch" ];
then
    last_digit=$(basename "$preload_dir_or_file" | cut -d'-' -f2 | cut -d'.' -f1)
fi
echo "all_model_checkpoint_paths: "/data/ckpt//gorge_walk_dqn//model.ckpt-$last_digit"" >> $checkpoint_file

judge_succ_or_fail $? "learner preload model file"