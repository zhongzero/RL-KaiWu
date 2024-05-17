#!/bin/bash
# 查看learner机器上的引擎文件生产步数

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/check_learner_step.sh sgame_1v1|sgame_5v5, \n such as: sh tools/check_learner_step.sh sgame_5v5 \n or sh tools/check_learner_step.sh sgame_1v1 \033[0m"
    
    exit -1
fi

app_name=$1
checkpoint_name=/data/ckpt/sgame_1v1_ppo/checkpoint
if [ $app_name == "sgame_5v5" ];
then
    checkpoint_name=/data/ckpt/sgame_5v5_ppo/checkpoint
fi

if [ ! -f $checkpoint_name ];
then
    echo -e "\033[31m $checkpoint_name file not exist, please check \033[0m"

    exit -2
fi

# 格式如下: 
# model_checkpoint_path: "/data/ckpt//sgame_1v1_ppo/model.ckpt-663"
# all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-594"
# all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-618"
# all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-645"
# all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-663"

step_str=`tail -n 1 $checkpoint_name`
a=`echo ${step_str##*-}`
b=`echo ${a%\"*}`

echo $b