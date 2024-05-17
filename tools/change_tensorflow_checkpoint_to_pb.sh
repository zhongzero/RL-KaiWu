#!/bin/bash
# 将tensorflow的checkpoint文件转换为pb文件



chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 4 ];
then
    echo -e "\033[31m useage: sh tools/change_tensorflow_checkpoint_to_pb.sh graphdef output_node_names checkpoint_dir output \
    such as: sh tools/change_tensorflow_checkpoint_to_pb.sh input_model.proto output0:0,output1:0 /data/ckpt/ model.pb  \033[0m"
    
    exit -1
fi

# tensorflow模型文件
graphdef=$1
# output_node_names, 注意需要确认清楚
output_node_names=$2
# outputs
checkpoint_dir=$3
# pb输出文件
output=$4

# 需要记录转换文件耗时
start=$(date +%s)

# 将checkpoint文件转换为PB文件
echo -e "\033[32m python3  -m ensorflow.python.tools.freeze_graph --input_graph ${graphdef}  \ 
        --input_binary=true --output_node_names=${output_node_names} --input_checkpoint=${checkpoint_dir} --output_graph=${output} \033[0m"

python3 -m tensorflow.python.tools.freeze_graph \
    --input_graph=${graphdef} \
    --input_binary=true \
    --output_node_names=${output_node_names} \
    --input_checkpoint=${checkpoint_dir} \
    --output_graph=${output}

judge_succ_or_fail $? "checkpoint to pb"

end=$(date +%s)
take=$(( end - start ))
echo -e "\033[32m Time taken to execute commands is ${take} seconds \033[0m"