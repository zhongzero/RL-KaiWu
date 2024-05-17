#!/bin/bash
# 将tensorflow的模型转换为trt模型, 包括PB文件和checkpoint文件
# onnx的git地址: https://github.com/onnx/tensorflow-onnx


chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 5 ];
then
    echo -e "\033[31m useage: sh tools/change_tensorflow_model_to_trt.sh graphdef|checkpoint input_file inputs outputs output_file such as: \
    sh tools/change_tensorflow_model_to_trt.sh graphdef input_model.pb Placeholder_1:0,Placeholder_2:0 output0:0,output1:0 output_model.onnx \
    \n or sh tools/change_tensorflow_model_to_trt.sh checkpoint input_model.checkpoint Placeholder_1:0,Placeholder_2:0 output0:0,output1:0 output_model.onnx \
    \n
    5V5的inputs是feature_p1:0,feature_p2:0,feature_p3:0,feature_p4:0,feature_p5:0 output是output_logits:0,output_value:0,output_meta_msg:0,lstm_cell_output:0,lstm_hidden_output:0
    1V1的inputs是feature:0,legal_action:0,lstm_cell:0,lstm_hidden:0 outputs是
    \033[0m"

    exit -1
fi

# pb或者checkpoint
graphdef_or_checkpoint=$1
# pb模型文件或者checkpoint文件
input_file=$2
# 模型的输入参数, 注意是多输入, 请确认清楚
inputs=$3
# 模型的输出参数, 注意是多输入, 请确认清楚
outputs=$4
# onnx输出文件
output_file=$5


# 5V5的inputs是feature_p1:0,feature_p2:0,feature_p3:0,feature_p4:0,feature_p5:0 output是output_logits:0,output_value:0,output_meta_msg:0,lstm_cell_output:0,lstm_hidden_output:0
# 1V1的inputs是feature:0,legal_action:0,lstm_cell:0,lstm_hidden:0 outputs是output_value:0


# 需要记录转换文件耗时
start=$(date +%s)

if [ $graphdef_or_checkpoint == "graphdef" ];
then
    echo -e "\033[32m python3  -m tf2onnx.convert --graphdef ${input_file} --output ${output_file} --inputs ${inputs} --outputs ${outputs} --opset 12 \033[0m"

    python3 -m tf2onnx.convert \
        --graphdef ${input_file} \
        --output ${output_file} \
        --inputs ${inputs} \
        --outputs ${outputs} \
        --opset 12

    judge_succ_or_fail $? "tf2onnx.convert pb to onnx"

elif [ $graphdef_or_checkpoint == "checkpoint" ];
then
    echo -e "\033[32m python3  -m tf2onnx.convert --checkpoint ${input_file} --output ${output_file} --inputs ${inputs} --outputs ${outputs} \033[0m"

    python3  -m tf2onnx.convert \
    --checkpoint ${input_file} \
    --output ${output_file} \
    --inputs ${inputs} \ 
    --outputs ${outputs}

    judge_succ_or_fail $? "tf2onnx.convert checkpoint to onnx"

else
    echo -e "\033[31m useage: sh tools/change_tensorflow_model_to_trt.sh graphdef|checkpoint input_file inputs outputs output_file such as: \
    sh tools/change_tensorflow_model_to_trt.sh graphdef input_model.pb Placeholder_1:0,Placeholder_2:0 output0:0,output1:0 output_model.onnx \
    \n or sh tools/change_tensorflow_model_to_trt.sh checkpoint input_model.checkpoint Placeholder_1:0,Placeholder_2:0 output0:0,output1:0 output_model.onnx \
    \n
    5V5的inputs是feature_p1:0,feature_p2:0,feature_p3:0,feature_p4:0,feature_p5:0 output是output_logits:0,output_value:0,output_meta_msg:0,lstm_cell_output:0,lstm_hidden_output:0
    1V1的inputs是feature:0,legal_action:0,lstm_cell:0,lstm_hidden:0 outputs是
    \033[0m"
    
    exit -1
fi

end=$(date +%s)
take=$(( end - start ))
echo -e "\033[32m Time taken to execute commands is ${take} seconds \033[0m"
