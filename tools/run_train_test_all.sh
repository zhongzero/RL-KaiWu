#!/bin/bash

# 将所有的在/data/projects下业务执行下python3 train_test.py

chmod +x tools/common.sh
. tools/common.sh


array=("/data/projects/1v1" "/data/projects/3v3" "/data/projects/5v5" "/data/projects/gorge_walk_v1" "/data/projects/gorge_walk_v2" "/data/projects/offline" \ 
       "/data/projects/traffic"
)

for element in ${array[@]}
do
    cd $element

    # 判断是否有ERROR日志出现, 出现即报错
    python3 train_test.py |& grep -q "ERROR"
    if [ $? -eq 0 ]; 
    then
        echo "Error occurred. Stopping script execution."
        exit 1
    fi
    judge_succ_or_fail $? "$element python3 train_test.py"
    
done