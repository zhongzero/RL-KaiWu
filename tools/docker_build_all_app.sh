#!/bin/bash

# 将所有的在/data/projects下业务打镜像的脚本, 新增业务新增下面的array即可

chmod +x tools/common.sh
. tools/common.sh


array=("/data/projects/1v1" "/data/projects/3v3" "/data/projects/5v5" "/data/projects/gorge_walk_v1" "/data/projects/gorge_walk_v2" "/data/projects/offline" \ 
       "/data/projects/traffic"
)

for element in ${array[@]}
do
    cd $element
    sh docker/build_all.sh
    judge_succ_or_fail $? "$element build_all.sh"
    
done