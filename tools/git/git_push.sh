#!/bin/bash


# git的push操作

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/git/git_push.sh [branch], such as: sh tools/git/git_push.sh master \033[0m"
    exit -1
fi

branch=$1

git push origin $branch 
judge_succ_or_fail $? "git push $branch"