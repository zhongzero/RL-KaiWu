#!/bin/bash


# git的pull操作

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/git/git_pull.sh [branch], such as: sh tools/git/git_pull.sh master \033[0m"
    exit -1
fi

branch=$1

git lfs pull origin $branch && git pull origin $branch
judge_succ_or_fail $? "git pull $branch"