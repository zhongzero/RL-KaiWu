#!/bin/bash


# 针对git lfs error时报错的处理

chmod +x tools/common.sh
. tools/common.sh

git rm .gitattributes
git reset --hard HEAD

judge_succ_or_fail $? "git lfs error"