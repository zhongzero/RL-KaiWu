#!/bin/bash


# git的branch操作

chmod +x tools/common.sh
. tools/common.sh


git branch

judge_succ_or_fail $? "git branch"