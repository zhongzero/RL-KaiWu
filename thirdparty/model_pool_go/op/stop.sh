#!/bin/bash

# 停止modelpool进程

rm -rf ../bin/files
rm -rf ../bin/model

process_num=`ps -ef | grep "modelpool" | grep -v grep | wc -l`
if [ $process_num -gt 0 ];
then
   ps -ef | grep "modelpool" | grep -v "grep" | awk '{print $2}' | xargs kill -9
fi
