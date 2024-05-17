#!/bin/bash


# 移除容器里安装的Protobuf

chmod +x tools/common.sh
. tools/common.sh


rm /usr/local/bin/protoc
rm /usr/bin/protoc
rm -rf /usr/local/protobuf/
rm -rf /usr/local/lib/libproto*
rm -rf /usr/lib/protoc
rm -rf /usr/local/lib/protoc

judge_succ_or_fail $? "remove protobuf"