#!/bin/bash

# modelpool进程停止脚本

chmod +x tools/common.sh
. tools/common.sh

# 依赖的第三方组件modelpool是需要独立部署的, 如果是开发测试环境可以单独手动启动
cd /data/projects/1v1
sh thirdparty/model_pool_go/op/stop.sh
judge_succ_or_fail $? "modelpool stop"