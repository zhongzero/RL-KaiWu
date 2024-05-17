#!/bin/bash

# 删除机器上废弃的docker镜像

chmod +x tools/common.sh
. tools/common.sh

docker system prune -af