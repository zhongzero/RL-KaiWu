#!/bin/bash


# 火焰图采集C++性能数据, 生成的火焰图需要采用浏览器打开, 采用contrl + c即可关闭进程, 查看out.svg文件, 注意确保perf抓取时间足够
# 1. perf record
# 2. 对perf.data进行解析
# 3. 将out.perf中的符号进行折叠
# 4. 生成svg图

chmod +x tools/common.sh
. tools/common.sh

# 参数如下:
# pid, 进程ID
if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/flamegraph_start.sh pid, such as sh tools/flamegraph_start.sh 1 \033[0m"

    exit -1
fi

pid=$1

# 下面是生成火焰图步骤
tools/perf record -p $pid -e cpu-clock -a -g
tools/perf script -i perf.data &> out.perf
tools/FlameGraph/stackcollapse-perf.pl out.perf &> out.folded
tools/FlameGraph/flamegraph.pl out.folded > out.svg