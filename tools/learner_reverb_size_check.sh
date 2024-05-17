#!/bin/bash
# 查看learner机器上的reverb的样本池子大小输出实例，只是需要抓住下面关键字
# current_size, 当前样本池子大小, 达到样本池设置的大小后不会再增长
# insert_stats, 当前样本池插入样本数目大小, 一直增长
# sample_stats, 当前样本池消耗样本数目大小, 一直增长


chmod +x tools/common.sh
. tools/common.sh

python3 -c "import reverb; server_info=reverb.client.Client('localhost:9999').server_info(); \
            print(f'current_size: {server_info[\"reverb_replay_buffer_table_0\"].current_size}'); \
            print(f'insert_stats: {server_info[\"reverb_replay_buffer_table_0\"].rate_limiter_info.insert_stats.completed}'); \
            print(f'sample_stats: {server_info[\"reverb_replay_buffer_table_0\"].rate_limiter_info.sample_stats.completed}')"