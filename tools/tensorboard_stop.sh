#!/bin/bash


# tensorboard 程停止脚本

chmod +x tools/common.sh
. tools/common.sh

service_name='tensorboard'

judge_process_exist_and_kill $service_name
judge_succ_or_fail $? "$service_name stop"