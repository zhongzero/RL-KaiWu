#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project : 1v1 
@File    : train_test.py
@Author  : kaiwu
@Date    : 2022/6/29 9:54 

"""

from multiprocessing import Process
from framework.common.config.config_control import CONFIG
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.common_func import (
    stop_process_by_name,
    python_exec_shell,
)
from tools.app.learner_step_check import countCkpt
import framework.server.learner.learner as learner
import framework.server.actor.actor as actor
import framework.server.aisrv.aisrv as aisrv
import time
import requests as req
import click
import os


@click.command()
@click.option(
    "--battlesvr_addr", default="127.0.0.1:12345", help="battlesrv adrress, not neccessary, has default value"
)
def train(battlesvr_addr="127.0.0.1:12345"):
    # Stop the process that has already been started.
    # 停止已经启动的进程
    stop()

    # Modify the value in the environment variable to initiate training for the learner as soon as possible.
    # 修改环境变量里的值, 尽快让learner进行训练
    os.environ.update(
        {
            "run_mode": "train",
            "replay_buffer_capacity": "10",
            "preload_ratio": "10",
            "train_batch_size": "1",
            "actor_addrs":'{"train_one":["127.0.0.1:8888"], "train_two":["127.0.0.1:8888"]}',
            "learner_addrs":'{"train_one":["127.0.0.1:9999"], "train_two":["127.0.0.1:9999"]}',
        }
    )

    CONFIG.set_configure_file("conf/framework/learner.toml")
    CONFIG.parse_learner_configure()

    # Start the processes related to training
    # 启动训练相关进程
    procs = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=actor.main, name="actor"))
    procs.append(Process(target=aisrv.main, name="aisrv"))
    procs.append(
        Process(
            target=python_exec_shell,
            args=("sh tools/modelpool_start.sh learner",),
            name="modelpool",
        )
    )

    for proc in procs:
        proc.start()
        time.sleep(10)
        check(proc)

    # count the number of existing checkpoints
    # 计算已有的checkpoint数量
    oldCkpt = countCkpt()

    # start battle
    # 启动对战
    stopBattle(battlesvr_addr)
    startBattle(battlesvr_addr)

    # wait for process to end
    # 等待进程退出
    code = 0
    while True:
        if code > 0:
            break

        newCkpt = countCkpt()
        # Exit when there is a new checkpoint output
        # 有新的checkpoint产出即退出
        if newCkpt - oldCkpt > 0:
            break

        time.sleep(2)
        for proc in procs:
            check(proc)

    stop()

    print(f"will exit: {code}")
    print("\033[92m" + "test successful" + "\033[0m")
    exit(code)


def check(proc: Process):
    if proc.is_alive():
        print(f"{proc.name} is alive")
    else:
        raise Exception(f"{proc.name} is not alive, please check error log")


def stop():
    stop_process_by_name(KaiwuDRLDefine.SERVER_MODELPOOL)
    stop_process_by_name(KaiwuDRLDefine.SERVER_MODELPOOL_PROXY)
    stop_process_by_name(KaiwuDRLDefine.SERVER_AISRV)
    stop_process_by_name(KaiwuDRLDefine.SERVER_ACTOR)
    stop_process_by_name(KaiwuDRLDefine.SERVER_LEARNER)
    stop_process_by_name(KaiwuDRLDefine.SERVER_BATTLE_SRV)
    time.sleep(6)



def startBattle(battlesvr_addr):
    rsp = req.post(
        f"http://{battlesvr_addr}/kaiwu_drl.BattleSvr/Start", json={"max_battle": 1}
    )
    if rsp.status_code > 300:
        raise Exception("start battle fail")


def stopBattle(battlesvr_addr):
    rsp = req.post(f"http://{battlesvr_addr}/kaiwu_drl.BattleSvr/Stop", json={})
    if rsp.status_code > 300:
        raise Exception("stop battle fail")


if __name__ == "__main__":
    train()
