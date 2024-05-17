#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import subprocess
import signal
import psutil
from framework.common.monitor.monitor_base import MonitorDecoratorBase


# ref: https://github.com/benfred/py-spy
# need pip install py-spy
# need pip install psutil

class PySpyMonitorDecorator(MonitorDecoratorBase):
    def __init__(self,
                 output_path,
                 do_profile=False,
                 start_profile_cnt=0,
                 end_profile_cnt=99999,
                 monitor_every_k_cnt=1):
        super().__init__(
            "py-spy",
            output_path,
            do_profile,
            start_profile_cnt,
            end_profile_cnt,
            monitor_every_k_cnt)

    def start_monitor(self):
        self._output_path = self._spec_output_path(self._output_path, "pyspy_profile", ".svg")
        path = self._output_path_add_cnt(self._output_path, self.cnt)
        self._spy_proc = subprocess.Popen(f'py-spy record -o {path} --pid {os.getpid()}', shell=True)
        self.logger.info(f"monitor # Start collecting performance profile via py-spy")

    def end_monitor(self):
        pid = self._spy_proc.pid
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            os.kill(child.pid, signal.SIGINT)
        os.kill(pid, signal.SIGINT)
        path = self._output_path_add_cnt(self._output_path, self.cnt)
        if not os.path.exists(path):
            self._valid_monitor = False
            self.logger.error(f"monitor Cannot call py-spy. Save {path} failed")
