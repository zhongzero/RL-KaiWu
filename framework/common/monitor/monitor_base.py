#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import datetime
from functools import wraps
from framework.common.logging.kaiwu_logger import KaiwuLogger
from framework.common.config.config_control import CONFIG


class MonitorDecoratorBase:
    def __init__(self,
                 name,
                 output_path,
                 do_profile=False,
                 start_cnt=0,
                 end_cnt=99999,
                 monitor_every_k_cnt=1):
        """
        :param output_path: 
        :param do_profile: if use profile
        :param start_cnt: How many times the target method is called to start recording
        :param end_cnt: How many times the target method is called to end the record
        :param monitor_every_k_cnt: How many times to record performance per interval
        """
        if do_profile:
            if os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self._name = name
        self._output_path = output_path
        self.cnt = 0
        self._start_cnt = start_cnt
        self._end_cnt = end_cnt
        self._monitor_every_k_cnt = monitor_every_k_cnt
        self._do_profile = do_profile


        self.logger = KaiwuLogger()
        self.logger.setLoggerFormat(f"/common/monitor_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'monitor')

        self._func = None

        self._valid_monitor = True

    def _do_monitor(self):
        if not self._do_profile:
            return False
        if self.cnt > self._end_cnt:
            self.stop_monitor()

        if self._start_cnt <= self.cnt <= self._end_cnt:
            return (self.cnt - self._start_cnt) % self._monitor_every_k_cnt == 0
        return False

    def __call__(self, func):
        if not self._valid_monitor:
            self.logger.warning(f"monitor Invalid monitor {self._name}")
            return func

        # decorator
        self._func = func

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            do_monitor = self._do_monitor()
            if do_monitor:
                self.start_monitor()
            return_context = func(*args, **kwargs)
            if do_monitor:
                self.end_monitor()
            self.cnt += 1
            return return_context

        return wrapped_function

    def __del__(self):
        self.stop_monitor()

    def start_monitor(self):
        raise NotImplementedError

    def end_monitor(self):
        raise NotImplementedError

    def stop_monitor(self):
        pass

    @staticmethod
    def _spec_output_path(output_path, file_name, suffix):
        assert suffix.startswith('.')
        if os.path.isdir(output_path):
            os.makedirs(name=output_path, exist_ok=True)
            output_path = os.path.join(output_path, file_name + suffix)
        if not output_path.endswith(suffix):
            output_path += suffix
        return output_path

    @staticmethod
    def _output_path_add_cnt(output_path, cnt):
        # a/b/ccc.ddd --> /a/b/ccc_cnt.ddd
        prefix_path, suffix_path = output_path.rsplit('.', 1)
        path = "".join([prefix_path, "_", str(cnt), ".", suffix_path])
        return path
