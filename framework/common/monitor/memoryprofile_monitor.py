#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import wraps
from framework.common.monitor.monitor_base import MonitorDecoratorBase


# ref: https://pypi.org/project/memory-profiler/
# need pip install guppy3
# need pip install memory_profiler

class MemoryProfilerMonitorDecorator(MonitorDecoratorBase):
    def __init__(self,
                 output_path,
                 do_profile=False,
                 start_profile_cnt=0,
                 end_profile_cnt=99999,
                 monitor_every_k_cnt=1):
        super().__init__(
            "memory_profiler",
            output_path,
            do_profile,
            start_profile_cnt,
            end_profile_cnt,
            monitor_every_k_cnt)
        try:
            from memory_profiler import profile
            self._profile = profile
        except ImportError:
            self._valid_monitor = False
            self.logger.error("monitor Cannot import memory_profiler. Try pip install memory_profiler")

    def __call__(self, func):
        # decorator
        self._func = func
        if not self._valid_monitor:
            self.logger.error("monitor Cannot import memory_profiler. Try pip install guppy3")
            return func

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            do_monitor = self._do_monitor()
            if do_monitor:
                self._output_path = self._spec_output_path(self._output_path, "memory_profiler", ".mp")
                path = self._output_path_add_cnt(self._output_path, self.cnt)
                file_stream = open(path, 'w')
                self.logger.info(f"monitor ### memory profiler, start profile {self._func}")
                func = self._profile(
                    self._func,
                    stream=file_stream,
                    precision=4,
                )
                return_context = func(*args, **kwargs)
            else:
                return_context = self._func(*args, **kwargs)
            self.cnt += 1
            return return_context

        return wrapped_function

    def start_monitor(self):
        pass

    def end_monitor(self):
        pass
