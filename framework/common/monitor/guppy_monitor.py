#!/usr/bin/env python
# -*- coding: utf-8 -*-


from framework.common.monitor.monitor_base import MonitorDecoratorBase

# ref: https://pypi.org/project/guppy3/
# need pip install guppy3

class GuppyMonitorDecorator(MonitorDecoratorBase):
    def __init__(self,
                 output_path,
                 do_profile=False,
                 start_profile_cnt=0,
                 end_profile_cnt=99999,
                 monitor_every_k_cnt=1):

        super().__init__(
            "guppy",
            output_path,
            do_profile,
            start_profile_cnt,
            end_profile_cnt,
            monitor_every_k_cnt)

        try:
            from guppy import hpy
            self._hpy = hpy
        except ImportError:
            self._valid_monitor = False
            self.logger.error("monitor Cannot import guppy. Try pip install guppy3")

    def start_monitor(self):
        self.profile = self._hpy()
        self.profile.setrelheap()
        self.logger.warning(f"monitor # Start collecting memory profile via guppy")

    def end_monitor(self):
        h = self.profile.heap()
        self.logger.warning(f"monitor #### Guppy Memory Profile {self._func} ####")
        self.logger.warning(f'monitor {h.__str__()}')

        self._output_path = self._spec_output_path(self._output_path, "guppy_profile", ".guppy")
        path = self._output_path_add_cnt(self._output_path, self.cnt)

        try:
            with open(f"{path}", 'w') as file:
                file.write(h.__str__())
        except IOError as e:
            self.logger.error(f'monitor {e}')
