#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cProfile
import pstats
import io
import os
from pstats import SortKey
from framework.common.monitor.monitor_base import MonitorDecoratorBase


# Python cProfile
# need pip3 install flameprof

class CProfileMonitorDecorator(MonitorDecoratorBase):
    def __init__(self,
                 top_n,
                 output_path,
                 do_profile=False,
                 start_profile_cnt=0,
                 end_profile_cnt=99999,
                 monitor_every_k_cnt=1):
        super().__init__(
            "cProfile",
            output_path,
            do_profile,
            start_profile_cnt,
            end_profile_cnt,
            monitor_every_k_cnt)
        self._top_n = top_n

    def start_monitor(self):
        self.profile = cProfile.Profile()
        self.profile.enable()
        self.logger.warning(f"monitor # Start collecting performance profile via cProfile")

    def end_monitor(self):
        self.profile.disable()

        def dump(sortby):
            s = io.StringIO()
            ps = pstats.Stats(self.profile, stream=s).sort_stats(sortby)
            ps.print_stats()
            result = s.getvalue().split("\n")
            result = result[:min(self._top_n + 5, len(result))]
            result_str = "".join(r + "\n" for r in result)
            self.logger.warning(f'monitor {result_str}')

        self.logger.warning(f"monitor #### Performance Profile {self._func} ####")
        dump(SortKey.CUMULATIVE)
        dump(SortKey.TIME)

        self._output_path = self._spec_output_path(self._output_path, "cprofile", ".stats")
        path = self._output_path_add_cnt(self._output_path, self.cnt)
        try:
            self.profile.dump_stats(path)
            os.system(f'python -m flameprof {path} > {path}.svg')
            os.remove(path)
            if not os.path.exists(f"{path}.svg"):
                self.logger.error("monitor Cannot call flameprof. Try pip install flameprof")
                self._valid_monitor = False
        except IOError as e:
            self.logger.error(f'monitor {e}')
            self._valid_monitor = False
