#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import datetime
import linecache
import json
from framework.common.monitor.monitor_base import MonitorDecoratorBase


# ref: https://docs.python.org/3/library/tracemalloc.html

class TraceMallocMonitorDecorator(MonitorDecoratorBase):
    def __init__(self,
                 top_n,
                 output_path,
                 do_profile=False,
                 start_profile_cnt=0,
                 end_profile_cnt=99999,
                 monitor_every_k_cnt=1):
        super().__init__(
            "tracemalloc",
            output_path,
            do_profile,
            start_profile_cnt,
            end_profile_cnt,
            monitor_every_k_cnt)

        try:
            import tracemalloc
            self._tracemalloc = tracemalloc
        except ImportError:
            self.logger.error("monitor Cannot import tracemalloc.")

        self._top_n = top_n

        self._first_snapshot = None
        self._first_time = datetime.datetime.now()
        self._snapshot = None
        self._curr_time = datetime.datetime.now()

    def start_monitor(self):
        if not self._first_snapshot:
            self._tracemalloc.start()
            self._first_snapshot = self._take_snapshot()
            self._first_time = datetime.datetime.now()
        self._snapshot = self._take_snapshot()
        self._curr_time = datetime.datetime.now()

    def end_monitor(self):
        new_snapshot = self._take_snapshot()
        curr_time = datetime.datetime.now()

        stats_summary = dict()

        display = ""
        display += f"#### Tracemalloc Memory Profile. Current snapshot at {curr_time} ####\n"
        stats = new_snapshot.statistics("lineno")
        display += self._display_snapshot(stats)
        display += "\n"
        stats_summary['current_snapshot'] = self._snapshot_stats_dict(stats)
        stats_summary['current_snapshot']['timestamp'] = int(curr_time.timestamp())

        display += f"#### Tracemalloc Memory Profile. Compare memory snapshot from " \
                   f"{self._curr_time} to {curr_time} #### \n"
        stats = new_snapshot.compare_to(self._snapshot, 'lineno', cumulative=True)
        display += self._display_snapshot(stats, is_diff=True)
        display += "\n"
        stats_summary['diff_func'] = self._snapshot_stats_dict(stats, is_diff=True)
        stats_summary['diff_func']['start_timestamp'] = int(self._curr_time.timestamp())
        stats_summary['diff_func']['end_timestamp'] = int(curr_time.timestamp())

        display += f"#### Tracemalloc Memory Profile. Compare memory snapshot from " \
                   f"{self._first_time} to {curr_time} #### \n"
        stats = new_snapshot.compare_to(self._first_snapshot, 'lineno', cumulative=True)
        display += self._display_snapshot(stats, is_diff=True)
        display += "\n"
        stats_summary['diff_first'] = self._snapshot_stats_dict(stats, is_diff=True)
        stats_summary['diff_first']['start_timestamp'] = int(self._curr_time.timestamp())
        stats_summary['diff_first']['end_timestamp'] = int(curr_time.timestamp())

        self.logger.info(f'monitor {display}')

        self._output_path = self._spec_output_path(self._output_path, "tracemalloc", ".json")
        path = self._output_path_add_cnt(self._output_path, self.cnt)
        with open(path, 'w') as file:
            json.dump(stats_summary, file, indent=4)

    def stop_monitor(self):
        self._tracemalloc.stop()

    def _take_snapshot(self):
        return self._tracemalloc.take_snapshot().filter_traces((
            self._tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            self._tracemalloc.Filter(False, "<unknown>"),
            self._tracemalloc.Filter(False, self._tracemalloc.__file__),
            self._tracemalloc.Filter(False, linecache.__file__)
        ))

    def _display_snapshot(self, stats, is_diff=False):
        top_stats = stats[:self._top_n]

        display = ""
        display += "## Top %d Stats\n" % self._top_n

        for index, stat in enumerate(top_stats):
            frame = stat.traceback[0]
            filename = os.sep.join(frame.filename.split(os.sep))
            if is_diff:
                display += "#%s: %s:%s: %.1f KiB (+%.1f Kib)\n" % (
                    index, filename, frame.lineno, stat.size / 1024, stat.size_diff / 1024)
            else:
                display += "#%s: %s:%s: %.1f KiB\n" % (index, filename, frame.lineno, stat.size / 1024)
            for frame in stat.traceback:
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    display += '    %s\n' % line

        other = top_stats[self._top_n:]
        if other:
            size = sum(stat.size for stat in other)
            display += "%s other: %.1f KiB\n" % (len(other), size / 1024)
        total = sum(stat.size for stat in stats)
        display += "Total allocated size: %.1f KiB\n" % (total / 1024)
        return display

    def _snapshot_stats_dict(self, stats, is_diff=False):
        top_stats = stats[:self._top_n]

        stats_result = dict()

        stats_result["top_n"] = self._top_n
        stats_result["unit"] = "KiB"
        stats_result['stats'] = []
        for index, stat in enumerate(top_stats):
            frame = stat.traceback[0]
            filename = os.sep.join(frame.filename.split(os.sep))
            a_stat = dict()
            a_stat["rank"] = index
            a_stat["alloc"] = stat.size / 1024
            a_stat["fileno"] = f"{filename}:{frame.lineno}"
            if is_diff:
                a_stat['diff'] = stat.size_diff / 1024
            a_stat["lines"] = []
            for frame in stat.traceback:
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    a_stat['lines'].append(line)
            stats_result['stats'].append(a_stat)
        stats_result['other_len'] = 0
        stats_result['other_alloc'] = 0
        other = top_stats[self._top_n:]
        if other:
            size = sum(stat.size for stat in other)
            stats_result['other_len'] = len(other)
            stats_result['other_alloc'] = size / 1024
        total = sum(stat.size for stat in stats)
        stats_result['total_alloc'] = total / 1024
        return stats_result
