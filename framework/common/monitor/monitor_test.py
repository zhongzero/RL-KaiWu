#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import multiprocessing as mp
from framework.common.monitor.cprofile_monitor import CProfileMonitorDecorator
from framework.common.monitor.guppy_monitor import GuppyMonitorDecorator
from framework.common.monitor.pyspy_monitor import PySpyMonitorDecorator
from framework.common.monitor.tracemalloc_monitor import TraceMallocMonitorDecorator
from framework.common.monitor.memoryprofile_monitor import MemoryProfilerMonitorDecorator


class MyTestCase(unittest.TestCase):
    def test_memoryprofiler_monitor(self):
        # Decorator
        # case 1: decorator, recommend
        # case 2: function wrapper
        # case 3: call start_monitor/end_monitor, manually update cnt
        print("test memoryprofiler monitor")

        def b():
            a = "a" * 1024
            return a

        # case 1.
        @MemoryProfilerMonitorDecorator(
            output_path="./test/",
            do_profile=True,
            start_profile_cnt=0,
            end_profile_cnt=6)
        def my_func():
            a = "a"
            for __ in range(100):
                a += b()
            return a

        for __ in range(10):
            my_func()

        # case 2.
        mp = MemoryProfilerMonitorDecorator(
            output_path="./test/",
            do_profile=True,
            start_profile_cnt=7,
            end_profile_cnt=9)

        for __ in range(10):
            mp(my_func)()

    def test_guppy_monitor(self):

        print("test guppy monitor")

        # case 1.
        @GuppyMonitorDecorator(output_path="./test/",
                               do_profile=True,
                               start_profile_cnt=0,
                               end_profile_cnt=6,
                               monitor_every_k_cnt=1)
        def test_func():
            a = [1] * 10000
            return a

        for __ in range(10):
            test_func()

        # case 3.
        guppy = GuppyMonitorDecorator(output_path="./test/",
                                      do_profile=True,
                                      start_profile_cnt=7,
                                      end_profile_cnt=9,
                                      monitor_every_k_cnt=1)

        b = []
        for __ in range(10):
            guppy.start_monitor()
            b += test_func()
            guppy.end_monitor()
            guppy.cnt += 1

    def test_tracemalloc_monitor(self):
        print("test tracemalloc monitor")

        def b():
            a = "a" * 1024
            return a

        @TraceMallocMonitorDecorator(top_n=10,
                                     output_path="./test/",
                                     do_profile=True,
                                     start_profile_cnt=0,
                                     end_profile_cnt=6)
        def my_func():
            a = "a"
            for __ in range(100):
                a += b()
            return a

        for __ in range(10):
            my_func()

    def test_cprofile_monitor(self):
        print("test cprofile monitor")
        import time
        import random

        def f1():
            time.sleep(0.01)

        def f2():
            time.sleep(0.02)

        def f3():
            time.sleep(0.03)

        @CProfileMonitorDecorator(top_n=10,
                                  output_path='./test/',
                                  do_profile=True,
                                  start_profile_cnt=4,
                                  end_profile_cnt=5, )
        def test_func():
            for __ in range(random.randint(1, 5)):
                f1()
            for __ in range(random.randint(1, 5)):
                f2()
            for __ in range(random.randint(1, 5)):
                f3()

        for __ in range(10):
            test_func()

    def test_pyspy_monitor(self):
        print("test pyspy monitor")
        import time
        import random

        def f1():
            time.sleep(0.01)

        def f2():
            time.sleep(0.02)

        def f3():
            time.sleep(0.03)

        @PySpyMonitorDecorator(output_path='./test/',
                               do_profile=True,
                               start_profile_cnt=4,
                               end_profile_cnt=5, )
        def test_func():
            for __ in range(random.randint(1, 5)):
                f1()
            for __ in range(random.randint(1, 5)):
                f2()
            for __ in range(random.randint(1, 5)):
                f3()

        for i in range(10):
            test_func()


if __name__ == '__main__':
    unittest.main()
