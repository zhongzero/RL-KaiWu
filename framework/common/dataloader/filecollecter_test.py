#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import os
from framework.common.config.config_control import CONFIG
from framework.common.dataloader.filecollecter import FileCollecter

class FileCollecterTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_all(self):
        fillcollecter = FileCollecter()


if __name__ == '__main__':
    unittest.main()
