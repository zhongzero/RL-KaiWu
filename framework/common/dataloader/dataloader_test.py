#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import unittest
import os
from framework.common.config.config_control import CONFIG
from framework.common.dataloader.dataloader import DataLoader

class DataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_all(self):
        dataloader = DataLoader()


if __name__ == '__main__':
    unittest.main()
