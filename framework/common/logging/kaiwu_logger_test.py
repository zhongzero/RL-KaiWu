#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import datetime
import unittest
from framework.common.config.config_control import CONFIG
from framework.common.logging.kaiwu_logger import KaiwuLogger

class LoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.set_configure_file("/data/projects/kaiwu-fwk/conf/framework/actor.toml")
        CONFIG.parse_actor_configure()

    def test_logging(self):
        print(f'log level is {CONFIG.level}')

        self.logger = KaiwuLogger()
        self.logger.setLoggerFormat(f"/logging/test_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'logging')

        self.logger.debug('logging 123')
        self.logger.info('logging 456')
        self.logger.error('logging 789')
    
    def test_logging_level(self):
        from loguru import logger

        # 清除打印到屏幕的日志输出, 即sys.stderr
        logger.remove(handler_id=None)
        logger.add("file.log", format="{time} {level} {message}", filter="", level="INFO")

        logger.debug("这是一条debug日志")
        logger.info("这是一条info日志")

        logger.add("file2.log", format="{time} {level} {message}", filter="", level="DEBUG")

        logger.debug("这是一条debug日志")
        logger.info("这是一条info日志")

if __name__ == '__main__':
    unittest.main()