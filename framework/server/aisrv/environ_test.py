#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
from framework.common.utils.common_func import Context
from framework.server.aisrv.environ import Environ


class MockEnviron(Environ):
    def __init__(self, simu_ctx):
        super(MockEnviron, self).__init__(simu_ctx)

    def init(self):
        pass

    def reset(self):
        pass

    def step(self, actions, extra_info=None):
        pass

    def reject(self, e):
        pass

    def finsh(self):
        pass

    def client_id(self):
        return 'fake client id'

    def client_version(self):
        return 'fake client version'

    def ep_id(self):
        return -1

    def policy_mapping_fn(self, agent_id):
        return 'train'


class TestEnviron(unittest.TestCase):
    def test_init(self):
        fake_simu_ctx = Context()
        MockEnviron(simu_ctx=fake_simu_ctx)


if __name__ == '__main__':
    unittest.main()
