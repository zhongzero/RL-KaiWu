#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from framework.common.utils.slots import Slots


class SlotsV2Test(unittest.TestCase):
    def test_all(self):
        slots = Slots(1000)
        slots.register_group("g_a")
        slots.register_group("g_b")

        self.assertEqual(slots.used_slot(), 0)

        slot_id = slots.get_slot()

        self.assertEqual(slot_id, 0)
        self.assertEqual(slots.used_slot(), 1)

        output_pipe = slots.get_output_pipe("g_a", slot_id)
        output_pipe.send("Hello")

        input_pipe = slots.get_input_pipe("g_a", slot_id)
        data = input_pipe.recv()

        self.assertEqual(data, "Hello")

        slots.put_slot(slot_id)
        self.assertEqual(slots.used_slot(), 0)


if __name__ == '__main__':
    unittest.main()
