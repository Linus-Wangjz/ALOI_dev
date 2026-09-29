from __future__ import annotations

import unittest
from typing import ClassVar

from aloi.ir.aim import verify_aim_module
from aloi.ir.device import verify_device_module
from aloi.ir.semantic import verify_semantic_module
from aloi.pipeline import CompilationResult
from tests.helpers import compilation


class ParallelAndLoweringTest(unittest.TestCase):
    result: ClassVar[CompilationResult]

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compilation()

    def test_tp_ownership_and_automatic_collectives(self) -> None:
        module = self.result.sharded
        verify_semantic_module(module)
        function = module.get_function()
        q = next(
            value
            for op in function.operations
            if op.attrs.get("semantic_name") == "attention.q"
            for value in op.results
            if "q_head" in value.type.axes
        )
        self.assertEqual(q.type.shape, (1, 1, 8, 128))
        cache = next(value for value in function.inputs if value.name == "k_cache")
        self.assertEqual(cache.type.shape, (1, 1, 32, 128))
        collectives = [op for op in function.operations if op.op == "aloi.all_reduce"]
        self.assertEqual(len(collectives), 2)
        self.assertEqual(
            {op.attrs["semantic_name"] for op in collectives},
            {"attention.output", "feed_forward.down"},
        )

    def test_device_ir_is_fully_legalized(self) -> None:
        module = self.result.device
        verify_device_module(module)
        operations = [op.op for op in module.get_function().operations]
        self.assertFalse(any(op.startswith("aloi.") for op in operations))
        self.assertIn("pim.gemv", operations)
        self.assertIn("pnm.softmax", operations)
        self.assertIn("cxl.all_reduce", operations)
        self.assertIn("device.transfer", operations)

    def test_allocated_aim_has_physical_metadata(self) -> None:
        module = self.result.aim
        verify_aim_module(module)
        for operation in module.get_function().operations:
            physical = operation.attrs["physical"]
            self.assertIn("channel_group", physical)
            self.assertIn("bank_group", physical)
            self.assertIn("row_range", physical)
            self.assertIn("gb_allocation", physical)
            self.assertIn("physical_tile", physical)
        operations = [op.op for op in module.get_function().operations]
        self.assertIn("aim.pim.mac_dram_gb", operations)
        self.assertIn("aim.read_acc", operations)
        self.assertIn("aim.pnm.softmax", operations)
        self.assertIn("aim.cxl.all_reduce", operations)


if __name__ == "__main__":
    unittest.main()
