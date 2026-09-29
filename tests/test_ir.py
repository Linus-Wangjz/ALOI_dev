from __future__ import annotations

import unittest

from aloi.ir import Function, Module, Operation, Printer, TensorType, Value
from aloi.ir.semantic import verify_semantic_module


class IRTest(unittest.TestCase):
    def test_m0_manual_ir_prints(self) -> None:
        activation = TensorType.get(
            (1, 1, 8192), role="activation", axes=("batch", "token", "hidden")
        )
        weight = TensorType.get((8192, 8192), role="weight", axes=("out_feature", "in_feature"))
        gamma = TensorType.get((8192,), role="weight", axes=("hidden",))
        x, w, g = Value("x", activation), Value("w", weight), Value("g", gamma)
        function = Function("main", inputs=[x, w, g])
        norm = function.add_op(
            Operation("aloi.rms_norm", [x, g], [activation], result_names=["norm"])
        )
        linear = function.add_op(
            Operation("aloi.linear", [norm.results[0], w], [activation], result_names=["y"])
        )
        function.outputs = list(linear.results)
        module = Module([function])

        verify_semantic_module(module)
        text = Printer().print_module(module)
        self.assertIn("aloi.rms_norm", text)
        self.assertIn("aloi.linear", text)
        self.assertIn("axes=[batch, token, hidden]", text)

    def test_sharded_type_preserves_global_shape(self) -> None:
        typ = TensorType.get((1, 1, 64, 128), axes=("batch", "token", "q_head", "head_dim"))
        local = typ.shard("q_head", "tp", 8)
        self.assertEqual(local.shape, (1, 1, 8, 128))
        self.assertEqual(local.global_shape, typ.shape)
        self.assertEqual(local.sharding.parts, 8)  # type: ignore[union-attr]

    def test_invalid_sharding_is_rejected(self) -> None:
        typ = TensorType.get((1, 7), axes=("batch", "feature"))
        with self.assertRaises(ValueError):
            typ.shard("feature", "tp", 8)


if __name__ == "__main__":
    unittest.main()
