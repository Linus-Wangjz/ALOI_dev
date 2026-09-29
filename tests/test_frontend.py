from __future__ import annotations

import unittest
from typing import ClassVar

import torch

from aloi.frontend import ImportExportedProgram
from aloi.frontend.torch_export import ExportBundle
from aloi.ir.base import Module
from aloi.ir.semantic import verify_semantic_module
from examples.llama2_70b.config import LLAMA2_70B
from examples.llama2_70b.export import export_block
from examples.llama2_70b.model import ToyLlama70BBlock, example_inputs


class FrontendTest(unittest.TestCase):
    bundle: ClassVar[ExportBundle]
    module: ClassVar[Module]

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = export_block(32)
        cls.module = ImportExportedProgram().run(cls.bundle.program)

    def test_exact_config_and_meta_only_weights(self) -> None:
        cfg = LLAMA2_70B
        self.assertEqual((cfg.hidden_size, cfg.num_attention_heads), (8192, 64))
        self.assertEqual((cfg.num_key_value_heads, cfg.head_dim), (8, 128))
        self.assertEqual(cfg.intermediate_size, 28672)
        expected = {
            "wq": (8192, 8192),
            "wk": (1024, 8192),
            "wv": (1024, 8192),
            "wo": (8192, 8192),
            "w1": (28672, 8192),
            "w3": (28672, 8192),
            "w2": (8192, 28672),
        }
        model = ToyLlama70BBlock()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.device.type, "meta")
            self.assertEqual(parameter.data_ptr(), 0)
            if name in expected:
                self.assertEqual(tuple(parameter.shape), expected[name])

    def test_decode_input_shapes(self) -> None:
        x, k_cache, v_cache, position = example_inputs(2048)
        self.assertEqual(tuple(x.shape), (1, 1, 8192))
        self.assertEqual(tuple(k_cache.shape), (1, 8, 2048, 128))
        self.assertEqual(tuple(v_cache.shape), (1, 8, 2048, 128))
        self.assertEqual(position.dtype, torch.int64)

    def test_torch_export_and_semantic_import(self) -> None:
        exported = str(self.bundle.program)
        for op in ("aloi.linear", "aloi.rms_norm", "aloi.rope", "aloi.kv_update"):
            self.assertIn(op, exported)
        verify_semantic_module(self.module)
        operations = [op.op for op in self.module.get_function().operations]
        for op in (
            "aloi.rms_norm",
            "aloi.linear",
            "aloi.rope",
            "aloi.kv_update",
            "aloi.matmul",
            "aloi.softmax",
            "aloi.silu",
            "aloi.add",
            "aloi.mul",
        ):
            self.assertIn(op, operations)
        all_values = [
            *self.module.get_function().inputs,
            *(value for op in self.module.get_function().operations for value in op.results),
        ]
        self.assertTrue(all(value.type.role for value in all_values))
        self.assertTrue(all(not value.type.rank or value.type.axes for value in all_values))

    def test_functional_persistent_cache_outputs(self) -> None:
        outputs = self.module.get_function().outputs
        self.assertEqual(len(outputs), 3)
        self.assertTrue(outputs[1].type.persistent)
        self.assertTrue(outputs[2].type.persistent)
        self.assertEqual(outputs[1].type.role, "kv_cache")


if __name__ == "__main__":
    unittest.main()
