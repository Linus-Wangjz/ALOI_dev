"""Attention and persistent-state lowering recipes."""

from __future__ import annotations

from aloi.ir.base import Operation, Value
from aloi.lowering.registry import LoweringRecipe, LoweringResult


class _SingleOpRecipe(LoweringRecipe):
    device_op: str

    def lower(self, operation: Operation, operands: list[Value]) -> LoweringResult:
        lowered = Operation(
            self.device_op,
            operands,
            [value.type for value in operation.results],
            attrs=self.base_attrs(operation),
            result_names=[value.name for value in operation.results],
        )
        return LoweringResult((lowered,), tuple(lowered.results))


class RopeRecipe(_SingleOpRecipe):
    name = "rope.pnm"
    semantic_op = "aloi.rope"
    required_capabilities = ("pnm.rope",)
    device_op = "pnm.rope"


class KVUpdateRecipe(_SingleOpRecipe):
    name = "kv_update.pnm"
    semantic_op = "aloi.kv_update"
    required_capabilities = ("pnm.kv_update",)
    device_op = "pnm.kv_update"


class MatmulRecipe(_SingleOpRecipe):
    name = "matmul.pnm"
    semantic_op = "aloi.matmul"
    required_capabilities = ("pnm.matmul",)
    device_op = "pnm.matmul"


class SoftmaxRecipe(_SingleOpRecipe):
    name = "softmax.pnm"
    semantic_op = "aloi.softmax"
    required_capabilities = ("pnm.softmax",)
    device_op = "pnm.softmax"


def attention_recipes() -> list[LoweringRecipe]:
    return [RopeRecipe(), KVUpdateRecipe(), MatmulRecipe(), SoftmaxRecipe()]
