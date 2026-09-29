"""Linear lowering."""

from __future__ import annotations

from aloi.ir.base import Operation, Value
from aloi.lowering.registry import LoweringRecipe, LoweringResult


class LinearRecipe(LoweringRecipe):
    name = "linear.pim_gemv"
    semantic_op = "aloi.linear"
    required_capabilities = ("pim.gemv",)

    def lower(self, operation: Operation, operands: list[Value]) -> LoweringResult:
        lowered = Operation(
            "pim.gemv",
            operands,
            [value.type for value in operation.results],
            attrs=self.base_attrs(operation),
            result_names=[value.name for value in operation.results],
        )
        return LoweringResult((lowered,), tuple(lowered.results))
