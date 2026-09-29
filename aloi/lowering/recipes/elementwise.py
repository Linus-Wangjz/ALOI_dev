"""Elementwise, view, and collective lowering recipes."""

from __future__ import annotations

from aloi.ir.base import Operation, Value
from aloi.lowering.registry import LoweringRecipe, LoweringResult


class GenericRecipe(LoweringRecipe):
    def __init__(self, name: str, semantic_op: str, device_op: str) -> None:
        self.name = name
        self.semantic_op = semantic_op
        self.device_op = device_op
        self.required_capabilities = (device_op,)

    def lower(self, operation: Operation, operands: list[Value]) -> LoweringResult:
        lowered = Operation(
            self.device_op,
            operands,
            [value.type for value in operation.results],
            attrs=self.base_attrs(operation),
            result_names=[value.name for value in operation.results],
        )
        return LoweringResult((lowered,), tuple(lowered.results))


def elementwise_recipes() -> list[LoweringRecipe]:
    pairs = [
        ("silu.pnm", "aloi.silu", "pnm.silu"),
        ("add.pim", "aloi.add", "pim.ewadd"),
        ("mul.pim", "aloi.mul", "pim.ewmul"),
        ("reshape.pnm", "aloi.reshape", "pnm.reshape"),
        ("transpose.pnm", "aloi.transpose", "pnm.transpose"),
        ("expand.pnm", "aloi.expand", "pnm.expand"),
        ("slice.pnm", "aloi.slice", "pnm.slice"),
        ("all_reduce.cxl", "aloi.all_reduce", "cxl.all_reduce"),
        ("broadcast.cxl", "aloi.broadcast", "cxl.broadcast"),
    ]
    return [GenericRecipe(*pair) for pair in pairs]
