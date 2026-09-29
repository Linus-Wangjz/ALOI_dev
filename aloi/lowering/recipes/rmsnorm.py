"""RMSNorm decomposition across PIM and PNM."""

from __future__ import annotations

from aloi.ir.base import Operation, Value
from aloi.lowering.registry import LoweringRecipe, LoweringResult


class RMSNormRecipe(LoweringRecipe):
    name = "rms_norm.pim_pnm"
    semantic_op = "aloi.rms_norm"
    required_capabilities = (
        "pim.ewmul",
        "device.transfer",
        "pnm.reduce_sum",
        "pnm.rsqrt",
    )

    def lower(self, operation: Operation, operands: list[Value]) -> LoweringResult:
        output_type = operation.results[0].type
        attrs = self.base_attrs(operation)
        square = Operation(
            "pim.ewmul",
            [operands[0], operands[0]],
            [output_type],
            attrs={**attrs, "phase": "square"},
            result_names=[f"{operation.results[0].name}_square"],
        )
        to_pnm = Operation(
            "device.transfer",
            square.results,
            [output_type],
            attrs={**attrs, "source": "pim", "target": "pnm"},
            result_names=[f"{operation.results[0].name}_pnm"],
        )
        reduction = Operation(
            "pnm.reduce_sum",
            to_pnm.results,
            [output_type],
            attrs={**attrs, "axis": "hidden"},
            result_names=[f"{operation.results[0].name}_sum"],
        )
        rsqrt = Operation(
            "pnm.rsqrt",
            reduction.results,
            [output_type],
            attrs=attrs,
            result_names=[f"{operation.results[0].name}_rsqrt"],
        )
        to_pim = Operation(
            "device.transfer",
            rsqrt.results,
            [output_type],
            attrs={**attrs, "source": "pnm", "target": "pim"},
            result_names=[f"{operation.results[0].name}_pim"],
        )
        normalize = Operation(
            "pim.ewmul",
            [operands[0], to_pim.results[0], operands[1]],
            [output_type],
            attrs={**attrs, "phase": "normalize"},
            result_names=[operation.results[0].name],
        )
        operations = (square, to_pnm, reduction, rsqrt, to_pim, normalize)
        return LoweringResult(operations, tuple(normalize.results))
