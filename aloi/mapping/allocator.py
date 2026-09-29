"""Deterministic physical channel/bank/row/GB allocation."""

from __future__ import annotations

from copy import deepcopy
from math import ceil

from aloi.device.spec import DeviceSpec
from aloi.ir.base import Module, Pass
from aloi.ir.device import verify_device_module


class PhysicalAllocator(Pass):
    def __init__(self, device_spec: DeviceSpec) -> None:
        self.device_spec = device_spec

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        channels = int(self.device_spec.parameter("physical.channels", 32))
        bank_groups = int(self.device_spec.parameter("physical.bank_groups", 4))
        row_bytes = int(self.device_spec.parameter("physical.row_bytes", 2048))
        gb_bytes = int(self.device_spec.parameter("physical.global_buffer_bytes", 2048))
        if min(channels, bank_groups, row_bytes, gb_bytes) < 1:
            raise ValueError("physical device parameters must be positive")

        row_cursor = 0
        gb_cursor = 0
        operation_index = 0
        for function in result.functions:
            for operation in function.operations:
                weight = next(
                    (value for value in operation.operands if value.type.role == "weight"),
                    None,
                )
                activation = next(
                    (value for value in operation.operands if value.type.role != "weight"),
                    operation.results[0] if operation.results else None,
                )
                weight_bytes = weight.type.nbytes if weight is not None else 0
                rows = max(1, ceil(weight_bytes / row_bytes))
                gb_size = min(gb_bytes, activation.type.nbytes if activation is not None else 1)
                if gb_cursor + gb_size > gb_bytes:
                    gb_cursor = 0
                mapping = operation.attrs.get("mapping", {})
                operation.attrs["physical"] = {
                    "device": operation.op.split(".", 1)[0],
                    "channel_group": operation_index % channels,
                    "bank_group": operation_index % bank_groups,
                    "row_range": [row_cursor, row_cursor + rows - 1],
                    "gb_allocation": [gb_cursor, gb_cursor + gb_size - 1],
                    "physical_tile": mapping.get("tile", {}),
                }
                row_cursor += rows
                gb_cursor += gb_size
                operation_index += 1
        result.attrs["stage"] = "allocated"
        verify_device_module(result)
        return result
