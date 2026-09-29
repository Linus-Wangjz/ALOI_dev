"""Device IR verification utilities."""

from __future__ import annotations

from aloi.ir.base import Module

DEVICE_PREFIXES = ("pim.", "pnm.", "cxl.", "device.")


def verify_device_module(module: Module) -> None:
    module.verify()
    for function in module.functions:
        for operation in function.operations:
            if not operation.op.startswith(DEVICE_PREFIXES):
                raise ValueError(f"operation was not legalized to Device IR: {operation.op}")
