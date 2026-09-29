"""AIM IR verification utilities."""

from __future__ import annotations

from aloi.ir.base import Module


def verify_aim_module(module: Module) -> None:
    module.verify()
    for function in module.functions:
        for operation in function.operations:
            if not operation.op.startswith("aim."):
                raise ValueError(f"non-AIM operation in AIM IR: {operation.op}")
