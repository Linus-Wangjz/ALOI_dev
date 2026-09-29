"""Emit aim_simulator-compatible trace text from AIM IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aloi.backend.aim.isr import AimInstruction, RegisterWrite
from aloi.ir.aim import verify_aim_module
from aloi.ir.base import Module, Operation


def _physical(operation: Operation) -> dict[str, Any]:
    value = operation.attrs.get("physical")
    if not isinstance(value, dict):
        raise ValueError(f"{operation.op} has no physical mapping")
    return value


def _opsize(operation: Operation) -> int:
    physical = _physical(operation)
    tile = physical.get("physical_tile", {})
    factor = int(tile.get("in_feature", 256)) if isinstance(tile, dict) else 256
    return max(1, min(64, (factor + 15) // 16))


def _comment(prefix: str, operation: Operation) -> str:
    semantic_name = operation.attrs.get("semantic_name", "")
    return f"# {prefix} {operation.op.removeprefix('aim.')} {semantic_name}".rstrip()


class AimTraceEmitter:
    """Translate PIM AIM ops to ISR and preserve PNM/CXL semantics as comments."""

    channel_mask = "0xffffffff"

    def emit(self, module: Module) -> str:
        verify_aim_module(module)
        lines = [
            "# ALOI v0.1 AiM simulator trace",
            "# PNM and CXL operations are retained as comments by design.",
        ]
        for function in module.functions:
            for operation in function.operations:
                lines.extend(self._emit_operation(operation))
        lines.append(AimInstruction("EOC").render())
        return "\n".join(lines) + "\n"

    def write(self, module: Module, path: str | Path) -> None:
        Path(path).write_text(self.emit(module), encoding="utf-8")

    def _emit_operation(self, operation: Operation) -> list[str]:
        if operation.op == "aim.view":
            return [_comment("ALOI_PIM", operation)]
        if operation.op == "aim.load_gb":
            return [
                _comment("ALOI_PIM", operation),
                AimInstruction("WR_GB", (_opsize(operation), 0, self.channel_mask)).render(),
            ]
        if operation.op == "aim.pim.mac_dram_gb":
            row = int(_physical(operation)["row_range"][0]) % 16384
            size = _opsize(operation)
            return [
                _comment("ALOI_PIM", operation),
                RegisterWrite("CFR", 0, 1).render(),
                AimInstruction("WR_BIAS", (0, self.channel_mask)).render(),
                AimInstruction("MAC_ABK", (size, self.channel_mask, row)).render(),
            ]
        if operation.op == "aim.read_acc":
            return [
                _comment("ALOI_PIM", operation),
                AimInstruction("RD_MAC", (128, self.channel_mask)).render(),
            ]
        if operation.op == "aim.pim.ewmul":
            row = int(_physical(operation)["row_range"][0]) % 16384
            return [
                _comment("ALOI_PIM", operation),
                RegisterWrite("CFR", 1, 0).render(),
                AimInstruction("EWMUL", (_opsize(operation), self.channel_mask, row)).render(),
            ]
        if operation.op == "aim.pim.ewadd":
            return [
                _comment("ALOI_PIM", operation),
                AimInstruction("EWADD", (_opsize(operation), 0, 4)).render(),
            ]
        if operation.op.startswith("aim.pnm."):
            return [_comment("ALOI_PNM", operation)]
        if operation.op.startswith("aim.cxl."):
            return [_comment("ALOI_CXL", operation)]
        if operation.op == "aim.transfer":
            return [_comment("ALOI_TRANSFER", operation)]
        raise ValueError(f"no trace emission rule for {operation.op}")
