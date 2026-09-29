"""Validated textual AiM simulator instructions."""

from __future__ import annotations

from dataclasses import dataclass

_ARITY = {
    "WR_GB": 3,
    "WR_BIAS": 2,
    "MAC_ABK": 3,
    "RD_MAC": 2,
    "EWMUL": 3,
    "EWADD": 3,
    "SYNC": 0,
    "EOC": 0,
}


@dataclass(frozen=True)
class AimInstruction:
    opcode: str
    operands: tuple[int | str, ...] = ()

    def __post_init__(self) -> None:
        try:
            arity = _ARITY[self.opcode]
        except KeyError as exc:
            raise ValueError(f"unsupported AiM opcode: {self.opcode}") from exc
        if len(self.operands) != arity:
            raise ValueError(f"{self.opcode} expects {arity} operands, got {len(self.operands)}")

    def render(self) -> str:
        suffix = " " + " ".join(str(operand) for operand in self.operands) if self.operands else ""
        return f"AiM {self.opcode}{suffix}"


@dataclass(frozen=True)
class RegisterWrite:
    register: str
    index: int
    value: int

    def render(self) -> str:
        if self.register not in ("CFR", "GPR"):
            raise ValueError(f"unsupported register file: {self.register}")
        if self.register == "GPR":
            return f"W GPR {self.index}"
        return f"W CFR {self.index} {self.value}"
