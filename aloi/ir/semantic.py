"""Semantic-dialect definitions and verification."""

from __future__ import annotations

from aloi.ir.base import Module

SEMANTIC_OPS = frozenset(
    {
        "aloi.linear",
        "aloi.rms_norm",
        "aloi.rope",
        "aloi.kv_update",
        "aloi.matmul",
        "aloi.softmax",
        "aloi.silu",
        "aloi.add",
        "aloi.mul",
        "aloi.reshape",
        "aloi.transpose",
        "aloi.expand",
        "aloi.slice",
        "aloi.all_reduce",
        "aloi.broadcast",
    }
)

_FORBIDDEN_SEMANTIC_TERMS = ("pim", "pnm", "cxl", "channel", "bank", "dram", "row")


def verify_semantic_module(module: Module) -> None:
    module.verify()
    for function in module.functions:
        for operation in function.operations:
            if operation.op not in SEMANTIC_OPS:
                raise ValueError(f"non-semantic operation in Semantic IR: {operation.op}")
            for key in operation.attrs:
                lowered = key.lower()
                if any(term in lowered for term in _FORBIDDEN_SEMANTIC_TERMS):
                    raise ValueError(f"physical attribute {key!r} leaked into Semantic IR")
