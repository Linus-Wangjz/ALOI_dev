"""User-facing ownership plan."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from aloi.ir.base import Module, Pass
from aloi.ir.semantic import verify_semantic_module


@dataclass(frozen=True)
class ShardRule:
    target: str
    axis: str
    mesh_axis: str = "tp"


class ParallelPlan:
    """Ownership constraints; communication is intentionally not expressible here."""

    def __init__(self, tp: int = 1) -> None:
        if tp < 1:
            raise ValueError("tp must be positive")
        self.tp = tp
        self.rules: list[ShardRule] = []

    def shard(self, target: str, *, axis: str, mesh_axis: str = "tp") -> ParallelPlan:
        if not target or not axis:
            raise ValueError("shard target and axis must be non-empty")
        rule = ShardRule(target, axis, mesh_axis)
        self.rules = [existing for existing in self.rules if existing.target != target]
        self.rules.append(rule)
        return self

    def rule_for(self, target: str) -> ShardRule | None:
        return next((rule for rule in self.rules if rule.target == target), None)


class ApplyParallelPlan(Pass):
    """Seed sharding on explicitly owned semantic tensors."""

    def __init__(self, plan: ParallelPlan) -> None:
        self.plan = plan

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        if self.plan.tp == 1:
            result.attrs["tp"] = 1
            return result

        matched: set[str] = set()
        for function in result.functions:
            for operation in function.operations:
                name = operation.attrs.get("semantic_name")
                if not isinstance(name, str):
                    continue
                rule = self.plan.rule_for(name)
                if rule is None:
                    continue
                for value in operation.results:
                    if rule.axis in value.type.axes:
                        value.type = value.type.shard(rule.axis, rule.mesh_axis, self.plan.tp)
                        matched.add(rule.target)
        unmatched = sorted({rule.target for rule in self.plan.rules} - matched)
        if unmatched:
            raise ValueError(f"parallel-plan targets did not match shardable tensors: {unmatched}")
        result.attrs["tp"] = self.plan.tp
        result.attrs["stage"] = "parallel-seeded"
        verify_semantic_module(result)
        return result
