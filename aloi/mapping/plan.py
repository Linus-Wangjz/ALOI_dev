"""Schedule constraint application."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from aloi.ir.base import Module, Operation, Pass
from aloi.ir.device import verify_device_module
from aloi.mapping.schedule import Directive, Schedule


@dataclass
class MappingPlan:
    constraints: dict[str, list[Directive]] = field(default_factory=dict)

    @classmethod
    def from_schedules(cls, schedules: list[Schedule]) -> MappingPlan:
        plan = cls()
        for item in schedules:
            plan.constraints.setdefault(item.target, []).extend(item.directives)
        return plan


def _targets(operation: Operation) -> set[str]:
    targets = {str(operation.attrs.get("semantic_name", ""))}
    weight = operation.attrs.get("weight")
    if isinstance(weight, str):
        prefix = "attention" if weight in {"wq", "wk", "wv", "wo"} else "feed_forward"
        targets.add(f"{prefix}.{weight}")
    return targets


class ApplyScheduleConstraints(Pass):
    def __init__(self, plan: MappingPlan) -> None:
        self.plan = plan

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        matched: set[str] = set()
        for function in result.functions:
            for operation in function.operations:
                directives: list[Directive] = []
                for target in _targets(operation):
                    if target in self.plan.constraints:
                        matched.add(target)
                        directives.extend(self.plan.constraints[target])
                if directives:
                    operation.attrs["schedule_constraints"] = [
                        {"kind": directive.kind, **directive.parameters} for directive in directives
                    ]
        unmatched = sorted(set(self.plan.constraints) - matched)
        if unmatched:
            raise ValueError(f"schedule targets did not match Device IR: {unmatched}")
        result.attrs["stage"] = "scheduled"
        verify_device_module(result)
        return result
