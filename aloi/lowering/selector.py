"""Deterministic capability-only recipe selection."""

from __future__ import annotations

from copy import deepcopy

from aloi.device.spec import DeviceSpec
from aloi.ir.base import Module, Operation, Pass
from aloi.lowering.registry import LoweringRecipe, LoweringRecipeRegistry


class FirstLegalSelector:
    def select(
        self,
        operation: Operation,
        candidates: list[LoweringRecipe],
        device_spec: DeviceSpec,
    ) -> LoweringRecipe:
        for recipe in candidates:
            if recipe.is_legal(operation, device_spec):
                return recipe
        names = [recipe.name for recipe in candidates]
        raise ValueError(
            f"no legal lowering recipe for {operation.op}; candidates={names}, "
            f"device={device_spec.source}"
        )


class SelectFirstLegal(Pass):
    def __init__(
        self,
        registry: LoweringRecipeRegistry,
        device_spec: DeviceSpec,
        selector: FirstLegalSelector | None = None,
    ) -> None:
        self.registry = registry
        self.device_spec = device_spec
        self.selector = selector or FirstLegalSelector()

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        for function in result.functions:
            for operation in function.operations:
                candidates = self.registry.candidates(operation)
                selected = self.selector.select(operation, candidates, self.device_spec)
                operation.attrs["selected_recipe"] = selected.name
        return result
