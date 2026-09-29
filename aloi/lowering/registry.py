"""Lowering recipe registry and Semantic-to-Device conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from aloi.device.spec import DeviceSpec
from aloi.ir.base import Function, Module, Operation, Pass, Value
from aloi.ir.device import verify_device_module


@dataclass(frozen=True)
class LoweringResult:
    operations: tuple[Operation, ...]
    outputs: tuple[Value, ...]


class LoweringRecipe(ABC):
    """One legal implementation candidate for a semantic operation."""

    name: str
    semantic_op: str
    required_capabilities: tuple[str, ...]

    def match(self, operation: Operation) -> bool:
        return operation.op == self.semantic_op

    def is_legal(self, operation: Operation, device_spec: DeviceSpec) -> bool:
        dtype = operation.results[0].type.dtype if operation.results else None
        return all(
            device_spec.supports(capability, dtype) for capability in self.required_capabilities
        )

    @abstractmethod
    def lower(self, operation: Operation, operands: list[Value]) -> LoweringResult:
        """Produce Device IR using already-remapped operands."""

    def base_attrs(self, operation: Operation) -> dict[str, object]:
        attrs = {
            key: value
            for key, value in operation.attrs.items()
            if key not in ("legal_recipes", "selected_recipe")
        }
        attrs["source_op"] = operation.op
        attrs["recipe"] = self.name
        return attrs


class LoweringRecipeRegistry:
    def __init__(self) -> None:
        self._recipes: list[LoweringRecipe] = []

    def register(self, recipe: LoweringRecipe) -> LoweringRecipe:
        if any(existing.name == recipe.name for existing in self._recipes):
            raise ValueError(f"duplicate lowering recipe: {recipe.name}")
        self._recipes.append(recipe)
        return recipe

    def candidates(self, operation: Operation) -> list[LoweringRecipe]:
        return [recipe for recipe in self._recipes if recipe.match(operation)]

    def by_name(self, name: str) -> LoweringRecipe:
        for recipe in self._recipes:
            if recipe.name == name:
                return recipe
        raise KeyError(f"lowering recipe not registered: {name}")


class RecipeSelector(Protocol):
    def select(
        self,
        operation: Operation,
        candidates: list[LoweringRecipe],
        device_spec: DeviceSpec,
    ) -> LoweringRecipe: ...


class EnumerateLegalRecipes(Pass):
    def __init__(self, registry: LoweringRecipeRegistry, device_spec: DeviceSpec) -> None:
        self.registry = registry
        self.device_spec = device_spec

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        for function in result.functions:
            for operation in function.operations:
                legal = [
                    recipe.name
                    for recipe in self.registry.candidates(operation)
                    if recipe.is_legal(operation, self.device_spec)
                ]
                operation.attrs["legal_recipes"] = legal
        return result


class LowerSemanticToDevice(Pass):
    def __init__(
        self,
        registry: LoweringRecipeRegistry,
        device_spec: DeviceSpec,
        selector: RecipeSelector | None = None,
    ) -> None:
        if selector is None:
            from aloi.lowering.selector import FirstLegalSelector

            selector = FirstLegalSelector()
        self.registry = registry
        self.device_spec = device_spec
        self.selector = selector

    def run(self, module: Module) -> Module:
        functions: list[Function] = []
        for function in module.functions:
            value_map: dict[Value, Value] = {}
            inputs = [Value(value.name, value.type) for value in function.inputs]
            value_map.update(zip(function.inputs, inputs, strict=True))
            lowered_function = Function(function.name, inputs=inputs, attrs=dict(function.attrs))

            for operation in function.operations:
                operands = [value_map[value] for value in operation.operands]
                selected_name = operation.attrs.get("selected_recipe")
                if isinstance(selected_name, str):
                    recipe = self.registry.by_name(selected_name)
                else:
                    candidates = self.registry.candidates(operation)
                    recipe = self.selector.select(operation, candidates, self.device_spec)
                result = recipe.lower(operation, operands)
                if not result.operations:
                    raise ValueError(f"recipe {recipe.name} produced no operations")
                for lowered_operation in result.operations:
                    lowered_function.add_op(lowered_operation)
                if len(result.outputs) != len(operation.results):
                    raise ValueError(f"recipe {recipe.name} returned the wrong result count")
                value_map.update(zip(operation.results, result.outputs, strict=True))

            lowered_function.outputs = [value_map[value] for value in function.outputs]
            functions.append(lowered_function)

        result_module = Module(functions, attrs=dict(module.attrs))
        result_module.attrs["stage"] = "device"
        result_module.attrs["device_spec"] = self.device_spec.source
        verify_device_module(result_module)
        return result_module


def default_registry() -> LoweringRecipeRegistry:
    from aloi.lowering.recipes.attention import attention_recipes
    from aloi.lowering.recipes.elementwise import elementwise_recipes
    from aloi.lowering.recipes.linear import LinearRecipe
    from aloi.lowering.recipes.rmsnorm import RMSNormRecipe

    registry = LoweringRecipeRegistry()
    for recipe in [
        LinearRecipe(),
        RMSNormRecipe(),
        *attention_recipes(),
        *elementwise_recipes(),
    ]:
        registry.register(recipe)
    return registry
