"""Small, dependency-free SSA/dataflow IR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aloi.ir.types import TensorType


@dataclass(eq=False)
class Value:
    """A typed SSA value."""

    name: str
    type: TensorType
    producer: Operation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name.removeprefix("%")
        if not self.name:
            raise ValueError("SSA value name must be non-empty")


class Operation:
    """An operation with ordered operands, results, and serializable attributes."""

    def __init__(
        self,
        op: str,
        operands: Iterable[Value] = (),
        result_types: Iterable[TensorType] = (),
        *,
        attrs: dict[str, Any] | None = None,
        result_names: Sequence[str] | None = None,
    ) -> None:
        if "." not in op:
            raise ValueError(f"operation must have a dialect prefix: {op!r}")
        self.op = op
        self.operands = list(operands)
        self.attrs = dict(attrs or {})
        types = list(result_types)
        names = list(result_names or (f"result{i}" for i in range(len(types))))
        if len(names) != len(types):
            raise ValueError("result name/type counts differ")
        self.results = [Value(name, typ, self) for name, typ in zip(names, types, strict=True)]

    @property
    def dialect(self) -> str:
        return self.op.split(".", 1)[0]

    def clone_with(
        self,
        *,
        operands: Iterable[Value] | None = None,
        result_types: Iterable[TensorType] | None = None,
        attrs: dict[str, Any] | None = None,
        result_names: Sequence[str] | None = None,
    ) -> Operation:
        return Operation(
            self.op,
            self.operands if operands is None else operands,
            [value.type for value in self.results] if result_types is None else result_types,
            attrs=self.attrs if attrs is None else attrs,
            result_names=[value.name for value in self.results]
            if result_names is None
            else result_names,
        )


@dataclass
class Function:
    """One straight-line SSA function."""

    name: str
    inputs: list[Value] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def add_op(self, operation: Operation) -> Operation:
        used_names = {value.name for value in self.inputs}
        used_names.update(value.name for op in self.operations for value in op.results)
        for result in operation.results:
            base = result.name
            suffix = 0
            while result.name in used_names:
                suffix += 1
                result.name = f"{base}_{suffix}"
            used_names.add(result.name)
        self.operations.append(operation)
        return operation

    def replace_uses(self, old: Value, new: Value, *, after: int = -1) -> None:
        for operation in self.operations[after + 1 :]:
            operation.operands = [
                new if operand is old else operand for operand in operation.operands
            ]
        self.outputs = [new if value is old else value for value in self.outputs]

    def verify(self) -> None:
        available = set(self.inputs)
        names = {value.name for value in self.inputs}
        for operation in self.operations:
            missing = [operand.name for operand in operation.operands if operand not in available]
            if missing:
                raise ValueError(f"{operation.op} uses unavailable values: {missing}")
            for result in operation.results:
                if result.name in names:
                    raise ValueError(f"duplicate SSA value %{result.name}")
                available.add(result)
                names.add(result.name)
        missing_outputs = [value.name for value in self.outputs if value not in available]
        if missing_outputs:
            raise ValueError(f"function returns unavailable values: {missing_outputs}")


@dataclass
class Module:
    """A named collection of functions."""

    functions: list[Function] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def get_function(self, name: str = "main") -> Function:
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"function not found: {name}")

    def verify(self) -> None:
        if len({function.name for function in self.functions}) != len(self.functions):
            raise ValueError("duplicate function name")
        for function in self.functions:
            function.verify()


class Pass(ABC):
    """Base class for module transformations."""

    @abstractmethod
    def run(self, module: Module) -> Module:
        """Transform and return ``module``."""

    def __call__(self, module: Module) -> Module:
        return self.run(module)
