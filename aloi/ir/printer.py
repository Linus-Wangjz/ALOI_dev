"""Deterministic textual printer for ALOI IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aloi.ir.base import Function, Module, Operation, Value
from aloi.ir.types import TensorType


def _format_attr(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, dict):
        body = ", ".join(f"{key}: {_format_attr(value[key])}" for key in sorted(value))
        return "{" + body + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_format_attr(item) for item in value) + "]"
    return str(value)


class Printer:
    """Print the stable, human-readable `.aloi` representation."""

    def print_type(self, typ: TensorType) -> str:
        shape = "x".join(str(dim) for dim in typ.shape) or "scalar"
        fields = [f"role={typ.role}"]
        if typ.axes:
            fields.append("axes=[" + ", ".join(typ.axes) + "]")
        if typ.persistent:
            fields.append("persistent=true")
        if typ.sharding is not None:
            fields.append(
                f"sharding={typ.sharding.axis}@{typ.sharding.mesh_axis}/{typ.sharding.parts}"
            )
        if typ.global_shape is not None:
            fields.append("global_shape=[" + ", ".join(map(str, typ.global_shape)) + "]")
        return f"tensor<{shape}x{typ.dtype}, " + ", ".join(fields) + ">"

    def print_value(self, value: Value) -> str:
        return f"%{value.name}: {self.print_type(value.type)}"

    def print_operation(self, operation: Operation) -> str:
        results = ", ".join(f"%{result.name}" for result in operation.results)
        assignment = f"{results} = " if results else ""
        operands = ", ".join(f"%{operand.name}" for operand in operation.operands)
        attrs = ""
        if operation.attrs:
            attrs = (
                " {"
                + ", ".join(
                    f"{key}={_format_attr(operation.attrs[key])}" for key in sorted(operation.attrs)
                )
                + "}"
            )
        result_types = ", ".join(self.print_type(result.type) for result in operation.results)
        type_suffix = f" : {result_types}" if result_types else ""
        return f"{assignment}{operation.op} {operands}{attrs}{type_suffix}".rstrip()

    def print_function(self, function: Function) -> str:
        inputs = ", ".join(self.print_value(value) for value in function.inputs)
        outputs = ", ".join(self.print_type(value.type) for value in function.outputs)
        header = f"func @{function.name}({inputs})"
        if outputs:
            header += f" -> ({outputs})"
        lines = [header + " {"]
        lines.extend(f"  {self.print_operation(operation)}" for operation in function.operations)
        returned = ", ".join(f"%{value.name}" for value in function.outputs)
        lines.append(f"  return {returned}".rstrip())
        lines.append("}")
        return "\n".join(lines)

    def print_module(self, module: Module) -> str:
        attrs = ""
        if module.attrs:
            attrs = (
                " attributes {"
                + ", ".join(
                    f"{key}={_format_attr(module.attrs[key])}" for key in sorted(module.attrs)
                )
                + "}"
            )
        functions = "\n\n".join(self.print_function(function) for function in module.functions)
        body = "\n".join(f"  {line}" if line else line for line in functions.splitlines())
        return f"module{attrs} {{\n{body}\n}}\n"

    def write(self, module: Module, path: str | Path) -> None:
        Path(path).write_text(self.print_module(module), encoding="utf-8")
