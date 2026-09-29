"""Derive collectives from partitioned semantic reductions."""

from __future__ import annotations

from aloi.ir.base import Module, Operation, Pass, Value
from aloi.ir.semantic import verify_semantic_module


class InsertCollectives(Pass):
    """Insert all-reduce after every partitioned linear reduction."""

    def run(self, module: Module) -> Module:
        inserted = 0
        for function in module.functions:
            replacements: dict[Value, Value] = {}
            operations: list[Operation] = []
            for operation in function.operations:
                operation.operands = [
                    replacements.get(value, value) for value in operation.operands
                ]
                operations.append(operation)
                if not operation.attrs.get("partitioned_reduction"):
                    continue
                if len(operation.results) != 1:
                    raise ValueError("partitioned reduction must have exactly one result")
                partial = operation.results[0]
                collective = Operation(
                    "aloi.all_reduce",
                    [partial],
                    [partial.type],
                    attrs={
                        "mesh_axis": "tp",
                        "reduction": "sum",
                        "semantic_name": operation.attrs.get("semantic_name", ""),
                    },
                    result_names=[f"{partial.name}_all_reduced"],
                )
                operations.append(collective)
                replacements[partial] = collective.results[0]
                inserted += 1
            function.operations = operations
            function.outputs = [replacements.get(value, value) for value in function.outputs]
        module.attrs["collectives_inserted"] = inserted
        module.attrs["stage"] = "sharded"
        verify_semantic_module(module)
        return module
