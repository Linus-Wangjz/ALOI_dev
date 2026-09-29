"""Semantic sharding propagation and partitioned-reduction analysis."""

from __future__ import annotations

from aloi.ir.base import Module, Pass, Value
from aloi.ir.semantic import verify_semantic_module
from aloi.ir.types import ShardingSpec

_LAYOUT_OR_ELEMENTWISE = {
    "aloi.reshape",
    "aloi.transpose",
    "aloi.expand",
    "aloi.slice",
    "aloi.silu",
    "aloi.softmax",
}

_AXIS_PRIORITY = (
    "q_head",
    "kv_head",
    "intermediate",
    "hidden",
    "kv_feature",
    "feature",
    "out_feature",
    "in_feature",
)


def _axis_for(value: Value, spec: ShardingSpec) -> str | None:
    if spec.axis in value.type.axes:
        return spec.axis
    global_shape = value.type.global_shape or value.type.shape
    for axis in _AXIS_PRIORITY:
        if axis in value.type.axes:
            size = global_shape[value.type.axes.index(axis)]
            if size % spec.parts == 0:
                return axis
    return None


def _propagate(source: Value, target: Value) -> bool:
    spec = source.type.sharding
    if spec is None or target.type.sharding is not None:
        return False
    axis = _axis_for(target, spec)
    if axis is None:
        return False
    target.type = target.type.shard(axis, spec.mesh_axis, spec.parts)
    return True


def _connect(values: list[Value]) -> bool:
    """Propagate the first sharding found to every compatible value."""

    source = next((value for value in values if value.type.sharding is not None), None)
    if source is None:
        return False
    return any(_propagate(source, value) for value in values if value is not source)


class PropagateSharding(Pass):
    """Propagate ownership and flag reductions whose inputs are partitioned."""

    def run(self, module: Module) -> Module:
        for function in module.functions:
            changed = True
            while changed:
                changed = False
                # Reverse order carries seeds through reshapes back to projections.
                for operation in reversed(function.operations):
                    if operation.op in _LAYOUT_OR_ELEMENTWISE and operation.results:
                        changed |= _connect([operation.operands[0], operation.results[0]])
                    elif operation.op == "aloi.rope":
                        changed |= _connect([operation.operands[0], operation.results[0]])
                        changed |= _connect([operation.operands[1], operation.results[1]])
                    elif operation.op == "aloi.kv_update":
                        changed |= _connect(
                            [operation.operands[0], operation.operands[1], operation.results[0]]
                        )
                    elif operation.op == "aloi.matmul":
                        changed |= _connect([*operation.operands, operation.results[0]])
                    elif operation.op in ("aloi.mul", "aloi.add") and operation.results:
                        changed |= _connect([*operation.operands, operation.results[0]])
                    elif operation.op == "aloi.linear":
                        activation, weight = operation.operands[:2]
                        output = operation.results[0]
                        if output.type.sharding is not None:
                            spec = output.type.sharding
                            if weight.type.sharding is None:
                                weight.type = weight.type.shard(
                                    "out_feature", spec.mesh_axis, spec.parts
                                )
                                changed = True
                        elif activation.type.sharding is not None:
                            spec = activation.type.sharding
                            if weight.type.sharding is None:
                                weight.type = weight.type.shard(
                                    "in_feature", spec.mesh_axis, spec.parts
                                )
                                changed = True
                            operation.attrs["partitioned_reduction"] = True

            # A projection is a reduction only when its input was partitioned
            # independently of its output ownership.
            for operation in function.operations:
                if operation.op != "aloi.linear" or not operation.results:
                    continue
                activation = operation.operands[0]
                output = operation.results[0]
                if activation.type.sharding is not None and output.type.sharding is None:
                    operation.attrs["partitioned_reduction"] = True

        module.attrs["stage"] = "sharding-propagated"
        verify_semantic_module(module)
        return module
