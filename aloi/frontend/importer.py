"""Import a torch.export ExportedProgram into ALOI Semantic IR."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import torch
from torch.fx import Node

from aloi.ir.base import Function, Module, Operation, Pass, Value
from aloi.ir.semantic import verify_semantic_module
from aloi.ir.types import TensorType, normalize_dtype

_TARGET_TO_SEMANTIC = {
    "aloi.linear.default": "aloi.linear",
    "aloi.rms_norm.default": "aloi.rms_norm",
    "aloi.rope.default": "aloi.rope",
    "aloi.kv_update.default": "aloi.kv_update",
    "aten.matmul.default": "aloi.matmul",
    "aten.mm.default": "aloi.matmul",
    "aten.bmm.default": "aloi.matmul",
    "aten.softmax.int": "aloi.softmax",
    "aten._softmax.default": "aloi.softmax",
    "aten.silu.default": "aloi.silu",
    "aten.add.Tensor": "aloi.add",
    "aten.mul.Tensor": "aloi.mul",
    "aten.reshape.default": "aloi.reshape",
    "aten.view.default": "aloi.reshape",
    "aten.unsqueeze.default": "aloi.reshape",
    "aten.transpose.int": "aloi.transpose",
    "aten.permute.default": "aloi.transpose",
    "aten.expand.default": "aloi.expand",
    "aten.slice.Tensor": "aloi.slice",
}

_PARAMETER_SEMANTICS = {
    "attention_norm": "attention.norm",
    "wq": "attention.q",
    "wk": "attention.k",
    "wv": "attention.v",
    "wo": "attention.output",
    "ffn_norm": "feed_forward.norm",
    "w1": "feed_forward.gate",
    "w3": "feed_forward.up",
    "w2": "feed_forward.down",
}


def _tensor_metadata(value: Any) -> tuple[tuple[int, ...], str]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor metadata, got {type(value).__name__}")
    return tuple(int(dim) for dim in value.shape), normalize_dtype(value.dtype)


def _infer_axes(shape: tuple[int, ...], semantic_name: str = "") -> tuple[str, ...]:
    if not shape:
        return ()
    if len(shape) == 1:
        return ("hidden",)
    if len(shape) == 2:
        return ("out_feature", "in_feature")
    if len(shape) == 3:
        last = shape[-1]
        if last == 8192:
            feature = "hidden"
        elif last == 1024:
            feature = "kv_feature"
        elif last == 28672:
            feature = "intermediate"
        else:
            feature = "feature"
        return ("batch", "token", feature)
    if len(shape) == 4:
        if shape[1] == 8 and shape[-1] == 128:
            return ("batch", "kv_head", "sequence", "head_dim")
        if shape[1] == 1 and shape[2] in (8, 64) and shape[-1] == 128:
            head = "q_head" if shape[2] == 64 else "kv_head"
            return ("batch", "token", head, "head_dim")
        if shape[1] == 64 and shape[-1] == 128:
            middle = "token" if shape[2] == 1 and "context" in semantic_name else "sequence"
            return ("batch", "q_head", middle, "head_dim")
        if shape[1] == 64:
            return ("batch", "q_head", "token", "sequence")
    if len(shape) == 5 and shape[1] == 8:
        return ("batch", "kv_head", "group", "sequence", "head_dim")
    return tuple(f"dim{index}" for index in range(len(shape)))


def _placeholder_metadata(
    name: str,
    parameter: str | None,
    shape: tuple[int, ...],
) -> tuple[str, tuple[str, ...], bool]:
    if parameter is not None:
        axes = ("hidden",) if len(shape) == 1 else ("out_feature", "in_feature")
        return "weight", axes, False
    if name in ("k_cache", "v_cache"):
        return "kv_cache", ("batch", "kv_head", "sequence", "head_dim"), True
    if name == "position":
        return "index", (), False
    if name == "x":
        return "activation", ("batch", "token", "hidden"), False
    return "activation", _infer_axes(shape), False


def _target_name(target: Any) -> str:
    text = str(target)
    return text.removeprefix("torch.ops.")


def _literal(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return normalize_dtype(value)
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, (list, tuple)):
        return [_literal(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ImportExportedProgram:
    """Translate the stable torch-export operator subset into Semantic IR."""

    def run(self, exported_program: Any) -> Module:
        graph = exported_program.graph_module.graph
        signature = exported_program.graph_signature
        parameters = dict(getattr(signature, "inputs_to_parameters", {}))
        function = Function("main")
        node_values: dict[Node, Value | list[Value]] = {}
        value_semantics: dict[Value, str] = {}
        parameter_by_value: dict[Value, str] = {}
        matmul_count = 0
        add_count = 0

        for node in graph.nodes:
            if node.op == "placeholder":
                shape, dtype = _tensor_metadata(node.meta["val"])
                parameter = parameters.get(node.name)
                role, axes, persistent = _placeholder_metadata(node.name, parameter, shape)
                value = Value(
                    node.name,
                    TensorType(shape, dtype, role=role, axes=axes, persistent=persistent),
                )
                function.inputs.append(value)
                node_values[node] = value
                if parameter is not None:
                    parameter_by_value[value] = parameter
                    value_semantics[value] = _PARAMETER_SEMANTICS.get(parameter, parameter)
                elif node.name in ("k_cache", "v_cache"):
                    value_semantics[value] = f"attention.{node.name}"
                elif node.name == "x":
                    value_semantics[value] = "block.input"
                continue

            if node.op == "output":
                for output_node in self._nodes(node.args[0]):
                    mapped = node_values[output_node]
                    if isinstance(mapped, list):
                        function.outputs.extend(mapped)
                    else:
                        function.outputs.append(mapped)
                continue

            if node.op != "call_function":
                raise ValueError(f"unsupported torch.export node kind: {node.op}")

            if node.target is operator.getitem:
                source_node, index = node.args
                values = node_values[source_node]
                if not isinstance(values, list):
                    raise ValueError("getitem source is not a multi-result operation")
                node_values[node] = values[int(index)]
                continue

            target = _target_name(node.target)
            try:
                semantic_op = _TARGET_TO_SEMANTIC[target]
            except KeyError as exc:
                raise ValueError(f"unsupported exported operator: {target}") from exc

            operands: list[Value] = []
            for arg_node in self._nodes(node.args):
                mapped = node_values[arg_node]
                if isinstance(mapped, list):
                    operands.extend(mapped)
                else:
                    operands.append(mapped)

            attrs: dict[str, Any] = {}
            literals = [_literal(arg) for arg in node.args if not isinstance(arg, Node)]
            if literals:
                attrs["arguments"] = literals
            if node.kwargs:
                attrs["keyword_arguments"] = {
                    str(key): _literal(value) for key, value in node.kwargs.items()
                }

            semantic_name = value_semantics.get(operands[0], "") if operands else ""
            if semantic_op == "aloi.linear":
                weight = parameter_by_value.get(operands[1])
                if weight is None:
                    raise ValueError("linear weight is not an exported parameter")
                attrs["weight"] = weight
                semantic_name = _PARAMETER_SEMANTICS[weight]
            elif semantic_op == "aloi.rms_norm":
                weight = parameter_by_value.get(operands[1], "")
                semantic_name = _PARAMETER_SEMANTICS.get(weight or "", semantic_name)
            elif semantic_op == "aloi.rope":
                semantic_name = "attention.rope"
            elif semantic_op == "aloi.kv_update":
                semantic_name = value_semantics.get(operands[0], "attention.kv_cache")
            elif semantic_op == "aloi.matmul":
                semantic_name = "attention.scores" if matmul_count == 0 else "attention.context"
                matmul_count += 1
            elif semantic_op == "aloi.softmax":
                semantic_name = "attention.probabilities"
            elif semantic_op == "aloi.silu":
                semantic_name = "feed_forward.gate_activation"
            elif semantic_op == "aloi.mul" and operands and operands[0].type.rank == 3:
                semantic_name = "feed_forward.fused"
            elif semantic_op == "aloi.add":
                semantic_name = "attention.residual" if add_count == 0 else "block.output"
                add_count += 1
            if semantic_name:
                attrs["semantic_name"] = semantic_name

            metadata = node.meta.get("val")
            result_metadata = list(metadata) if isinstance(metadata, (tuple, list)) else [metadata]
            result_types: list[TensorType] = []
            for index, item in enumerate(result_metadata):
                shape, dtype = _tensor_metadata(item)
                item_name = semantic_name
                if semantic_op == "aloi.rope":
                    item_name = "attention.q" if index == 0 else "attention.k"
                persistent = semantic_op == "aloi.kv_update"
                role = "kv_cache" if persistent else "activation"
                result_types.append(
                    TensorType(
                        shape,
                        dtype,
                        role=role,
                        axes=_infer_axes(shape, item_name),
                        persistent=persistent,
                    )
                )

            result_names = (
                [node.name]
                if len(result_types) == 1
                else [f"{node.name}_{index}" for index in range(len(result_types))]
            )
            operation = function.add_op(
                Operation(
                    semantic_op,
                    operands,
                    result_types,
                    attrs=attrs,
                    result_names=result_names,
                )
            )
            mapped_results = operation.results
            node_values[node] = mapped_results[0] if len(mapped_results) == 1 else mapped_results
            for index, result in enumerate(mapped_results):
                if semantic_op == "aloi.rope":
                    value_semantics[result] = "attention.q" if index == 0 else "attention.k"
                elif semantic_name:
                    value_semantics[result] = semantic_name

        module = Module([function], attrs={"stage": "semantic"})
        return AnnotateRolesAndAxes().run(SemanticCanonicalize().run(module))

    @staticmethod
    def _nodes(value: Any) -> Iterable[Node]:
        if isinstance(value, Node):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from ImportExportedProgram._nodes(item)


class SemanticCanonicalize(Pass):
    """Validate that import produced only the canonical semantic op set."""

    def run(self, module: Module) -> Module:
        verify_semantic_module(module)
        return module


class AnnotateRolesAndAxes(Pass):
    """Require semantic metadata on every tensor before parallelization."""

    def run(self, module: Module) -> Module:
        for function in module.functions:
            for value in [*function.inputs, *(r for op in function.operations for r in op.results)]:
                if not value.type.role:
                    raise ValueError(f"%{value.name} has no semantic role")
                if value.type.rank and len(value.type.axes) != value.type.rank:
                    raise ValueError(f"%{value.name} has incomplete axis semantics")
        verify_semantic_module(module)
        return module
