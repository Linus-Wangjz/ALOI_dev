"""Deterministic completion of partial mapping constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from aloi.device.spec import DeviceSpec
from aloi.ir.base import Module, Pass
from aloi.ir.device import verify_device_module


class DefaultMapper(Pass):
    def __init__(self, device_spec: DeviceSpec) -> None:
        self.device_spec = device_spec

    def run(self, module: Module) -> Module:
        result = deepcopy(module)
        vector_width = int(self.device_spec.parameter("pim.gemv.vector_width", 16))
        for function in result.functions:
            for operation in function.operations:
                mapping: dict[str, Any] = {
                    "split": {},
                    "tile": {},
                    "place": {},
                    "replicate": {},
                    "shard": {},
                    "bind": {},
                    "reduce_at": None,
                    "pipeline": [],
                }
                for directive in operation.attrs.get("schedule_constraints", []):
                    kind = directive["kind"]
                    if kind in ("split", "tile", "replicate"):
                        mapping[kind][directive["axis"]] = directive["factor"]
                    elif kind == "place":
                        mapping[kind][directive["value"]] = directive["device"]
                    elif kind == "shard":
                        mapping[kind][directive["axis"]] = directive["mesh_axis"]
                    elif kind == "bind":
                        mapping[kind][directive["axis"]] = directive["resource"]
                    elif kind == "reduce_at":
                        mapping[kind] = directive["axis"]
                    elif kind == "pipeline":
                        mapping[kind] = directive["stages"]

                if operation.op == "pim.gemv":
                    mapping["tile"].setdefault("in_feature", vector_width * 16)
                    mapping["bind"].setdefault("out_feature", "channel")
                    mapping["place"].setdefault("weight", "pim")
                    mapping["place"].setdefault("activation", "gb")
                    mapping["reduce_at"] = mapping["reduce_at"] or "out_feature"
                elif operation.op.startswith("pim."):
                    mapping["place"].setdefault("activation", "pim")
                elif operation.op.startswith("pnm."):
                    mapping["place"].setdefault("activation", "pnm")
                elif operation.op.startswith("cxl."):
                    mapping["place"].setdefault("activation", "cxl")
                else:
                    mapping["place"].setdefault("activation", "transfer")
                operation.attrs["mapping"] = mapping
        result.attrs["stage"] = "mapped"
        verify_device_module(result)
        return result
