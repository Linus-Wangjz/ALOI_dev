"""Lower allocated Device IR into higher-level AIM SSA/dataflow IR."""

from __future__ import annotations

from aloi.ir.aim import verify_aim_module
from aloi.ir.base import Function, Module, Operation, Pass, Value


def _attrs(operation: Operation) -> dict[str, object]:
    attrs = dict(operation.attrs)
    attrs["source_device_op"] = operation.op
    return attrs


class LowerDeviceToAIM(Pass):
    def run(self, module: Module) -> Module:
        functions: list[Function] = []
        for function in module.functions:
            inputs = [Value(value.name, value.type) for value in function.inputs]
            value_map: dict[Value, Value] = dict(zip(function.inputs, inputs, strict=True))
            aim_function = Function(function.name, inputs=inputs, attrs=dict(function.attrs))

            for operation in function.operations:
                if "physical" not in operation.attrs:
                    raise ValueError(f"Device operation {operation.op} has no physical allocation")
                operands = [value_map[value] for value in operation.operands]
                if operation.op == "pim.gemv":
                    output = self._lower_gemv(operation, operands)
                    for aim_operation in output[0]:
                        aim_function.add_op(aim_operation)
                    value_map[operation.results[0]] = output[1]
                    continue

                if operation.op.startswith("pim."):
                    aim_name = f"aim.pim.{operation.op.split('.', 1)[1]}"
                elif operation.op.startswith("pnm."):
                    aim_name = f"aim.pnm.{operation.op.split('.', 1)[1]}"
                elif operation.op.startswith("cxl."):
                    aim_name = f"aim.cxl.{operation.op.split('.', 1)[1]}"
                elif operation.op == "device.transfer":
                    aim_name = "aim.transfer"
                else:
                    raise ValueError(f"unsupported Device IR operation: {operation.op}")

                lowered = Operation(
                    aim_name,
                    operands,
                    [value.type for value in operation.results],
                    attrs=_attrs(operation),
                    result_names=[value.name for value in operation.results],
                )
                aim_function.add_op(lowered)
                value_map.update(zip(operation.results, lowered.results, strict=True))

            aim_function.outputs = [value_map[value] for value in function.outputs]
            functions.append(aim_function)

        aim_module = Module(functions, attrs=dict(module.attrs))
        aim_module.attrs["stage"] = "aim"
        verify_aim_module(aim_module)
        return aim_module

    @staticmethod
    def _lower_gemv(operation: Operation, operands: list[Value]) -> tuple[list[Operation], Value]:
        attrs = _attrs(operation)
        activation, weight = operands[:2]
        weight_view = Operation(
            "aim.view",
            [weight],
            [weight.type],
            attrs={**attrs, "view": "physical_weight_tile"},
            result_names=[f"{operation.results[0].name}_weight_tile"],
        )
        load = Operation(
            "aim.load_gb",
            [activation],
            [activation.type],
            attrs=attrs,
            result_names=[f"{operation.results[0].name}_gb"],
        )
        mac = Operation(
            "aim.pim.mac_dram_gb",
            [weight_view.results[0], load.results[0]],
            [operation.results[0].type],
            attrs=attrs,
            result_names=[f"{operation.results[0].name}_acc"],
        )
        read = Operation(
            "aim.read_acc",
            mac.results,
            [operation.results[0].type],
            attrs=attrs,
            result_names=[operation.results[0].name],
        )
        return [weight_view, load, mac, read], read.results[0]
