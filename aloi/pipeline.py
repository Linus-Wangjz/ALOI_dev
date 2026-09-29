"""The fixed ALOI v0.1 compiler pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from aloi.backend.aim import AimTraceEmitter, LowerDeviceToAIM
from aloi.device import DeviceSpec
from aloi.frontend import ImportExportedProgram
from aloi.frontend.torch_export import ExportBundle
from aloi.ir.base import Module
from aloi.lowering import (
    EnumerateLegalRecipes,
    LowerSemanticToDevice,
    SelectFirstLegal,
    default_registry,
)
from aloi.mapping import (
    ApplyScheduleConstraints,
    DefaultMapper,
    MappingPlan,
    PhysicalAllocator,
    Schedule,
)
from aloi.parallel import ApplyParallelPlan, InsertCollectives, ParallelPlan, PropagateSharding


@dataclass(frozen=True)
class CompilationResult:
    exported_program_text: str
    semantic: Module
    sharded: Module
    device: Module
    aim: Module
    trace: str


def compile_exported_program(
    bundle: ExportBundle,
    parallel_plan: ParallelPlan,
    device_spec: DeviceSpec,
    schedules: list[Schedule] | None = None,
) -> CompilationResult:
    semantic = ImportExportedProgram().run(bundle.program)
    seeded = ApplyParallelPlan(parallel_plan).run(semantic)
    propagated = PropagateSharding().run(seeded)
    sharded = InsertCollectives().run(propagated)

    registry = default_registry()
    enumerated = EnumerateLegalRecipes(registry, device_spec).run(sharded)
    selected = SelectFirstLegal(registry, device_spec).run(enumerated)
    device = LowerSemanticToDevice(registry, device_spec).run(selected)

    mapping_plan = MappingPlan.from_schedules(schedules or [])
    scheduled = ApplyScheduleConstraints(mapping_plan).run(device)
    mapped = DefaultMapper(device_spec).run(scheduled)
    allocated = PhysicalAllocator(device_spec).run(mapped)
    aim = LowerDeviceToAIM().run(allocated)
    trace = AimTraceEmitter().emit(aim)
    return CompilationResult(bundle.text(), semantic, sharded, device, aim, trace)
