"""Shared integration fixtures."""

from __future__ import annotations

from functools import cache

from aloi.device import DeviceSpec
from aloi.mapping import load_schedules
from aloi.pipeline import CompilationResult, compile_exported_program
from examples.llama2_70b.export import export_block
from examples.llama2_70b.parallel_plan import make_parallel_plan


@cache
def compilation(seq_len: int = 32, tp: int = 8) -> CompilationResult:
    return compile_exported_program(
        export_block(seq_len),
        make_parallel_plan(tp),
        DeviceSpec.load("configs/devices/cent.yaml"),
        load_schedules("examples/llama2_70b/schedule.py"),
    )
