"""Compile the metadata-only Llama2-70B example end to end."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

from aloi.device import DeviceSpec
from aloi.frontend.torch_export import ExportBundle
from aloi.ir.printer import Printer
from aloi.mapping import load_schedules
from aloi.parallel import ParallelPlan
from aloi.pipeline import compile_exported_program


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="example model directory")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_example(model_dir: Path, seq_len: int, tp: int) -> tuple[ExportBundle, ParallelPlan]:
    required = [model_dir / "export.py", model_dir / "parallel_plan.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model directory is missing required files: {missing}")
    project_root = model_dir.resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    export_namespace: dict[str, Any] = runpy.run_path(str(model_dir / "export.py"))
    plan_namespace: dict[str, Any] = runpy.run_path(str(model_dir / "parallel_plan.py"))
    bundle = export_namespace["export_block"](seq_len)
    plan = plan_namespace["make_parallel_plan"](tp)
    if not isinstance(bundle, ExportBundle) or not isinstance(plan, ParallelPlan):
        raise TypeError("example export/parallel-plan hooks returned unexpected values")
    return bundle, plan


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seq_len < 1 or args.tp < 1:
        raise SystemExit("--seq-len and --tp must be positive")
    bundle, parallel_plan = _load_example(args.model, args.seq_len, args.tp)
    device_spec = DeviceSpec.load(args.device)
    schedules = load_schedules(args.schedule) if args.schedule else []
    result = compile_exported_program(bundle, parallel_plan, device_spec, schedules)

    args.output.mkdir(parents=True, exist_ok=True)
    printer = Printer()
    if args.dump_ir:
        (args.output / "exported_program.txt").write_text(
            result.exported_program_text, encoding="utf-8"
        )
        printer.write(result.semantic, args.output / "01_semantic.aloi")
        printer.write(result.sharded, args.output / "02_sharded.aloi")
        printer.write(result.device, args.output / "03_device.aloi")
        printer.write(result.aim, args.output / "04_aim.aloi")
    (args.output / "05_aim.trace").write_text(result.trace, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
