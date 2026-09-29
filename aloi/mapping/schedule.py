"""Partial-constraint Python mapping DSL."""

from __future__ import annotations

import runpy
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Directive:
    kind: str
    parameters: dict[str, Any]


_ACTIVE_SINK: ContextVar[list[Schedule] | None] = ContextVar("aloi_schedule_sink", default=None)
_INTERACTIVE_SCHEDULES: list[Schedule] = []


class Schedule:
    """A collection of partial physical constraints for one semantic target."""

    def __init__(self, target: str) -> None:
        if not target:
            raise ValueError("schedule target must be non-empty")
        self.target = target
        self.directives: list[Directive] = []

    def __enter__(self) -> Schedule:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            return
        sink = _ACTIVE_SINK.get()
        (sink if sink is not None else _INTERACTIVE_SCHEDULES).append(self)

    def _add(self, kind: str, **parameters: Any) -> Schedule:
        self.directives.append(Directive(kind, parameters))
        return self

    def split(self, axis: str, *, factor: int) -> Schedule:
        return self._factor("split", axis, factor)

    def tile(self, axis: str, *, factor: int) -> Schedule:
        return self._factor("tile", axis, factor)

    def _factor(self, kind: str, axis: str, factor: int) -> Schedule:
        if factor < 1:
            raise ValueError(f"{kind} factor must be positive")
        return self._add(kind, axis=axis, factor=factor)

    def place(self, value: str, device: str) -> Schedule:
        return self._add("place", value=value, device=device)

    def replicate(self, axis: str, *, factor: int) -> Schedule:
        return self._factor("replicate", axis, factor)

    def shard(self, axis: str, mesh_axis: str) -> Schedule:
        return self._add("shard", axis=axis, mesh_axis=mesh_axis)

    def bind(self, axis: str, resource: str) -> Schedule:
        return self._add("bind", axis=axis, resource=resource)

    def reduce_at(self, axis: str) -> Schedule:
        return self._add("reduce_at", axis=axis)

    def pipeline(self, *stages: str) -> Schedule:
        if not stages:
            raise ValueError("pipeline needs at least one stage")
        return self._add("pipeline", stages=list(stages))


def schedule(target: str) -> Schedule:
    return Schedule(target)


def load_schedules(path: str | Path) -> list[Schedule]:
    collected: list[Schedule] = []
    token = _ACTIVE_SINK.set(collected)
    try:
        namespace = runpy.run_path(str(path))
    finally:
        _ACTIVE_SINK.reset(token)
    declared = namespace.get("SCHEDULES")
    if declared is not None:
        if not isinstance(declared, list) or not all(
            isinstance(item, Schedule) for item in declared
        ):
            raise TypeError("SCHEDULES must be a list of Schedule objects")
        for item in declared:
            if item not in collected:
                collected.append(item)
    return collected
