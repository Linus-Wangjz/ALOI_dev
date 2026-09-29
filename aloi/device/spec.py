"""Parameterized YAML device capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aloi.ir.types import normalize_dtype


@dataclass(frozen=True)
class DeviceSpec:
    data: Mapping[str, Any]
    source: str = "<memory>"

    @classmethod
    def load(cls, path: str | Path) -> DeviceSpec:
        source = Path(path)
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"device spec {source} must contain a mapping")
        return cls(loaded, str(source))

    def capability(self, operation: str) -> Mapping[str, Any] | None:
        parts = operation.split(".")
        current: Any = self.data
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        if isinstance(current, bool):
            return {"supported": current}
        return current if isinstance(current, Mapping) else None

    def supports(self, operation: str, dtype: object | None = None) -> bool:
        capability = self.capability(operation)
        if capability is None or not bool(capability.get("supported", False)):
            return False
        dtypes = capability.get("dtypes")
        if dtype is None or not dtypes:
            return True
        return normalize_dtype(dtype) in {normalize_dtype(item) for item in dtypes}

    def parameter(self, path: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current
