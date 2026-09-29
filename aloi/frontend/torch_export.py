"""torch.export entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ExportBundle:
    """An exported program and the metadata-only example inputs that produced it."""

    program: Any
    inputs: tuple[torch.Tensor, ...]

    def text(self) -> str:
        return str(self.program)


def export_model(model: torch.nn.Module, *inputs: torch.Tensor) -> ExportBundle:
    """Export ``model`` with strict graph capture.

    ALOI models and example tensors live on the meta device, so this captures
    shape and dataflow without allocating or executing the real GEMVs.
    """

    if any(tensor.device.type != "meta" for tensor in inputs):
        raise ValueError("ALOI export inputs must be metadata-only tensors on the meta device")
    for name, parameter in model.named_parameters():
        if parameter.device.type != "meta":
            raise ValueError(f"parameter {name!r} is not on the meta device")
    model.eval()
    return ExportBundle(torch.export.export(model, tuple(inputs), strict=True), tuple(inputs))
