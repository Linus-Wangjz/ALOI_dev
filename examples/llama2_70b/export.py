"""Export helper for the Llama2-70B example."""

from __future__ import annotations

from aloi.frontend.torch_export import ExportBundle, export_model
from examples.llama2_70b.model import ToyLlama70BBlock, example_inputs


def export_block(seq_len: int) -> ExportBundle:
    return export_model(ToyLlama70BBlock(), *example_inputs(seq_len))
