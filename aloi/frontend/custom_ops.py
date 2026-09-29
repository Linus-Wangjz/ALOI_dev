"""Shape-only custom operations used as the torch.export frontend contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


@torch.library.custom_op("aloi::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return a metadata-only linear projection result."""

    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)


@linear.register_fake
def _linear_fake(x: Tensor, weight: Tensor) -> Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op("aloi::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Return the RMSNorm result shape without doing the reduction."""

    del weight, eps
    return torch.empty_like(x)


@rms_norm.register_fake
def _rms_norm_fake(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    del weight, eps
    return torch.empty_like(x)


@torch.library.custom_op("aloi::rope", mutates_args=())
def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shape-only rotary-position-encoded Q and K tensors."""

    del position, theta
    return torch.empty_like(q), torch.empty_like(k)


@rope.register_fake
def _rope_fake(q: Tensor, k: Tensor, position: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    del position, theta
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("aloi::kv_update", mutates_args=())
def kv_update(
    cache: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Represent a functional persistent-cache update without materializing data."""

    del value, position
    return torch.empty_like(cache)


@kv_update.register_fake
def _kv_update_fake(cache: Tensor, value: Tensor, position: Tensor) -> Tensor:
    del value, position
    return torch.empty_like(cache)
