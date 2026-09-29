"""Types shared by all ALOI IR levels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import prod

_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int32": 4, "int64": 8, "bool": 1}


def normalize_dtype(dtype: object) -> str:
    """Return the stable ALOI spelling for a dtype-like object."""

    text = str(dtype).lower().replace("torch.", "")
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        "long": "int64",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class ShardingSpec:
    """Ownership of one logical tensor axis by a mesh axis."""

    axis: str
    mesh_axis: str
    parts: int

    def __post_init__(self) -> None:
        if not self.axis or not self.mesh_axis:
            raise ValueError("sharding axes must be non-empty")
        if self.parts < 1:
            raise ValueError("sharding parts must be positive")


@dataclass(frozen=True)
class TensorType:
    """Tensor shape plus semantic and physical annotations.

    ``shape`` is the current (possibly local) shape. ``global_shape`` is set
    after sharding, preserving the model-level type without ambiguity.
    """

    shape: tuple[int, ...]
    dtype: str = "fp16"
    role: str = "activation"
    axes: tuple[str, ...] = ()
    persistent: bool = False
    sharding: ShardingSpec | None = None
    global_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "axes", tuple(self.axes))
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"tensor dimensions must be non-negative: {self.shape}")
        if self.axes and len(self.axes) != len(self.shape):
            raise ValueError(f"{len(self.axes)} axes do not describe rank-{len(self.shape)} tensor")
        if self.global_shape is not None and len(self.global_shape) != len(self.shape):
            raise ValueError("global and local tensor ranks differ")

    @classmethod
    def get(
        cls,
        shape: Iterable[int],
        dtype: object = "fp16",
        *,
        role: str = "activation",
        axes: Iterable[str] = (),
        persistent: bool = False,
    ) -> TensorType:
        return cls(tuple(shape), normalize_dtype(dtype), role, tuple(axes), persistent)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        try:
            itemsize = _DTYPE_BYTES[self.dtype]
        except KeyError as exc:
            raise ValueError(f"unknown dtype size: {self.dtype}") from exc
        return self.numel * itemsize

    def with_metadata(
        self,
        *,
        role: str | None = None,
        axes: Iterable[str] | None = None,
        persistent: bool | None = None,
    ) -> TensorType:
        return replace(
            self,
            role=self.role if role is None else role,
            axes=self.axes if axes is None else tuple(axes),
            persistent=self.persistent if persistent is None else persistent,
        )

    def shard(self, axis: str, mesh_axis: str, parts: int) -> TensorType:
        if axis not in self.axes:
            raise ValueError(f"axis {axis!r} not present in {self.axes}")
        index = self.axes.index(axis)
        global_shape = self.global_shape or self.shape
        size = global_shape[index]
        if size % parts:
            raise ValueError(f"axis {axis!r} of size {size} is not divisible by {parts}")
        local_shape = list(global_shape)
        local_shape[index] = size // parts
        return replace(
            self,
            shape=tuple(local_shape),
            sharding=ShardingSpec(axis=axis, mesh_axis=mesh_axis, parts=parts),
            global_shape=global_shape,
        )
