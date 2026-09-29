"""Fixed dimensions for the compiler-oriented Llama2-70B block."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Llama70BConfig:
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 28672
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: str = "fp16"
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.num_layers != 1:
            raise ValueError("the v0.1 frontend represents exactly one transformer block")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal query heads times head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")


LLAMA2_70B = Llama70BConfig()
