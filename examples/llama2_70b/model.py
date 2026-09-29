"""A compiler-oriented, metadata-only Llama2-70B transformer block."""

from __future__ import annotations

import torch

from aloi.frontend import custom_ops
from examples.llama2_70b.config import LLAMA2_70B, Llama70BConfig


def _meta_parameter(*shape: int, dtype: torch.dtype) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False)


class ToyLlama70BBlock(torch.nn.Module):
    """One global/unsharded decoder block with real Llama2-70B dimensions."""

    def __init__(self, config: Llama70BConfig = LLAMA2_70B) -> None:
        super().__init__()
        self.config = config
        dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
        hidden = config.hidden_size
        kv_size = config.num_key_value_heads * config.head_dim
        intermediate = config.intermediate_size

        self.attention_norm = _meta_parameter(hidden, dtype=dtype)
        self.wq = _meta_parameter(hidden, hidden, dtype=dtype)
        self.wk = _meta_parameter(kv_size, hidden, dtype=dtype)
        self.wv = _meta_parameter(kv_size, hidden, dtype=dtype)
        self.wo = _meta_parameter(hidden, hidden, dtype=dtype)
        self.ffn_norm = _meta_parameter(hidden, dtype=dtype)
        self.w1 = _meta_parameter(intermediate, hidden, dtype=dtype)
        self.w3 = _meta_parameter(intermediate, hidden, dtype=dtype)
        self.w2 = _meta_parameter(hidden, intermediate, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch, tokens, _ = x.shape

        normed = custom_ops.rms_norm(x, self.attention_norm, cfg.rms_norm_eps)
        q = custom_ops.linear(normed, self.wq).reshape(
            batch, tokens, cfg.num_attention_heads, cfg.head_dim
        )
        k = custom_ops.linear(normed, self.wk).reshape(
            batch, tokens, cfg.num_key_value_heads, cfg.head_dim
        )
        v = custom_ops.linear(normed, self.wv).reshape(
            batch, tokens, cfg.num_key_value_heads, cfg.head_dim
        )
        q, k = custom_ops.rope(q, k, position, cfg.rope_theta)
        new_k_cache = custom_ops.kv_update(k_cache, k, position)
        new_v_cache = custom_ops.kv_update(v_cache, v, position)

        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        expanded_k = (
            new_k_cache.unsqueeze(2)
            .expand(batch, cfg.num_key_value_heads, groups, k_cache.shape[2], cfg.head_dim)
            .reshape(batch, cfg.num_attention_heads, k_cache.shape[2], cfg.head_dim)
        )
        expanded_v = (
            new_v_cache.unsqueeze(2)
            .expand(batch, cfg.num_key_value_heads, groups, v_cache.shape[2], cfg.head_dim)
            .reshape(batch, cfg.num_attention_heads, v_cache.shape[2], cfg.head_dim)
        )

        q_heads = q.transpose(1, 2)
        scores = torch.matmul(q_heads, expanded_k.transpose(-1, -2))
        scores = torch.mul(scores, cfg.head_dim**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, expanded_v)
        context = context.transpose(1, 2).reshape(batch, tokens, cfg.hidden_size)
        attention_output = custom_ops.linear(context, self.wo)
        residual = torch.add(x, attention_output)

        ffn_input = custom_ops.rms_norm(residual, self.ffn_norm, cfg.rms_norm_eps)
        gate = torch.nn.functional.silu(custom_ops.linear(ffn_input, self.w1))
        up = custom_ops.linear(ffn_input, self.w3)
        fused = torch.mul(gate, up)
        ffn_output = custom_ops.linear(fused, self.w2)
        output = torch.add(residual, ffn_output)
        return output, new_k_cache, new_v_cache


def example_inputs(
    seq_len: int,
    config: Llama70BConfig = LLAMA2_70B,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
    return (
        torch.empty((1, 1, config.hidden_size), dtype=dtype, device="meta"),
        torch.empty(
            (1, config.num_key_value_heads, seq_len, config.head_dim),
            dtype=dtype,
            device="meta",
        ),
        torch.empty(
            (1, config.num_key_value_heads, seq_len, config.head_dim),
            dtype=dtype,
            device="meta",
        ),
        torch.tensor(0, dtype=torch.int64, device="meta"),
    )
