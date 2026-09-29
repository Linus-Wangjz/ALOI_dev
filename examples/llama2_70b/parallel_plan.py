"""Reference TP ownership for the toy Llama2-70B block."""

from __future__ import annotations

from aloi.parallel.plan import ParallelPlan


def make_parallel_plan(tp: int) -> ParallelPlan:
    plan = ParallelPlan(tp=tp)
    if tp == 1:
        return plan
    plan.shard("attention.q", axis="q_head", mesh_axis="tp")
    plan.shard("attention.k", axis="kv_head", mesh_axis="tp")
    plan.shard("attention.v", axis="kv_head", mesh_axis="tp")
    plan.shard("feed_forward.gate", axis="intermediate", mesh_axis="tp")
    plan.shard("feed_forward.up", axis="intermediate", mesh_axis="tp")
    return plan
