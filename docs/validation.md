# ALOI v0.1 validation record

This record maps the active execution plan to executable evidence. It is not a
numerical-correctness or performance claim; those are explicitly outside v0.1.

## Acceptance review

- Python IR: `Operation`, `Value`, `TensorType`, `Function`, `Module`, `Printer`,
  and `Pass` are implemented. The test suite constructs and prints
  `aloi.rms_norm` and `aloi.linear` manually.
- Frontend: `ToyLlama70BBlock` has one decoder layer, exact Llama2-70B tensor
  dimensions, GQA, RoPE, functional KV updates, attention, residuals, and
  SwiGLU. All parameters and example inputs are meta tensors with null data
  pointers. Strict `torch.export.export` succeeds at `SEQ_LEN=2048`.
- Semantic IR: the importer canonicalizes the exported graph to the v0.1 op
  set. Every tensor has a shape, dtype, role, and axis semantics; cache values
  are persistent. Semantic verification rejects non-semantic operations and
  physical metadata.
- Tensor parallelism: the global 64-Q-head/8-KV-head model is seeded by
  `ParallelPlan(tp=8)` and becomes 8 Q heads and 1 KV head per partition.
  Sharding propagation partitions projection/FFN weights and identifies the
  `Wo` and `W2` reductions. Two `aloi.all_reduce` operations are derived; the
  user plan contains no communication directives.
- Device legalization: the YAML `DeviceSpec`, recipe registry, legality check,
  first-legal selector, and lowering pass eliminate all `aloi.*` operations.
  The resulting operations use only `pim.*`, `pnm.*`, `cxl.*`, and
  `device.transfer`.
- Mapping: all eight planned DSL directives are available. The reference
  schedule supplies partial constraints, the default mapper completes them,
  and physical allocation records device, tile, channel group, bank group,
  DRAM row range, and global-buffer allocation.
- AIM and trace: allocated Device IR lowers to SSA AIM IR with `aim.view`,
  `aim.load_gb`, `aim.pim.mac_dram_gb`, and `aim.read_acc`. PNM and CXL work is
  retained as `aim.pnm.*` and `aim.cxl.*`. The emitter produces real `WR_GB`,
  `MAC_ABK`, `RD_MAC`, elementwise, and final `EOC` commands while representing
  PNM/CXL operations as accepted comments.
- Demo: the exact command in the plan produces all six named files under
  `build/` for `SEQ_LEN=2048` and TP=8.

## Validation commands

The implementation was validated in the Anaconda `cent` environment with:

```bash
python -m unittest discover -s tests -v
mypy aloi examples tests
ruff check aloi examples tests
python -m aloi.cli.compile examples/llama2_70b \
  --seq-len 2048 --tp 8 \
  --device configs/devices/cent.yaml \
  --schedule examples/llama2_70b/schedule.py \
  --dump-ir --output build/
```

The generated `build/05_aim.trace` was also passed to the bundled AiM simulator
with its `test/example.yaml` configuration. The simulator exited successfully,
reported one EOC, seven `MAC_ABK`, seven `WR_GB`, seven `RD_MAC`, six `EWMUL`,
and two `EWADD` requests.

## Deviations

None. The implementation stays within the explicit v0.1 scope and does not add
weight loading, numerical execution, prefill, dynamic sequence lengths,
mapping search, a cost model, or PNM/CXL timing.
