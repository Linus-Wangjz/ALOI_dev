# ALOI v0.1

ALOI is a metadata-only compiler prototype that lowers one Llama2-70B decoder
block from `torch.export` through semantic, device, and physical AIM IR into an
AiM-simulator trace. Real tensor dimensions and decoder dataflow are retained;
real 70B weights are never allocated.

The reference demo is:

```bash
python -m aloi.cli.compile \
  examples/llama2_70b \
  --seq-len 2048 \
  --tp 8 \
  --device configs/devices/cent.yaml \
  --schedule examples/llama2_70b/schedule.py \
  --dump-ir \
  --output build/
```

It emits `exported_program.txt`, `01_semantic.aloi`, `02_sharded.aloi`,
`03_device.aloi`, `04_aim.aloi`, and `05_aim.trace`.

This v0.1 prototype deliberately does not implement numerical execution,
checkpoint loading, dynamic sequence lengths, mapping search, or performance
modeling. PNM and CXL work remains present in AIM IR and is emitted as trace
comments because the AiM simulator consumes only PIM commands.

The acceptance-criterion mapping and validation evidence are recorded in
[`docs/validation.md`](docs/validation.md).
# ALOI_dev
