# ALOI v0.1 Implementation Plan

## 1. v0.1 唯一目标

ALOI v0.1 是一个 lowering prototype。

不要求：

* numerical correctness
* performance correctness
* 和 CENT latency 对齐
* 和 HuggingFace output 对齐
* 加载真实 Llama2 checkpoint
* 运行完整 80-layer model
* cost-model optimization

唯一 Definition of Done：

```text
Toy Llama2-70B Transformer Block
              ↓
          torch.export
              ↓
        Semantic IR
              ↓
         ParallelPlan
              ↓
     Sharding Propagation
              ↓
 Automatic Collective Insertion
              ↓
       Device Lowering
              ↓
          Device IR
              ↓
 Mapping DSL + Default Mapping
              ↓
           AIM IR
              ↓
        AIM Trace Emitter
              ↓
aim_simulator-compatible .trace
```

只验证：

> 整个 lowering chain 能否完整走通。

---

# 2. Frontend 不再使用现有 Llama implementation

ALOI repository 自己包含：

```text
examples/llama2_70b/
    model.py
```

这是一个专门为 compiler frontend 设计的：

```text
ToyLlama70BBlock
```

它不是 HuggingFace model implementation，也不是 Meta inference runtime。

它只需要保持：

```text
真实 Llama2-70B tensor dimensions
+
真实 decoder-block dataflow
+
GQA structure
+
KV-cache decode structure
```

因此没有：

```text
FairScale
distributed runtime
CUDA allocation
checkpoint loading
tokenizer
embedding
LM head
80 decoder layers
```

只有：

```text
one TransformerBlock
```

---

# 3. 固定的 Llama2-70B Config

```python
@dataclass
class Llama70BConfig:
    hidden_size: int = 8192

    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128

    intermediate_size: int = 28672

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    dtype: str = "fp16"
```

完整 Llama2-70B 有 80 层，但 toy frontend：

```text
num_layers = 1
```

因为编译目标本来就是：

```text
one representative TransformerBlock
```

而不是完整模型。

---

# 4. Toy Block 的精确 Dataflow

输入：

```text
x
[batch=1, token=1, hidden=8192]

k_cache
[batch=1, kv_head=8, sequence=SEQ_LEN, head_dim=128]

v_cache
[batch=1, kv_head=8, sequence=SEQ_LEN, head_dim=128]
```

`SEQ_LEN` 在 v0.1 是 compile-time parameter。

例如：

```text
SEQ_LEN = 2048
```

---

## Attention

```text
x
 │
 ▼
RMSNorm
 │
 ├──────────────┬──────────────┐
 ▼              ▼              ▼
Wq Linear      Wk Linear      Wv Linear
 │              │              │
 ▼              ▼              ▼
Q              K              V
 │              │
 └──── RoPE ────┘
 │              │
 │          KV Update
 │              │
 ▼              ▼
Q         K/V Cache
 │              │
 └──────┬───────┘
        ▼
      Q @ Kᵀ
        │
      scale
        │
     softmax
        │
        ▼
   scores @ V
        │
        ▼
     Wo Linear
        │
        ▼
   Residual Add
```

然后：

```text
RMSNorm
   │
   ├────────────┐
   ▼            ▼
 W1 Linear    W3 Linear
   │            │
 SiLU           │
   │            │
   └──── Mul ───┘
         │
         ▼
      W2 Linear
         │
         ▼
   Residual Add
```

---

# 5. 真实 Tensor Shapes

Attention projection：

```text
Wq:
    [8192, 8192]

Wk:
    [1024, 8192]

Wv:
    [1024, 8192]

Wo:
    [8192, 8192]
```

因为：

```text
Q:
64 heads × 128 = 8192

K/V:
8 heads × 128 = 1024
```

Decode 输入：

```text
x:
[1, 1, 8192]
```

得到：

```text
Q:
[1, 1, 64, 128]

K:
[1, 1, 8, 128]

V:
[1, 1, 8, 128]
```

FFN：

```text
W1:
[28672, 8192]

W3:
[28672, 8192]

W2:
[8192, 28672]
```

两个 RMSNorm weight：

```text
gamma:
[8192]
```

这些 shape 是 compiler testcase 的核心。

---

# 6. 不创建真实 70B Weight

这一点作为正式设计原则：

> Toy model 表示真实 shape，而不是实际 data。

例如不能：

```python
nn.Linear(8192, 28672)
```

然后真的分配 FP16 storage。

而采用：

```text
MetaTensor / FakeTensor
```

或者：

```text
Parameter on meta device
```

因此：

```text
Wq : tensor<8192x8192xf16>
```

存在于 graph/type system 中，但：

```text
allocated bytes = 0
```

我们只需要：

```text
shape
dtype
semantic identity
```

不需要 weight content。

---

# 7. Compiler-friendly PyTorch Custom Ops

Toy model 可以有意使用 ALOI custom ops。

这不是为了模拟真实 PyTorch runtime，而是给 compiler 一个稳定 frontend contract。

第一版：

```text
torch.ops.aloi.linear
torch.ops.aloi.rms_norm
torch.ops.aloi.rope
torch.ops.aloi.kv_update
```

可以进一步让这些较复杂的 attention primitives也保持普通 PyTorch op：

```text
torch.matmul
torch.add
torch.mul
torch.softmax
reshape
transpose
expand
```

Custom op 都提供 FakeTensor kernel。

例如概念上：

```python
@torch.library.custom_op(
    "aloi::linear",
    mutates_args=()
)
def linear(x, weight):
    ...

@linear.register_fake
def linear_fake(x, weight):
    output_shape = ...
    return torch.empty(
        output_shape,
        dtype=x.dtype,
        device=x.device,
    )
```

因此 `torch.export`：

```text
只进行 shape/dataflow tracing

而不执行：

8192 × 28672 GEMV
```

---

# 8. KV Cache 采用显式函数式接口

Toy model 不拥有内部：

```python
self.cache_k
self.cache_v
```

而是：

```python
def forward(
    x,
    k_cache,
    v_cache,
    position,
):
```

内部：

```text
new_k_cache = aloi.kv_update(
    k_cache,
    new_k,
    position
)

new_v_cache = aloi.kv_update(
    v_cache,
    new_v,
    position
)
```

返回：

```text
output
new_k_cache
new_v_cache
```

所以整个 block 在 frontend 看来是：

```text
(x, Kcache, Vcache)
        ↓
 TransformerBlock
        ↓
(y, Kcache', Vcache')
```

这非常适合以后扩展到：

```text
paged KV cache
```

因为 Semantic IR 只知道：

```text
logical persistent KV tensor
```

而不知道物理 page。

---

# 9. Semantic Tensor

Semantic tensor metadata：

```text
shape
dtype
role
axis semantics
persistent
sharding
```

例如：

```text
%x:
tensor<
    1x1x8192xf16,
    role=activation,
    axes=[batch, token, hidden]
>
```

Q：

```text
%q:
tensor<
    1x1x64x128xf16,
    role=activation,
    axes=[batch, token, q_head, head_dim]
>
```

KV cache：

```text
%k_cache:
tensor<
    1x2048x8x128xf16,
    role=kv_cache,
    persistent=true,
    axes=[batch, sequence, kv_head, head_dim]
>
```

Weight：

```text
%wq:
tensor<
    8192x8192xf16,
    role=weight,
    axes=[out_feature, in_feature]
>
```

Semantic IR 不允许出现：

```text
PIM
PNM
CXL
channel
bank
GB
row
```

---

# 10. Semantic Op Set

v0.1：

```text
aloi.linear

aloi.rms_norm
aloi.rope
aloi.kv_update

aloi.matmul
aloi.softmax
aloi.silu

aloi.add
aloi.mul

aloi.reshape
aloi.transpose
aloi.expand
aloi.slice
```

Parallelization pass 可以另外生成：

```text
aloi.all_reduce
aloi.broadcast
```

Frontend 不生成 collective。

---

# 11. ParallelPlan

原则不变：

> ParallelPlan describes ownership, not communication.

例如：

```python
plan = ParallelPlan(tp=8)

plan.shard(
    "attention.q",
    axis="q_head",
    mesh_axis="tp",
)

plan.shard(
    "attention.k",
    axis="kv_head",
    mesh_axis="tp",
)

plan.shard(
    "attention.v",
    axis="kv_head",
    mesh_axis="tp",
)
```

ALOI：

```text
seed sharding
      ↓
propagate sharding
      ↓
analyze operation semantics
      ↓
detect partitioned reduction
      ↓
insert collective
```

例如：

```text
Wo Linear
```

如果 input feature 对 TP partition：

```text
local partial output
       ↓
aloi.all_reduce
```

用户不用写 `all_reduce`。

---

# 12. 一个重要 simplification

Toy model **永远是 global/unsharded model**。

也就是说：

```text
64 Q heads
8 KV heads
```

永远完整地存在于最初 Semantic IR。

TP=8 后才变成：

```text
per partition:

Q heads = 8
KV heads = 1
```

这可以让我们真正测试：

```text
Model semantics
        ↓
ParallelPlan
```

而不是直接从一个已经 tensor-parallelized 的 PyTorch implementation 开始。

---

# 13. Semantic → Device Lowering

结构仍然使用：

```text
LoweringRecipeRegistry
```

Python API：

```python
class LoweringRecipe:
    def match(op):
        ...

    def is_legal(op, device_spec):
        ...

    def lower(op, context):
        ...
```

所有合法 candidate 都可以注册。

但 v0.1：

```text
enumerate legal candidates
        ↓
select first legal
```

没有 cost model。

---

# 14. Device IR

继续保持：

```text
pim.*
pnm.*
device.transfer
cxl.*
```

例如：

```text
aloi.linear
    ↓
pim.gemv
```

RMSNorm：

```text
aloi.rms_norm
       ↓
pim.ewmul
       ↓
device.transfer
       ↓
pnm.reduce_sum
       ↓
pnm.rsqrt
       ↓
device.transfer
       ↓
pim.ewmul
```

Softmax：

```text
aloi.softmax
    ↓
pnm.softmax
```

CXL 是 communication fabric：

```text
cxl.send
cxl.recv
cxl.broadcast
cxl.all_reduce
```

而不是 compute device。

---

# 15. DeviceSpec

继续采用：

```text
YAML
```

并从第一天支持 parameterized capability：

```yaml
pim:

  gemv:
    supported: true
    dtypes:
      - fp16

    vector_width: 16

  ewmul:
    supported: true

pnm:

  reduce_sum:
    supported: true

  rsqrt:
    supported: true

  softmax:
    supported: true

cxl:

  broadcast:
    supported: true

  all_reduce:
    supported: true
```

v0.1 recipe selection：

```text
capability legality only
```

不做 cost comparison。

---

# 16. Mapping DSL

Python DSL。

例如：

```python
with schedule("attention.wq") as s:

    s.tile(
        "in_feature",
        factor=256,
    )

    s.bind(
        "out_feature",
        "channel",
    )

    s.place(
        "weight",
        "pim",
    )
```

支持：

```text
split
tile
place
replicate
shard
bind
reduce_at
pipeline
```

它是：

```text
partial constraint
```

没有指定的部分：

```text
DefaultMapper
```

自动补齐。

v0.1 不搜索最优解。

---

# 17. AIM IR

保持较高级 SSA/dataflow representation。

例如：

```text
%w_tile = aim.view %weight {...}

%x_gb = aim.load_gb %x {...}

%acc =
    aim.pim.mac_dram_gb
        %w_tile,
        %x_gb

%result =
    aim.read_acc %acc
```

AIM IR 开始记录：

```text
DRAM / GB

device
channel group
bank group
row range

GB allocation

physical tile
```

但 v0.1：

```text
不实现完整 verifier
```

我们依赖：

```text
Device→AIM lowering
```

生成合法 operands。

---

# 18. AIM Codegen

例如：

```text
aim.load_gb
       ↓
AiM WR_GB
```

```text
aim.pim.mac_dram_gb
       ↓
one or more
AiM MAC_ABK
```

```text
aim.read_acc
       ↓
AiM RD_MAC
```

最终：

```text
AIM IR
    ↓
AimTraceEmitter
    ↓
*.trace
```

结束：

```text
AiM EOC
```

---

# 19. PNM / CXL 在 v0.1 的处理

ALOI IR 中完整保留：

```text
aim.pnm.*
aim.cxl.*
```

但是 aim_simulator 只消费 PIM 部分。

所以 emitter：

```text
PIM AIM operation
    → real ISR command

PNM AIM operation
    → trace comment

CXL AIM operation
    → trace comment
```

例如：

```text
# ALOI_PNM reduce_sum ...
# ALOI_PNM rsqrt ...
# ALOI_CXL all_reduce ...
```

因此 `.trace`：

```text
依然能被 aim_simulator parse
```

同时整个 ALOI lowering chain 没有丢掉 PNM/CXL semantics。

以后增加 CENT analytical backend：

```text
AIM IR
   ├── PIM → AiM simulator
   └── PNM/CXL → analytical timing
```

v0.1 不实现 timing aggregation。

---

# 20. Revised Repository Layout

```text
aloi/
├── frontend/
│   ├── custom_ops.py
│   ├── torch_export.py
│   └── importer.py
│
├── ir/
│   ├── base.py
│   ├── types.py
│   ├── semantic.py
│   ├── device.py
│   ├── aim.py
│   └── printer.py
│
├── parallel/
│   ├── plan.py
│   ├── propagation.py
│   └── collectives.py
│
├── lowering/
│   ├── registry.py
│   ├── selector.py
│   └── recipes/
│       ├── linear.py
│       ├── rmsnorm.py
│       ├── attention.py
│       └── elementwise.py
│
├── device/
│   └── spec.py
│
├── mapping/
│   ├── schedule.py
│   ├── plan.py
│   ├── default_mapper.py
│   └── allocator.py
│
├── backend/
│   └── aim/
│       ├── lowering.py
│       ├── isr.py
│       └── trace_emitter.py
│
└── cli/
    └── compile.py


examples/
└── llama2_70b/
    ├── model.py
    ├── config.py
    ├── export.py
    ├── parallel_plan.py
    └── schedule.py


configs/
└── devices/
    └── cent.yaml
```

删除之前计划中的：

```text
configs/models/llama2_70b.yaml
```

也可以。

因为这个 toy model 的 architecture 本身就是 compiler testcase，可以直接：

```python
LLAMA2_70B = LlamaConfig(...)
```

保存在：

```text
examples/llama2_70b/config.py
```

---

# 21. Pass Pipeline

最终固定为：

```text
ToyLlama70BBlock
        ↓
torch.export
        ↓
ImportExportedProgram
        ↓
SemanticCanonicalize
        ↓
AnnotateRolesAndAxes
        ↓
        │
        │  Semantic IR
        ▼
ApplyParallelPlan
        ↓
PropagateSharding
        ↓
InsertCollectives
        ↓
EnumerateLegalRecipes
        ↓
SelectFirstLegal
        ↓
LowerSemanticToDevice
        ↓
        │
        │  Device IR
        ▼
ApplyScheduleConstraints
        ↓
CompleteDefaultMapping
        ↓
PhysicalAllocation
        ↓
LowerDeviceToAIM
        ↓
        │
        │  AIM IR
        ▼
EmitAimSimulatorTrace
```

---

# 22. Revised Milestones

## M0 — Python IR

只实现：

```text
Operation
Value
TensorType
Function
Module
Printer
Pass
```

完成条件：

```text
能手写并打印：

aloi.linear
aloi.rms_norm
```

---

## M1 — Toy Llama2-70B Model

这是现在真正的第一个重要 checkpoint。

实现：

```text
ToyLlama70BBlock
```

只使用：

```text
meta/fake parameters
```

包含：

```text
RMSNorm
Q/K/V projection
RoPE
KV update
GQA attention
output projection
residual
RMSNorm
SwiGLU
residual
```

完成条件：

```python
torch.export.export(...)
```

成功。

不加载 checkpoint。

不执行真实 GEMV。

不要求输出正确。

---

## M2 — ExportedProgram → Semantic IR

完成条件：

```text
01_semantic.aloi
```

包含类似：

```text
aloi.rms_norm
aloi.linear
aloi.rope
aloi.kv_update

aloi.matmul
aloi.softmax

aloi.linear
aloi.silu
aloi.mul
aloi.linear
```

并且所有 tensor：

```text
shape正确
semantic axes存在
role存在
```

---

## M3 — TP

输入：

```text
--tp 8
```

完成：

```text
head sharding
KV-head sharding
sharding propagation
automatic all-reduce insertion
```

输出：

```text
02_sharded.aloi
```

这是第一个真正验证：

> ParallelPlan describes ownership, communication is derived.

的 milestone。

---

## M4 — Device Legalization

实现：

```text
DeviceSpec
LoweringRecipe
RecipeRegistry
FirstLegalSelector
```

得到：

```text
03_device.aloi
```

要求 Semantic compute op 基本全部消失。

剩：

```text
pim.*
pnm.*
cxl.*
device.transfer
```

---

## M5 — Mapping

实现最笨但 deterministic 的：

```text
DefaultMapper
```

再允许：

```text
schedule.py
```

覆盖部分 decision。

暂时：

```text
no search
no cost model
```

---

## M6 — AIM IR

生成：

```text
04_aim.aloi
```

此时已经有：

```text
tile
channel
bank group
DRAM rows
GB allocation
```

但还是 SSA/dataflow IR。

---

## M7 — ISR

实现：

```text
AimTraceEmitter
```

生成：

```text
05_aim.trace
```

只要求：

```text
aim_simulator trace parser accepts it
```

不比较性能。

不比较 correctness。

不和 CENT 对齐。

---

# 23. 最终 Demo

最终只需要一个命令：

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

输出：

```text
build/
├── exported_program.txt
├── 01_semantic.aloi
├── 02_sharded.aloi
├── 03_device.aloi
├── 04_aim.aloi
└── 05_aim.trace
```

然后：

```text
05_aim.trace
       ↓
aim_simulator
       ↓
trace accepted
```

ALOI v0.1 即完成。

---

# 24. v0.1 明确不做

为了防止 scope creep，下列全部推迟：

```text
real Llama weights
HuggingFace dependency
full 80-layer model
embedding
LM head

prefill
batch > 1
dynamic seq_len
paged attention

functional interpreter
numerical validation

MLIR/xDSL

AIM IR type verifier

cost model
mapping search
auto tuning

performance comparison with CENT

PNM timing model
CXL timing model
```

这些都不是 v0.1 lowering prototype 的 prerequisite。

---

# 25. v0.1 的核心研究问题

最终这个 prototype 实际验证的是三个问题：

1. **能否从 model-level PyTorch dataflow 保留足够 semantic information？**

2. **能否从 semantic tensor ownership 自动推导 TP communication？**

3. **能否用 DeviceSpec + lowering recipe + Mapping DSL，把同一个 semantic graph 系统地 lowering 成 AIM physical operations 和 ISR commands？**

如果这三个成立，ALOI 的 architecture 就已经被验证。

之后 performance model、自动 mapping、更多 model、dynamic shape，都只是沿现有 boundary 扩展。
