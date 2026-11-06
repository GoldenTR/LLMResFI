# LLMResFI

<p align="center">
  <b>LLMResFI = LLM Resilience Fault Injection</b><br/>
  面向大语言模型（LLM）<b>训练</b>与<b>推理</b>的软错误韧性故障注入框架，原生支持多 GPU 分布式训练（DDP/FSDP）。
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="#"><img src="https://img.shields.io/badge/LLMResFI-v0.2-orange.svg" alt="LLMResFI v0.2"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

---

## 为什么需要 LLMResFI？

现有故障注入工具仍存在明显能力缺口：

| 能力 | LLMFI | NVBitFI | PyTorchFI | **LLMResFI** |
|---|---|---|---|---|
| LLM 推理（阶段感知） | ✅ | ❌ | ❌ | ✅ |
| **训练支持** | ❌ | ❌ | ❌ | ✅ |
| **梯度/优化器状态故障** | ❌ | ❌ | ❌ | ✅ |
| **分布式通信故障** | ❌ | ❌ | ❌ | ✅ |
| 真实 IEEE 位翻转（符号/指数/尾数） | ⚠️ 部分支持 | ✅ | ❌ | ✅ |
| 故障时序（瞬态/永久/间歇） | ❌ | ⚠️ | ❌ | ✅ |
| 抽象层级 | 模型层 | 架构层 | 模型层 | 模型层 |

LLMResFI 使用统一 API 覆盖**训练与推理**，并可触达以往工具无法覆盖的状态：梯度、Adam 动量、混合精度主权重以及全归约通信张量。

## 主要特性

- **训练与推理统一**：使用 `run_mode` + `stage` + `target` 三要素描述注入行为。
  - 推理阶段：`prefill → first_token → decode`
  - 训练阶段：`forward → backward → optimizer_step → comm`
- **真实 IEEE 位翻转**：通过 `view(int) ^ mask` 实现，并支持定向翻转 `sign` / `exponent` / `mantissa`。
- **训练专属故障目标**：`gradient`、`optimizer`（Adam 的 `exp_avg`/`exp_avg_sq`）、`master_weight`、`loss`。
- **分布式注入**：支持 DDP 梯度桶在全归约前后的注入、FSDP 梯度归约钩子、rank 子集策略以及可选校验和守卫。
- **检测与恢复**：支持非有限值、loss、梯度范数检测，梯度清洗，无损步级回滚以及检查点回滚。
- **故障时序**：`transient`、`permanent`（持续 N 步）、`intermittent`（每 k 步触发一次）。
- **可复现**：基于随机种子、确定性贪心解码与 JSONL 日志。
- **结果分析**：每次运行生成 `summary.json`，支持相对基线的结果分类
  （`recovered` / `mitigated` / `SDC` / `masked` / `DUE`）、bit 区域敏感性以及耗时/显存开销表。

## 安装

```bash
git clone https://github.com/GoldenTR/LLMResFI.git
cd LLMResFI
pip install -e .
# 可选：安装开发与数据集依赖
pip install -e ".[dev,datasets]"
```

实际 LLM 工作负载推荐使用 GPU；CPU 可运行玩具训练示例。

框架已在 Python 3.9+、PyTorch 1.12（CUDA 11.3）以及 PyTorch 2.x 上测试。
使用 PyTorch 1.12 时请安装 `numpy<2`。

## 快速开始

### 1. 推理（单 GPU）

```bash
python examples/run_inference.py --config config.yml --prompt "Explain transformers briefly." --new_tokens 16
```

`config.yml` 的 `fault` 段控制故障目标、阶段与故障模型：

```yaml
fault:
  target: "weight"        # weight | activation | kv_cache
  stage: { type: "prefill" }   # prefill | first_token | decode
  fault_model: { type: "exponent" }   # single|double|sign|exponent|mantissa|...
  timing: { type: "transient" }       # transient | permanent | intermittent
```

### 2. 训练（单 GPU）

以下命令在小型 MLP 上演示训练专属故障目标，无需 GPU 或 Hugging Face 模型：

```bash
python examples/run_training.py --target gradient --stage backward
python examples/run_training.py --target optimizer --stage optimizer_step
python examples/run_training.py --target loss --stage backward
python examples/run_training.py --target master_weight --stage optimizer_step
```

示例输出（梯度注入）：

```text
[step 0] loss=2.3141 stage=forward
[step 1] loss=2.2390 stage=forward
[step 2] loss=2.1681 stage=forward
[step 3] loss=2.1005 stage=forward
[step 4] loss=2.0349 stage=forward

Injection record:
  target: gradient
  param_path: net.4.bias        # 故障参数（参数目标下 layer_idx 为 None）
  fault_model: single
  timing: transient
  stage: backward               # 在第 0 步反向传播期间注入
  step: 0
  token_step: -1
  flat_idx: 1
  bit_idxs: [1]                 # 翻转偏置张量第 1 个元素的第 1 位
```

### 3. 分布式训练（DDP）

```bash
# CPU / gloo 冒烟测试
torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --fault_model nan --mitigation step_rollback

# 两张 V100 / NCCL
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 3
```

> 注意：DDP 在 `loss.backward()` 期间执行梯度全归约，因此通信钩子会在 `backward` 阶段触发。

注入语义：

- `injection_point: before_reduce`：在全归约**之前**污染某个 rank 的本地梯度桶；
  错误会被平均到所有 rank（静默传播）。
- `injection_point: after_reduce`：只污染选定 rank 的全归约结果；
  该 rank 会与其余 rank 静默发散。
- `inject_rank`（单 rank）、`inject_ranks`（列表）或 `inject_rank_fraction`
  （确定性随机子集）用于控制哪些 rank 发生故障。
- `verify_comm: true`：归约前收集每个 rank 的轻量校验和，归约后同时检查
  归约一致性与跨 rank 一致性。

### 4. 分布式训练（FSDP）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection grad_shard --fault_model nan --mitigation step_rollback
```

- PyTorch >= 2.0：使用 FSDP 的梯度通信钩子（归约前/后注入）。
- PyTorch 1.12：降级为分片级注入（`param_shard` 在 all-gather 前，
  `grad_shard` 在优化器更新前）。此类记录带有 `legacy_fsdp: true`。

### 5. ResNet-18 / CIFAR-10 端到端实验

`examples/resnet18/run_resnet18.py` 在真实图像分类训练上验证“故障影响 -> 校验和检测 ->
步级回滚 -> 严格无损恢复”的完整链路。它使用 CIFAR 版 ResNet-18、DDP、分布式
sampler 和确定性预处理，并在 `summary.json` 中额外保存完整模型 state-dict 与
优化器状态的 SHA-256，用于逐 bit 无损性判定。

```bash
# 无故障基线
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario baseline --baseline --steps 96 --lr 0.05 --bit_idx 27 \
  --log_dir ../ResNet-18/runs/baseline

# 归约后非对称 SDC
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario fault_sdc --fault_model exponent --bit_idx 27 \
  --injection_point after_reduce --inject_rank 0 --mitigation none \
  --steps 96 --lr 0.05 --log_dir ../ResNet-18/runs/fault_sdc_after

# 校验和检测 + 无损步级回滚
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario fault_recovered --fault_model exponent --bit_idx 27 \
  --injection_point after_reduce --inject_rank 0 \
  --mitigation step_rollback --verify \
  --steps 96 --lr 0.05 --log_dir ../ResNet-18/runs/fault_recovered_after

# 生成中文聚合报告与 JSON
python examples/resnet18/analyze_resnet18.py \
  --root ../ResNet-18/runs \
  --output-md ../ResNet-18/analysis/report.md \
  --output-json ../ResNet-18/analysis/report.json
```

在两张 V100、双卡 DDP、96 step 的本机实验中：基线测试准确率为 54.69%；
`after_reduce` 的 bit 27 故障未被检测时降至 50.85%（-3.84 pp，测试损失
+0.0930）；开启校验和与回滚后，模型和优化器 SHA-256 均与基线完全一致。
归约前 bit 27 的小扰动会落在校验和相对容差内而漏检，说明单次全归约校验和
存在可观测性边界；更强的 bit 30 故障可被检测并无损恢复。

受控无故障测量中，仅开启通信校验和增加约 13.0% 的 step 时间，校验和 +
检测器 + 每步快照增加约 42.6%；真实故障还会额外重放一个 batch。完整协议、
结果表和目录说明见同级 `../ResNet-18/README.md`。

## 故障检测与恢复

通过以下配置启用缓解层：

```yaml
mitigation:
  enabled: true
  detector:
    non_finite: true
    grad_norm_ratio: 5.0
    loss_ratio: 3.0
  recovery:
    mode: step_rollback   # none | sanitize | step_rollback | checkpoint_rollback
    max_retries: 1
```

- `sanitize`：将非有限梯度清零（或裁剪）后继续训练。开销低，但对于较大的值故障并非无损。
- `step_rollback`：在每步开始时保存参数、优化器状态与 RNG；检测到故障后恢复并重放同一 batch。
  在重放确定性的前提下，瞬态故障可无损恢复；代价是故障步的一次额外前向/反向传播，
  以及模型和优化器状态的内存副本。
- `checkpoint_rollback`：恢复最近的磁盘检查点。调用方将训练步与数据迭代器回退到
  `checkpoint_step` 并重放其后的所有步骤时，可在检查点间隔内实现无损恢复；代价高于步级快照。
  第一次 `begin_step()` 会写入初始检查点；`--checkpoint_every N` 用于限制后续故障的重放距离。

所有 rank 通过一次 MAX 全归约对恢复决策达成一致，避免仅单 rank 回滚导致的死锁。
恢复事件会追加写入 `<log_dir>/recovery.jsonl`。

**检测边界。** 归约前的单 bit 翻转可能小于 float32 归约噪声，因此标量和校验和可能漏检。
非对称的归约后污染可通过跨 rank 一致性精确捕获。对于归约前的静默错误，应使用非有限值/
大幅值故障、梯度范数检测器或冗余重算。在没有复制梯度的情况下，这是固有的可观测性限制。

## 实验产物与结果分析

`run_ddp.py`、`run_fsdp.py` 与 `run_resnet18.py` 除 `recovery.jsonl` 外，还会写入
`<log_dir>/summary.json`。
该文件包含解析后的配置、各 rank 的注入记录、loss 曲线、已接受步骤耗时、峰值显存、
恢复事件、最终指标以及近似参数签名。

`run_resnet18.py` 还会在 summary 中写入完整模型与优化器状态的 SHA-256；
`examples/resnet18/analyze_resnet18.py` 用这两个哈希直接判定“严格无损恢复”，并生成中文报告。

将匹配的基线与故障用例放在不同目录中运行，然后进行聚合分析：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 20 \
  --log_dir runs/ddp_nan/seed42/baseline --baseline

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 20 \
  --log_dir runs/ddp_nan/seed42/fault

python examples/analyze_runs.py --root runs/ddp_nan/seed42
python examples/analyze_runs.py --root runs/ddp_nan/seed42 --json \
  --output runs/ddp_nan/report.json
```

文本报告会输出检测率/恢复率，并按故障模型、注入点、bit 区域、rank、bucket、时序和
缓解策略对结果分组，同时输出开销表。匹配基线由后端、模型、设备、world size、随机种子
和步数共同确定。故障专属分组不包含基线运行；`by_backend` 分组则包含基线，以展示参考总体。

| 结果 | 含义 |
|---|---|
| `baseline` | 无故障参考运行 |
| `no_fault` | 非基线运行，但未产生注入记录 |
| `recovered` | 已检测、回滚成功，且 loss/参数签名在容差内匹配基线 |
| `mitigated` | 已检测且无可测偏离，但未执行回滚（例如 `sanitize`） |
| `detected_unrecovered` | 已检测，但最终状态仍偏离基线 |
| `sdc` | 未检测到且存在可测偏离：静默数据损坏 |
| `masked` | 未检测到且无可测偏离 |
| `due` | 检测后运行中止 |
| `detected` / `undetected` | 无匹配基线时的降级分类 |

`time_ovh%` 与 `mem_ovh%` 相对于同后端的匹配基线行计算（优先选择相同缓解模式）。
已接受步骤的耗时已包含重试步骤的重放成本。`retried` 统计发生回滚的唯一 step 数；
`events` 统计每 rank 的恢复事件数，因此两 rank 的步级回滚通常产生两个事件。
`param_signature` 由带符号和、绝对值和、L2、位置加权校验和组成，是紧凑的近似比较，
并非逐 bit 的 state-dict 对照；实验需要严格无损证明时，应使用显式参考模型或完整 state dict。

## 配置

完整配置模式定义在 [`llmresfi/config.py`](llmresfi/config.py)（pydantic）。主要配置段如下：

| 配置段 | 字段 | 说明 |
|---|---|---|
| `run_mode` | `inference` / `training` | 选择阶段调度方式 |
| `model` | `repo`, `dtype`, `device` | Hugging Face 模型 ID 或本地路径 |
| `distributed` | `backend`, `inject_rank(s)`, `inject_rank_fraction`, `injection_point`, `bucket_index`, `fsdp_injection`, `verify_comm` | `none` / `ddp` / `fsdp`；rank 策略与通信检测控制 |
| `fault.target` | `weight/activation/kv_cache/loss/gradient/optimizer/master_weight/comm` | 需要污染的目标 |
| `fault.stage` | 取决于 `run_mode` | 注入发生的阶段 |
| `fault.fault_model` | `single/double/multi/sign/exponent/mantissa/random/random_value/zero/nan/inf/scale/stuck_at_0/stuck_at_1`，可选 `bit_idx` | 故障模型；`bit_idx` 可将位翻转固定到确定 bit |
| `fault.timing` | `transient/permanent/intermittent` + `steps`/`period` | 故障持续性 |
| `execution` | `random_seed`, `log_dir`, `verbosity` | 可复现性与日志记录 |
| `mitigation` | `enabled`, `detector`, `recovery` | 异常检测与清洗/回滚策略 |

## 编程接口

```python
from llmresfi import FIConfig, ModelInspector, StageTracker, FaultInjector, RecoveryManager

cfg = FIConfig.load("config.yml")
inspector = ModelInspector(model, optimizer)
tracker = StageTracker(cfg.run_mode)
injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
recovery = RecoveryManager(cfg, model, optimizer, tracker)

injector.arm(ddp_model)          # 通信目标需要传入包装后的模块
recovery.begin_step()
# ... 前向/反向传播，并推进 tracker ...
injector.inject_distributed_now()  # 仅旧版 FSDP 需要；DDP 下为空操作
signals = recovery.check(loss, injector=injector)
action = recovery.decide(signals)
if action == "retry_step":
    tracker.on_retry()
    # 重放同一 batch
optimizer.step()
recovery.end_step(loss, clean=not signals)
injector.disarm()
print(injector.records)          # 所有注入故障（可通过 gather_records 汇总各 rank）
```

## 仓库结构

```text
LLMResFI/
├── README.md
├── LICENSE
├── pyproject.toml
├── config.yml                     # 默认实验配置
├── llmresfi/
│   ├── __init__.py                # 公共 API
│   ├── config.py                  # YAML + pydantic 配置模式
│   ├── bitflip.py                 # 真实 IEEE 位翻转引擎
│   ├── fault_models.py            # 故障模型库
│   ├── scheduler.py               # StageTracker（推理/训练）
│   ├── inspector.py               # 层/模块/参数注册表
│   ├── hooks.py                   # 底层故障应用辅助函数
│   ├── injector.py                # FaultInjector（启用/停用）
│   ├── distributed.py             # DDP/FSDP 通信注入
│   ├── mitigation.py              # 检测、快照、回滚、检查点
│   ├── reporting.py               # summary.json 模式、结果聚合、开销统计
│   └── analysis.py                # 文本级 SDC/DUE/masked 分类
├── examples/
│   ├── run_inference.py
│   ├── run_training.py
│   ├── run_ddp.py
│   ├── run_fsdp.py
│   ├── analyze_runs.py
│   └── resnet18/
│       ├── README.md
│       ├── run_resnet18.py
│       └── analyze_resnet18.py
├── tests/
│   ├── test_bitflip.py
│   ├── test_config.py
│   ├── test_scheduler.py
│   ├── test_injector.py
│   ├── test_timing.py
│   ├── test_mitigation.py
│   ├── test_distributed_unit.py
│   ├── test_ddp_integration.py
│   ├── test_fsdp_gpu.py
│   └── test_reporting.py
└── docs/
    ├── design.md                  # 完整设计文档
    └── experiments.md             # 实验协议与指标
```

## 测试

```bash
pytest
CUDA_VISIBLE_DEVICES=0,1 pytest -q   # 同时运行双 GPU FSDP 测试
```

## 路线图

- **阶段 1**（v0.1 已完成）：单 GPU 推理，支持真实位翻转与时序故障。
- **阶段 2**（v0.1 已完成）：单 GPU 训练注入（梯度/优化器/主权重/loss）。
- **阶段 3**（v0.2 已完成）：DDP 归约前/后注入、rank 策略、校验和验证、
  FSDP 通信钩子以及 torch 1.12 分片降级。
- **阶段 4**（v0.2 部分完成）：异常检测、梯度清洗、无损步级回滚、检查点回滚、
  每 rank 指标；Campaign 编排与敏感性报告仍待后续实现。

完整设计见 [`docs/design.md`](docs/design.md)，实验协议见
[`docs/experiments.md`](docs/experiments.md)。

## 许可证

[Apache License 2.0](./LICENSE)
