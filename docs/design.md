# LLMResFI 设计方案

> **LLMResFI = LLM Resilience Fault Injection**（大语言模型可靠性故障注入工具）
> 一个面向大语言模型（LLM）训练与推理的、可扩展至多卡分布式（DDP/FSDP）的软错误故障注入框架。

---

## 0. 摘要

LLMResFI 是一个**模型层（应用级）**软错误故障注入工具，统一覆盖 LLM 的**推理**与**训练**两大场景，并以**真实 IEEE 位翻转**、**梯度/优化器状态故障**、**分布式通信故障**、**故障时序与传播建模**为核心创新点，区别于现有的 LLMFI、NVBitFI、PyTorchFI。

---

## 1. 背景与动机

### 1.1 现有工具缺口

| 维度 | LLMFI | NVBitFI | PyTorchFI | LLMResFI 目标 |
|---|---|---|---|---|
| 推理 | ✅ 权重/激活/KV，阶段感知 | ❌（非 ML 专用） | ✅ 权重/神经元 | ✅ 继承并增强 |
| 训练 | ❌ | ❌ | ❌ | ✅ 新增 |
| 梯度/优化器状态 | ❌ | ❌ | ❌ | ✅ 新增 |
| 分布式通信 | ❌ | ❌ | ❌ | ✅ 新增（DDP/FSDP） |
| 真 IEEE 位翻转 | ✅（single/double） | ✅（寄存器 XOR） | ⚠️ 量化翻转 | ✅ 定向位翻转 |
| 故障时序（瞬态/永久/间歇） | ⚠️ 仅瞬态、单故障/次 | ⚠️ 瞬态 + 永久(pf) | ❌ | ✅ 完整支持 |
| 抽象层级 | 模型层 | 架构层（最低） | 模型层 | 模型层 |
| 运行平台 | Python/HF | Linux + CUDA | Python/PyTorch | Python/PyTorch（跨平台） |

**结论**：训练阶段的软错误注入（梯度、Adam 状态、混合精度主权重、全归约通信）是当前空白，也是 LLMResFI 最大的创新空间。

### 1.2 设计目标

1. **训练 + 推理统一**：一套 API/配置，通过 `run_mode` + `stage` + `target` 三要素统一描述注入行为。
2. **覆盖训练关键状态**：梯度、优化器状态、混合精度、通信张量。
3. **支持分布式**：优先 DDP + FSDP，接口可扩展至张量/流水线并行。
4. **丰富故障模型**：位翻转（符号/指数/尾数定向）、值扰动、卡滞、时序类型。
5. **故障传播与时序建模**：持续性故障、步级传播、收敛影响分析。
6. **可复现**：随机种子 + 检查点重放 + 确定性解码。

---

## 2. 总体架构

分层设计，自顶向下：

```text
┌─────────────────────────────────────────────────────────────┐
│  运行器 / Campaign（实验编排、检查点、随机种子管理、并行执行）   │
├─────────────────────────────────────────────────────────────┤
│  配置（YAML + pydantic 校验，LLMFI 风格）                      │
├─────────────────────────────────────────────────────────────┤
│  检查器（目标注册表：层/参数/缓冲区/优化器状态/通信桶发现）      │
├─────────────────────────────────────────────────────────────┤
│  调度器（阶段 + 时序调度）                                    │
│        ├─ 阶段：prefill/first_token/decode（推理）           │
│        └─       forward/backward/optimizer_step/comm（训练）  │
│        └─ 时序：瞬态/永久/间歇                                │
├─────────────────────────────────────────────────────────────┤
│  注入引擎（故障模型库 + 真实位翻转引擎）                       │
├─────────────────────────────────────────────────────────────┤
│  钩子层（注入执行）                                           │
│        ├─ 权重/激活/KV（前向钩子）                           │
│        ├─ 损失（loss 张量）                                  │
│        ├─ 梯度（自动微分钩子）                               │
│        └─ 优化器（optimizer.step 补丁）                      │
├─────────────────────────────────────────────────────────────┤
│  分布式层（DDP/FSDP 通信钩子）                               │
├─────────────────────────────────────────────────────────────┤
│  报告层（summary.json + 基线对比 + 敏感性/开销聚合）           │
├─────────────────────────────────────────────────────────────┤
│  分析层（结果分类 SDC/masked/DUE + 收敛影响可视化）            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 故障模型库

每种故障由五元组描述：

```
Fault = { target_type, location, fault_model, timing, trigger_stage, bit_position? }
```

### 3.1 位翻转

| 类型 | 说明 |
|---|---|
| single / double / multi | 翻 1 / 2 / k 个比特 |
| sign | 仅翻符号位 |
| exponent | 仅翻指数位（对 fp 敏感度高） |
| mantissa | 仅翻尾数位 |
| random | 随机位 |

**关键实现**：真 IEEE 位翻转，通过 `tensor.view(int_dtype) ^ mask` 实现（区别于 PyTorchFI 的量化翻转）。

```python
class BitFlipper:
    _DTYPE_VIEW = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }
    @classmethod
    def flip(cls, t, bit_idxs):
        iv = t.view(cls._DTYPE_VIEW[t.dtype])
        mask = 0
        for b in bit_idxs:
            mask ^= (1 << b)
        iv ^= mask
        return iv.view(t.dtype)
```

### 3.2 值扰动

- `random_value(min, max)`
- `zero`（置零）
- `nan` / `inf`（注入非有限值）
- `scale(factor)`（乘性缩放）

### 3.3 卡滞

- `stuck_at_0` / `stuck_at_1`：将指定位强行置 0/1。

### 3.4 时序类型——核心创新之一

| 类型 | 语义 | 说明 |
|---|---|---|
| `transient` | 单次操作注入一次 | 传统默认 |
| `permanent` | 注入后持续 N 步 | 模拟永久/半永久硬件损伤 |
| `intermittent` | 每 k 步突发一次 | 模拟间歇性故障 |

---

## 4. 注入点矩阵（训练 + 推理统一）

### 4.1 推理注入点（`run_mode = inference`）

**阶段调度**：`prefill → first_token → decode×N`（沿用 LLMFI 的阶段跟踪思路，作为调度器的一等阶段）。

| 目标 | 注入机制 | 阶段 |
|---|---|---|
| 权重 | `register_forward_pre_hook`，瞬态恢复 | prefill/first_token/decode |
| 激活 | `register_forward_hook` | prefill/first_token/decode |
| KV cache | attention 模块 hook，随机选 K/V | decode |
| 损失/困惑度 | （可选）logits 后 hook | 任选 |

**推理部分相对 LLMFI 的增强**：
- 真位翻转 + sign/exponent/mantissa 定向（LLMFI 仅 single/double/random，exp/man 未实现）。
- 时序故障：可"注入后持续 N 个 decode step"（LLMFI 仅瞬态、单故障/次）。
- 多卡推理：张量/流水线并行的通信张量注入（进阶）。

### 4.2 训练注入点（`run_mode = training`）

**阶段调度**：`forward → backward → optimizer_step → comm`。

| 目标 | 注入机制 | 阶段 | 创新性 |
|---|---|---|---|
| 权重 | forward_pre_hook | forward | 与推理共享 |
| 激活 | forward_hook | forward | 与推理共享 |
| 损失 | loss tensor `.register_hook` | backward 前 | 新增 |
| **梯度** | 参数 `.grad` 的 `register_hook` / module `full_backward_hook` | backward | **新增（空白领域）** |
| **Adam 动量/方差** | patch `optimizer.step()` 注入 `exp_avg`/`exp_avg_sq` | optimizer_step | **新增** |
| **master weight(fp32)** | 混合精度下 fp32 副本注入 | optimizer_step | **新增** |
| GradScaler scale | 注入 `loss_scale` | 任意 | 新增 |
| **通信张量** | DDP all-reduce 钩子；FSDP reduce-scatter 钩子（torch≥2.0）或分片注入（torch 1.12） | comm | **新增** |

### 4.3 两模式共享机制

- 权重/激活的 hook 注入、`BitFlipper` 引擎、seed 复现、结果分类框架在训练/推理间复用。
- 差异点仅在于：推理专有 KV cache，训练专有梯度/优化器/混合精度/通信。

---

## 5. 分布式故障注入（DDP + FSDP）

### 5.1 DDP（数据并行）

- 通过 `model.register_comm_hook` 拦截梯度桶（gradient bucket）。
- 注入时机：
  - `before_reduce`：单卡梯度局部翻转，错误随后被平均进所有 rank（对称传播 / 静默污染）。
  - `after_reduce`：只翻转选定 rank 的归约结果，该 rank 与其余 rank 静默发散（非对称 SDC）。
- rank 选择：`inject_rank`（单 rank）、`inject_ranks`（显式集合）、
  `inject_rank_fraction`（由 `random_seed` 确定性采样的子集）。
- `bucket_index` 可定位到某一次 backward 中的特定梯度桶。
- 钩子自行执行 `all_reduce` 并按 `world_size` 做平均
  （PyTorch 文档要求自定义钩子负责该除法），避免 world size 改变训练语义。
- 可选 `verify_comm`：归约前 all-gather 每个 rank 的
  `(sum, sum|g|)` 检查量，归约后同时检查
  (a) 归约一致性 与 (b) 各 rank 结果一致性。

```python
def comm_hook(state, bucket):
    tensor = bucket.buffer()
    if inject_before:
        inject(tensor)
    fut = dist.all_reduce(tensor, async_op=True).get_future()
    def after(f):
        f.wait()
        tensor.div_(world_size)
        if inject_after:
            inject(tensor)
        if verify:
            check_checksums(tensor)
        return tensor
    return fut.then(after)
```

### 5.2 FSDP（ZeRO 分片）

- **PyTorch ≥ 2.0**：通过 `FSDP.register_comm_hook` 拦截归约前的完整
  未分片梯度；钩子执行 `reduce_scatter_tensor`（`NO_SHARD` 时
  `all_reduce`）并按 world size 做除法。支持 `before_reduce` /
  `after_reduce`，记录 `sharded=True`。
- **PyTorch 1.12**（无 FSDP comm hook）：降级为分片级注入：
  - `param_shard`：在前向前置钩子中翻转本地 `flat_param` 分片，随后被
    all-gather 到完整参数（模拟参数分片软错误）。
  - `grad_shard`：在 `optimizer.step` 前由
    `injector.inject_distributed_now()` 翻转本地分片梯度
    （模拟 reduce-scatter 后的梯度分片软错误）。
  - 记录 `legacy_fsdp=True`，明确区分于完整梯度注入。
- 显式步前注入保证检测/回滚发生在优化器更新之前。

### 5.3 通信校验与可观测性

- 校验和采用线性量 `sum(g)`；`sum|g|` 仅作为尺度估计，不能直接做等值比较
  （绝对值不满足线性）。
- `after_reduce` 的单 bit 翻转远低于 float32 归约噪声，但会被
  **跨 rank 一致性**检查精确捕获。
- `before_reduce` 的单 bit 翻转可能淹没在 float32 累加误差中；这是单次
  all-reduce 的观测极限，需要冗余（重算、复制桶或独立二次归约）才能严格检出。
- 每个故障记录包含：`rank`、`world_size`、`bucket_index`、`bucket_numel`、
  `dtype`、`flat_idx`、`bit_idxs`、`injection_point`、`backend`、
  `legacy_fsdp`。

### 5.4 检测与恢复

- **检测**：
  - 非有限梯度/参数（NaN/Inf）；
  - 梯度范数相对 EMA 的突增；
  - loss 相对 EMA 的突增或非有限；
  - DDP 校验和不一致。
- 所有 rank 通过一次 MAX all-reduce 对“是否发生故障”达成一致，避免单 rank
  回滚造成 DDP/FSDP 死锁。
- **恢复策略**：
  - `sanitize`：将非有限梯度置零并可做全局裁剪（低开销，非无损）；
  - `step_rollback`：步首保存参数/优化器/RNG 快照，故障时恢复并重放同一
    batch（瞬态故障下可无损，代价是快照内存 + 故障步的一次额外前后向）；
  - `checkpoint_rollback`：首次 `begin_step()` 自动保存初始检查点，之后按
    `checkpoint_every` 保存；恢复最近磁盘检查点（最多丢失一个检查点间隔）。
    恢复后必须把 tracker 与数据迭代器回退到 `checkpoint_step` 并重放之后的
    所有 step；只恢复参数但继续当前 step 语义上不等价于无损恢复。
- 事件写入 `<log_dir>/recovery.jsonl`，含触发信号、动作、重试次数。

### 5.5 可扩展性

- 通信钩子抽象为 `CommInjector` 接口，未来可扩展张量并行（TP）/流水线并行（PP）的通信张量注入。

---

## 6. 故障传播与时序建模

### 6.1 持续性故障

- `permanent`：注入后维持 N 步，追踪单点故障在训练步之间的累积传播。
- `intermittent`：周期突发，模拟现实间歇性硬件故障。

### 6.2 传播追踪

- 记录注入点（step、stage、rank、bucket、张量类型、bit 位置），
  支持跨 rank `all_gather` 汇总（`distributed.gather_records`）。
- 追踪下游指标：loss 曲线、梯度范数、最终准确率、PPL 变化；
  `distributed.all_gather_metrics` 提供每 rank 的 loss / grad norm。
- 输出故障→结果的因果链，支持"某 bit 位置/层/阶段"的敏感性排序。

### 6.3 可复现

- `random_seed` 统一控制：目标选择、bit 选择、时序选择、rank 子集选择。
- 训练：步级参数/优化器/RNG 快照重放；注入前后状态可复现。
- 推理：贪心解码（argmax）保证确定性（沿用 LLMFI）。

---

## 7. 结果分类与评估指标

### 7.1 结果分类（借鉴 NVBitFI，适配 LLM）

`reporting.classify_summary` 同时使用检测/恢复记录和同配置基线做分类。基线按
后端、模型、设备、world size、随机种子、步数匹配；没有基线时只能给出
已检测 / 未检测。

| 类别 | 判定 | 说明 |
|---|---|---|
| `baseline`（基线） | 运行标记为基线 | 无故障参考运行 |
| `no_fault`（无故障） | 非基线且无注入记录 | 检查时序/阶段/rank/bucket 配置，不应作为故障结果统计 |
| `recovered`（已恢复） | 检测 + 成功回滚 + loss/参数签名在容差内匹配基线 | 可称为“在给定容差内无损恢复” |
| `mitigated`（已缓解） | 检测 + 未回滚 + 无显著差异 | 典型为 `sanitize`；低开销但不保证严格无损 |
| `detected_unrecovered`（已检测未恢复） | 检测到但最终状态仍偏离基线 | 恢复逻辑、检查点重放或持续故障仍需排查 |
| SDC | 未检测到但最终状态偏离基线 | 静默数据损坏，是最需要关注的可靠性失效 |
| `masked`（已掩蔽） | 未检测到且无可测偏离 | 可能命中无关元素、被数值误差吸收或被粗粒度指标掩盖 |
| DUE | 检测后 abort | 重试耗尽、无快照/检查点等可检测错误 |
| `detected` / `undetected`（已检测/未检测） | 无匹配基线时的降级分类 | 只能评估可观测性，不能确认是否发生 SDC |

### 7.2 任务指标

- **推理**：PPL、下游任务准确率（如 BoolQ）、生成文本 diff。
- **训练**：收敛速度、最终准确率、loss 相对差、梯度范数异常。

### 7.3 敏感性分析

- bit 位置（符号、指数、尾数）→ 故障影响排序。
- 层敏感性（浅层与深层）、阶段敏感性（前向、反向、优化器）。

### 7.4 实验产物与聚合分析

- `run_ddp.py` / `run_fsdp.py` 在每个运行目录写 `summary.json`，包含解析后的
  配置、所有 rank 的注入记录、loss 曲线、每步耗时、峰值显存、恢复事件、
  最终梯度/参数签名和近似参数指纹。
- `examples/analyze_runs.py --root <campaign_dir>` 聚合所有 summary，输出
  `recovered` / `mitigated` / `SDC` / `masked` / `DUE` 计数与比例，并按
  故障模型、注入点、bit 区域、rank、bucket、时序、缓解策略分组；
  同时输出相对基线的每步耗时与峰值显存开销。
- `summary.json` 中的参数指纹由 signed sum、absolute sum、L2、位置加权和、
  max-abs 组成，采用分块 float64 计算以限制额外内存。它是紧凑的近似比较，
  不能替代完整 state-dict 的逐元素或逐比特对照。需要严格“无损”结论时，应
  保存最终状态字典或使用独立参考模型做精确比较。
- 同一实验目录只能保存一个 summary；baseline 与各故障配置应使用不同
  `--log_dir`，分析时放在同一 campaign 根目录下。

---

## 8. API 设计

### 8.1 配置（YAML + pydantic，LLMFI 风格）

```yaml
run_mode: training            # inference | training

model:
  repo: "meta-llama/Llama-3.2-1B"
  dtype: "bfloat16"
  device: "cuda"

distributed:
  backend: "fsdp"             # none | ddp | fsdp
  world_size: 4
  inject_rank: 0              # 单个 rank（null 表示使用下方策略字段）
  inject_ranks: null          # 显式子集，例如 [0, 2]
  inject_rank_fraction: 1.0   # 确定性随机子集；1.0 表示所有 rank
  injection_point: "before_reduce"  # before_reduce | after_reduce
  bucket_index: null          # 每次反向传播仅定位一个梯度桶
  fsdp_injection: "grad_shard"  # torch 1.12 降级：param_shard | grad_shard | both
  verify_comm: false          # 校验和守卫（DDP）

fault:
  target: "gradient"          # weight|activation|kv_cache|loss|gradient|optimizer|master_weight|comm
  locate: "random"            # random | manual
  layer_idx: null
  fault_model:
    type: "exponent"          # single|double|sign|exponent|mantissa|random|stuck_at|...
    bit_idx: null
  timing:
    type: "permanent"         # transient | permanent | intermittent
    steps: 10                 # permanent 持续步数
    period: 3                 # intermittent 周期
  stage:
    type: "backward"          # prefill|first_token|decode|forward|backward|optimizer_step|comm

execution:
  random_seed: 42
  log_dir: "./runs/exp1"
  one_fault_per_run: true

mitigation:
  enabled: true
  detector:
    non_finite: true
    check_parameters: true
    grad_norm_ratio: 5.0
    loss_ratio: 3.0
    warmup_steps: 5
  recovery:
    mode: "step_rollback"     # none | sanitize | step_rollback | checkpoint_rollback
    max_retries: 1
    sanitize: "zero"          # zero | clamp
    clip_norm: null
    snapshot_every: 1
    checkpoint_every: 0
    checkpoint_dir: null
    keep_checkpoints: 2
```

### 8.2 Python API（统一入口）

```python
from llmresfi import (
    FIConfig, FaultInjector, ModelInspector, RecoveryManager, StageTracker,
)

cfg = FIConfig.load("config.yml")
inspector = ModelInspector(model, optimizer)
tracker = StageTracker(cfg.run_mode)
injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
recovery = RecoveryManager(cfg, model, optimizer, tracker)

injector.arm(ddp_model)     # 通信目标必须传入 DDP/FSDP 包装后的模型
recovery.begin_step()
# ... 前向 / 反向传播 ...
injector.inject_distributed_now()   # torch 1.12 FSDP 需要；DDP 下为空操作
signals = recovery.check(loss, injector=injector)
action = recovery.decide(signals)    # continue / retry_step / abort
if action == "retry_step":
    tracker.on_retry()
    # 重放同一 batch
optimizer.step()
recovery.end_step(loss, clean=not signals)
injector.disarm()
```

---

## 9. 路线图

| 阶段 | 内容 | 里程碑 |
|---|---|---|
| **阶段 1** | 单卡推理：对齐并超越 LLMFI（真实位翻转 + 符号/指数/尾数定向 + 时序） | 推理完整可用 |
| **阶段 2** | 单卡训练：梯度/优化器/混合精度/损失注入 | 训练完整可用 |
| **阶段 3** | 分布式：DDP 归约前/后注入 + rank/bucket 策略 + 校验和；FSDP 通信钩子 + torch 1.12 分片降级 | **v0.2 已完成，多卡可用** |
| **阶段 4** | 检测/清洗/步级回滚/检查点回滚 + 每 rank 指标 | **v0.2 部分完成** |
| **阶段 5** | 时序传播建模 + Campaign 编排 + 敏感性分析报告 | 完整工具发布 |

---

## 10. 与现有工具差异化对照

| 能力 | LLMFI | NVBitFI | PyTorchFI | **LLMResFI** |
|---|---|---|---|---|
| 训练支持 | ❌ | ❌ | ❌ | ✅ |
| 梯度/优化器故障 | ❌ | ❌ | ❌ | ✅ |
| 分布式通信故障 | ❌ | ❌ | ❌ | ✅ |
| 真 IEEE 定向位翻转 | ⚠️ 部分 | ✅ | ❌ | ✅ |
| 故障时序（瞬态/永久/间歇） | ❌ | ⚠️ | ❌ | ✅ |
| 传播与时序建模 | ❌ | ❌ | ❌ | ✅ |
| 推理阶段感知 | ✅ | ❌ | ❌ | ✅ |

---

## 11. 参考

- LLMFI：Huang et al., "Not All Errors Are Equal", ICS'26。
- NVBitFI：NVIDIA, "An Architecture-level Fault Injection Tool"（基于 NVBit）。
- PyTorchFI：Mahmoud et al., "PyTorchFI: A Runtime Perturbation Tool for DNNs", DSN-W'20。
