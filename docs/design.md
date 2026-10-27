# LLMResFI 设计方案

> **LLMResFI = LLM Resilience Fault Injection**（大语言模型可靠性故障注入工具）
> 一个面向大语言模型（LLM）训练与推理的、可扩展至多卡分布式（DDP/FSDP）的软错误故障注入框架。

---

## 0. 摘要

LLMResFI 是一个**模型层（application-level）**软错误故障注入工具，统一覆盖 LLM 的**推理**与**训练**两大场景，并以**真 IEEE 位翻转**、**梯度/优化器状态故障**、**分布式通信故障**、**故障时序与传播建模**为核心创新点，区别于现有的 LLMFI、NVBitFI、PyTorchFI。

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

**结论**：训练阶段的软错误注入（梯度、Adam 状态、混合精度 master weight、all-reduce 通信）是当前空白，也是 LLMResFI 最大的创新空间。

### 1.2 设计目标

1. **训练 + 推理统一**：一套 API/配置，通过 `run_mode` + `stage` + `target` 三要素统一描述注入行为。
2. **覆盖训练关键状态**：梯度、优化器状态、混合精度、通信张量。
3. **支持分布式**：优先 DDP + FSDP，接口可扩展至张量/流水线并行。
4. **丰富故障模型**：位翻转（sign/exponent/mantissa 定向）、值扰动、卡滞、时序类型。
5. **故障传播与时序建模**：持续性故障、步级传播、收敛影响分析。
6. **可复现**：seed + checkpoint 重放 + 确定性解码。

---

## 2. 总体架构

分层设计，自顶向下：

```text
┌─────────────────────────────────────────────────────────────┐
│  Runner / Campaign（实验编排、checkpoint、seed 管理、并行执行） │
├─────────────────────────────────────────────────────────────┤
│  Config（YAML + pydantic 校验，LLMFI 风格）                    │
├─────────────────────────────────────────────────────────────┤
│  Inspector（目标注册表：层/参数/buffer/优化器状态/通信桶发现）   │
├─────────────────────────────────────────────────────────────┤
│  Scheduler（阶段 + 时序调度器）                               │
│        ├─ 阶段：prefill/first_token/decode（推理）           │
│        └─       forward/backward/optimizer_step/comm（训练）  │
│        └─ 时序：瞬态/永久/间歇                                │
├─────────────────────────────────────────────────────────────┤
│  Injection Engine（故障模型库 + 真位翻转引擎）                 │
├─────────────────────────────────────────────────────────────┤
│  Hooks（注入执行层）                                          │
│        ├─ 权重/激活/KV（forward hook）                       │
│        ├─ 损失（loss tensor）                               │
│        ├─ 梯度（autograd hook）                             │
│        └─ 优化器（optimizer.step 补丁）                      │
├─────────────────────────────────────────────────────────────┤
│  Distributed Layer（DDP/FSDP 通信钩子）                      │
├─────────────────────────────────────────────────────────────┤
│  Analysis（结果分类 SDC/masked/DUE + 敏感性分析 + 收敛影响）    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 故障模型库

每种故障由五元组描述：

```
Fault = { target_type, location, fault_model, timing, trigger_stage, bit_position? }
```

### 3.1 位翻转（Bit Flip）

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

### 3.2 值扰动（Value Perturbation）

- `random_value(min, max)`
- `zero`（置零）
- `nan` / `inf`（注入非有限值）
- `scale(factor)`（乘性缩放）

### 3.3 卡滞（Stuck-at）

- `stuck_at_0` / `stuck_at_1`：将指定位强行置 0/1。

### 3.4 时序类型（Timing）——核心创新之一

| 类型 | 语义 | 说明 |
|---|---|---|
| `transient` | 单次操作注入一次 | 传统默认 |
| `permanent` | 注入后持续 N 步 | 模拟永久/半永久硬件损伤 |
| `intermittent` | 每 k 步突发一次 | 模拟间歇性故障 |

---

## 4. 注入点矩阵（训练 + 推理统一）

### 4.1 推理注入点（`run_mode = inference`）

**阶段调度**：`prefill → first_token → decode×N`（沿用 LLMFI 的 PhaseTracker 思路，作为 scheduler 的一等阶段）。

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
| **通信张量** | DDP/FSDP 通信钩子 | comm | **新增** |

### 4.3 两模式共享机制

- 权重/激活的 hook 注入、`BitFlipper` 引擎、seed 复现、结果分类框架在训练/推理间复用。
- 差异点仅在于：推理专有 KV cache，训练专有梯度/优化器/混合精度/通信。

---

## 5. 分布式故障注入（DDP + FSDP）

### 5.1 DDP（数据并行）

- 通过 `model.register_comm_hook` 拦截梯度桶（gradient bucket）。
- 注入时机：
  - **all-reduce 前**：单卡梯度局部翻转，导致各 rank 梯度不一致。
  - **all-reduce 后**：翻转同步后的统一梯度。
- 支持"不一致注入"（仅部分 rank 注入），研究数据并行下的错误传播差异。

```python
def comm_hook(state, bucket):
    # 在 all_reduce 前/后对 bucket 张量注入
    tensor = bucket.buffer()
    inject(tensor)  # 可选 before/after
    fut = torch.distributed.all_reduce(tensor, async_op=True)
    return fut
```

### 5.2 FSDP（ZeRO 分片）

- 钩子点：`post_backward_hook`（梯度归约后）、`reshard`（分片回传）、`gather`（前向参数聚合）。
- 注入对象：分片参数、聚合后的 full param、梯度 shard。
- 目标：研究 ZeRO 分片/重分片过程中的数据损坏影响。

### 5.3 可扩展性

- 通信钩子抽象为 `CommInjector` 接口，未来可扩展张量并行（TP）/流水线并行（PP）的通信张量注入。

---

## 6. 故障传播与时序建模

### 6.1 持续性故障

- `permanent`：注入后维持 N 步，追踪单点故障在训练步之间的累积传播。
- `intermittent`：周期突发，模拟现实间歇性硬件故障。

### 6.2 传播追踪

- 记录注入点（step, stage, layer, tensor type, bit position）。
- 追踪下游指标：loss 曲线、梯度范数、最终准确率、PPL 变化。
- 输出故障→结果的因果链，支持"某 bit 位置/层/阶段"的敏感性排序。

### 6.3 可复现

- `random_seed` 统一控制：目标选择、bit 选择、时序选择。
- 训练：step 级 checkpoint 重放，注入前后状态可复现。
- 推理：贪心解码（argmax）保证确定性（沿用 LLMFI）。

---

## 7. 结果分类与评估指标

### 7.1 结果分类（借鉴 NVBitFI，适配 LLM）

| 类别 | 说明 |
|---|---|
| Masked | 故障被覆盖（写入前被读/未读取） |
| DUE | 可检测错误（NaN 崩溃、loss 爆炸、assert 失败、超时） |
| SDC | 静默数据损坏（输出错误但程序正常结束） |
| Detected | 通过 ECC/检查机制显式报错 |

### 7.2 任务指标

- **推理**：PPL、下游任务准确率（如 BoolQ）、生成文本 diff。
- **训练**：收敛速度、最终准确率、loss 相对差、梯度范数异常。

### 7.3 敏感性分析

- bit 位置（sign vs exponent vs mantissa）→ 故障影响排序。
- 层敏感性（浅层 vs 深层）、阶段敏感性（forward vs backward vs optimizer）。

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
```

### 8.2 Python API（统一入口）

```python
from llmresfi import FaultInjector, ModelInspector, StageTracker, FIConfig

cfg = FIConfig.load("config.yml")
inspector = ModelInspector(model)          # 发现层/参数/buffer/优化器状态
tracker = StageTracker(cfg.run_mode)       # 推理/训练阶段跟踪
injector = FaultInjector(cfg, inspector, tracker, rng)

injector.arm()      # 注册 hook / 补丁 / 通信钩子
# ... 运行训练或推理 ...
injector.disarm()   # 移除 hook + 恢复权重
```

---

## 9. Roadmap

| 阶段 | 内容 | 里程碑 |
|---|---|---|
| **Phase 1** | 单卡推理：对齐并超越 LLMFI（真位翻转 + sign/exp/man 定向 + 时序） | 推理完整可用 |
| **Phase 2** | 单卡训练：梯度/优化器/混合精度/损失注入 | 训练完整可用 |
| **Phase 3** | 分布式：DDP + FSDP 通信钩子 | 多卡训练可用 |
| **Phase 4** | 时序传播建模 + Campaign 编排 + 敏感性分析报告 | 完整工具发布 |

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
