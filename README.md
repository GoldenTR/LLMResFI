# LLMResFI（LLM Resilience Fault Injector）

LLMResFI 是一个独立的 PyTorch Tensor-level 软错误注入工具，当前版本为
`0.1.0`。它面向混合精度大语言模型训练中的故障注入、检测算法评估、局部恢复
实验、敏感性分析与消融实验，提供可控、可复现、可追踪的 IEEE-754 位翻转。

> **实验边界：** LLMResFI 修改 PyTorch Tensor 的软件位模式。它不模拟真实 GPU
> SRAM、DRAM、L2 cache、寄存器、Tensor Core 或计算单元中的物理 Single Event
> Upset（SEU），也不声称能够复现真实硬件的时序、空间相关性或微架构传播路径。
> 相关结论只能表述为“Tensor-level 数据损坏实验”。

## 研究背景与用途

LLMResFI 把故障注入拆成三个互不泄漏的职责：

1. **注入执行**：在指定 step、指定 Tensor、指定逻辑元素和 bit 上执行位翻转。
2. **审计日志**：独立记录 ground truth，包括成功和失败事件。
3. **训练可观测张量**：训练程序只把损坏后的 Tensor 送入正常计算路径。

后续 PARS/检测模块只能读取模型激活、梯度、损失、优化器统计等训练可观测量，
**不得读取** `fault_events.jsonl`、`fault_events.csv`、`InjectionResult` 或
`EventLogger.events`。这些对象含有注入位置和 bit 类型，属于评估标签。

## 已验证环境

环境审计详见 [docs/environment.md](docs/environment.md)，完整机器可读报告可由
`scripts/environment_check.py` 重新生成。

| 项目 | 实际结果 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS, Linux 6.8.0-124-generic |
| CPU | Intel Xeon Silver 4410Y, 2 sockets × 12 cores × 2 threads, 48 logical CPUs |
| Conda 环境 | `llmresfi` (`/data2/lijinshu/miniconda3/envs/llmresfi`) |
| Python | 3.10.22 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| 系统 NVCC | CUDA 12.2, V12.2.140 |
| cuDNN | 90100 |
| `torch.cuda.is_available()` | `True` |
| CUDA device count | `8` |
| GPU | 8 × NVIDIA GeForce RTX 4090，49140 MiB/卡，compute capability 8.9 |
| NVIDIA Driver | 535.309.01 |

沙箱内默认不可访问 `/dev/nvidia*`，因此 CUDA 检查必须在沙箱外执行。本次已在沙箱外
通过 `nvidia-smi` 和 PyTorch 确认真实 GPU，并使用 `CUDA_VISIBLE_DEVICES=1` 在空闲
GPU 1 上完成 CUDA 测试。

## 安装

所有 Python 操作必须在 `llmresfi` Conda 环境执行。建议使用：

```bash
conda run -n llmresfi python -m pip install -e . --no-deps --no-build-isolation
```

项目运行依赖为 `torch>=2.0` 和 `PyYAML>=6.0`，测试依赖为 `pytest>=7.0`。
当前环境已具备这些包；不要为了安装 LLMResFI 而升级现有 PyTorch/CUDA。

检查环境：

```bash
conda run -n llmresfi python scripts/environment_check.py \
  --json logs/environment_audit.json \
  --markdown docs/environment.md
```

## 快速开始

```python
import torch
from llmresfi import inject_bit_flip, pattern_to_binary

device = torch.device("cuda")
x = torch.tensor([1.0, -2.5, 0.0], dtype=torch.float16, device=device)
result = inject_bit_flip(
    x,
    flat_index=1,
    bit_position=10,     # FP16 exponent LSB
    inplace=False,
    tensor_name="example.vector",
)

print(result.tensor)                    # tensor([ 1.0000, -5.0000,  0.0000], dtype=torch.float16)
print(result.original_value)           # -2.5
print(result.corrupted_value)          # -5.0
print(result.bit_type)                 # BitType.EXPONENT
print(pattern_to_binary(result.original_bit_pattern, result.dtype))   # 1100000100000000
print(pattern_to_binary(result.corrupted_bit_pattern, result.dtype))  # 1100010100000000
print(result.only_target_bit_changed)  # True
```

可直接运行：

```bash
# 默认使用 CUDA
conda run -n llmresfi python examples/basic_bit_flip.py

# 显式使用 CPU
conda run -n llmresfi python examples/basic_bit_flip.py --device cpu
```

## Bit 编号与浮点格式

所有 dtype 使用统一编号：bit `0` 是 LSB，最高位是符号位；尾数位于低位，指数位
位于中间。

| dtype | 总位宽 | Sign | Exponent | Mantissa |
|---|---:|---|---|---|
| FP16 | 16 | bit 15 | bits 10–14 | bits 0–9 |
| BF16 | 16 | bit 15 | bits 7–14 | bits 0–6 |
| FP32 | 32 | bit 31 | bits 23–30 | bits 0–22 |

```python
from llmresfi import get_bit_type, get_float_format
import torch

assert get_bit_type(torch.bfloat16, 7).value == "exponent"
assert get_bit_type(torch.float32, 31).value == "sign"
assert get_float_format(torch.float16).mantissa_bits == 10
```

实现通过 `.view(torch.int16/int32)` 获取底层 IEEE-754 位模式并执行 XOR，不会先把
浮点数值转换成整数。符号位、指数位翻转可能产生 `Inf`、`NaN`、零或极大值，这是
预期结果，工具不会过滤这些数值。

## FP16 注入

```python
import torch
from llmresfi import inject_bit_flip

device = torch.device("cuda")
x = torch.tensor([-2.5], dtype=torch.float16, device=device)
for bit in (0, 10, 15):  # mantissa / exponent / sign
    result = inject_bit_flip(x, 0, bit)
    print(bit, result.bit_type.value, result.original_value, result.corrupted_value)
```

```bash
conda run -n llmresfi python examples/fp16_injection.py
conda run -n llmresfi python examples/fp16_injection.py --device cpu
```

## BF16 注入

```python
import torch
from llmresfi import inject_bit_flip

device = torch.device("cuda")
x = torch.tensor([-2.5], dtype=torch.bfloat16, device=device)
result = inject_bit_flip(x, 0, 7)  # BF16 exponent LSB
assert result.bit_type.value == "exponent"
```

```bash
conda run -n llmresfi python examples/bf16_injection.py
conda run -n llmresfi python examples/bf16_injection.py --device cpu
```

## FP32 注入

```python
import torch
from llmresfi import inject_bit_flip

device = torch.device("cuda")
x = torch.tensor([-2.5], dtype=torch.float32, device=device)
result = inject_bit_flip(x, 0, 31)  # sign
assert result.corrupted_value == 2.5
```

```bash
conda run -n llmresfi python examples/fp32_injection.py
conda run -n llmresfi python examples/fp32_injection.py --device cpu
```

## 模型参数注入

`select_target_tensor()` 支持精确参数名、dtype 过滤、正则名称匹配和本地 seeded
随机选择，并返回逻辑名称、shape、dtype、device、stride、storage offset 等元信息。

```python
import torch
from llmresfi import inject_bit_flip, select_target_tensor

device = torch.device("cuda")
model = torch.nn.Linear(8, 4).to(device)
target = select_target_tensor(
    model,
    target_name="weight",
    dtype="float32",
)
print(target.name, target.shape, target.dtype, target.device)

result = inject_bit_flip(
    target.tensor,
    flat_index=0,
    bit_position=10,
    inplace=True,       # 叶子参数可安全原地修改
    tensor_name=target.name,
    step=0,
)
print(result.original_value, result.corrupted_value)
```

完整示例会检查故障是否真正改变前向输出：

```bash
conda run -n llmresfi python examples/model_parameter_injection.py
conda run -n llmresfi python examples/model_parameter_injection.py --device cpu
```

## 指定训练 Step 注入

训练 step 约定为 **zero-based**，与 `for step, batch in enumerate(dataloader)` 一致。
第一次返回 `True` 的 `should_inject(step)` 会“认领”该事件；同一 step 再次调用不会
重复触发。

```python
from llmresfi import FaultScheduler

scheduler = FaultScheduler(
    injection_steps=[100, 200],
    max_faults=2,
    seed=42,
)

for step in range(201):
    if scheduler.should_inject(step):
        print("inject at", step)
```

支持：

```python
FaultScheduler(injection_steps=100)
FaultScheduler(injection_steps=[100, 200], max_faults=1, seed=42)
FaultScheduler(step_range=(100, 200), step_interval=10)
FaultScheduler(step_range=(100, 200), random_steps=5, seed=42)
FaultScheduler(step_range=(100, 500), max_faults=20, seed=42)
```

## 随机故障注入

随机计划使用局部 `random.Random(seed)`，不读取或修改 Python/PyTorch 全局随机状态。
默认每个逻辑元素最多被选中一次。

```python
import torch
from llmresfi import inject_random_bit_flips, plan_random_injections

weight = torch.ones((64, 64), dtype=torch.float16)
plan = plan_random_injections(weight, count=4, seed=42, bit_type="mantissa")
print([(item.flat_index, item.bit_position) for item in plan])

results = inject_random_bit_flips(
    weight, count=4, seed=42, bit_type="mantissa", inplace=False
)
assert len(results) == 4
assert all(item.only_target_bit_changed for item in results)
```

需要指定多个元素各自翻转同一 bit 时，可使用 `inject_bit_flips()`：

```python
from llmresfi import inject_bit_flips

results = inject_bit_flips(weight, [(0, 10), (10, 10), (100, 10)], inplace=False)
```

## 训练循环集成

训练循环示例位于 `examples/training_loop_integration.py`，默认使用 CUDA BF16，
可通过 `--device cpu` 回退到 CPU。它实际执行 forward、backward、AdamW step，并
验证注入不会破坏正常 autograd 流程。

```bash
conda run -n llmresfi python examples/training_loop_integration.py \
  --mode persistent_low_precision --dtype bfloat16 \
  --steps 6 --injection-steps 3

conda run -n llmresfi python examples/training_loop_integration.py \
  --mode transient_compute_copy --dtype bfloat16 \
  --steps 6 --injection-steps 3

conda run -n llmresfi python examples/training_loop_integration.py \
  --mode amp_fp32_parameter --dtype bfloat16 \
  --steps 6 --injection-steps 3

# CPU 回退：在任一命令后追加 --device cpu
conda run -n llmresfi python examples/training_loop_integration.py \
  --mode persistent_low_precision --device cpu --dtype bfloat16 \
  --steps 6 --injection-steps 3
```

标准流程：

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    if scheduler.should_inject(step):
        target = select_target_tensor(model, dtype="float32", seed=step)
        original_snapshot = target.tensor.detach().clone()
        spec = plan_random_injections(
            target.tensor, count=1, seed=step, bit_type="mantissa"
        )[0]
        result = inject_bit_flip(
            target.tensor,
            spec.flat_index,
            spec.bit_position,
            inplace=True,           # 仅在安全的叶子参数/阶段使用
            tensor_name=target.name,
            step=step,
            parameter_version=step,
            injection_phase="before_forward_amp_fp32_parameter",
        )
        logger.log(result, original_tensor=original_snapshot)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

### 混合精度数据路径的关键区别

**1. 标准 AMP：FP32 参数 + autocast**

PyTorch 通常持久化 FP32 参数；autocast 在算子执行时把操作数临时转换为 FP16/BF16。
这些低精度操作数通常是临时 Tensor，不是持久化权重副本。直接对 `model.weight`
注入修改的是 FP32 存储；该破坏会在后续 forward 中参与 autocast 转换，但它不是
一次“对 FP16 持久化权重的 bit flip”。若研究目标必须是实际低精度计算操作数，应在
自定义算子/forward hook 中获得该低精度 Tensor，并注入其返回结果；不要假设标准
AMP 必然存在可修改的持久化 FP16/BF16 权重。

**2. 纯低精度训练：持久化低精度参数**

```python
model = model.to(dtype=torch.bfloat16)
target = select_target_tensor(model, target_name="lm_head.weight")
inject_bit_flip(target.tensor, flat_index=123, bit_position=4, inplace=True)
```

此时参数就是持久化 BF16 Tensor。注入发生在 forward 前会立即影响前向；优化器更新
后该损坏可能被改写。用于“持续故障”时，应在明确的生命周期内保持并管理损坏状态，
而不是每一步无记录地重复翻转。

**3. 自定义低精度 compute weight 副本**

```python
compute_weight = master_weight.to(dtype=torch.float16)  # 可微转换
result = inject_bit_flip(compute_weight, 0, 10, inplace=False)
logits = torch.nn.functional.linear(hidden.to(torch.float16), result.tensor)
```

`inplace=False` 返回值仍与 autograd 图兼容，可送入后续计算。若副本每步重建，故障
只影响当前 step；若框架缓存副本，则故障会跨 step 持续，直到同步或恢复逻辑覆盖。

### 注入阶段

- forward 前修改持久化参数：影响当前 forward、backward 和优化器更新。
- forward 后/backward 前对非叶子激活原地修改：可能使 backward 报版本错误；工具
  默认拒绝这类 `inplace=True`，应改为 `inplace=False` 并替换下游输入。
- backward 后、optimizer step 前：影响参数更新方向或梯度状态，但已完成的 forward
  不受影响。
- optimizer step 后：通常只影响下一步，直到参数再次被更新。

## 故障日志

```python
import torch
from llmresfi import EventLogger, inject_bit_flip

tensor = torch.ones((4, 4), dtype=torch.float16)
original_snapshot = tensor.detach().clone()
result = inject_bit_flip(
    tensor, flat_index=3, bit_position=10, inplace=False,
    parameter_version=100, injection_phase="before_forward",
)

logger = EventLogger(
    experiment_id=None,       # None -> 自动生成唯一 experiment_id
    output_dir="logs/auto_exp",
    format="jsonl",           # jsonl | csv | both
    save_tensor_backup=False,
    random_seed=42,
)
event = logger.log(
    result,
    original_tensor=original_snapshot,
    step=100,
    parameter_version=100,
    injection_phase="before_forward",
)
logger.log_failure(
    step=101,
    tensor_name="lm_head.weight",
    failure_reason="flat_index out of range",
    flat_index=10**12,
)
```

每个事件至少包含：

`experiment_id`, `fault_id`, `step`, `tensor_name`, `tensor_shape`,
`tensor_dtype`, `tensor_device`, `flat_index`, `bit_position`, `bit_type`,
`original_value`, `corrupted_value`, `injection_timestamp`, `random_seed`,
`injection_success`, `failure_reason`, `parameter_version`, `injection_phase`，
并额外记录精确的 `original_bit_pattern`、`corrupted_bit_pattern`、
`tensor_version_before/after` 和 `injection_time_ns`。

日志目录还会写入 `run_metadata.json`，包含工具/Python/PyTorch/CUDA/GPU 探测
信息。配置实验可调用 `write_run_metadata()` 保存解析后的配置、scheduler plan
和原始 YAML 副本。

**日志是实验 ground truth。检测/恢复算法不得读取日志、`InjectionResult` 或 logger
内部状态，也不得从 Tensor 属性反查注入位置。**

## YAML 实验配置

示例见 [configs/example_fault.yaml](configs/example_fault.yaml)：

```yaml
experiment:
  experiment_id: exp_001
  seed: 42
fault:
  enabled: true
  target_type: model_parameter
  target_name: transformer.h.0.attn.c_attn.weight
  dtype: float16
  injection_mode: single_bit
  bit_type: exponent
  injection_step: 100
  injection_count: 1
  selection_mode: random_element
logging:
  enabled: true
  format: jsonl
  output_dir: logs/exp_001
```

```python
from llmresfi import load_config, write_run_metadata

config = load_config("configs/example_fault.yaml")
scheduler = config.build_scheduler()
write_run_metadata(config, config.logging.output_dir)
print(config.experiment.experiment_id, scheduler.injection_steps)
```

未提供 `experiment_id` 时，加载器会生成带 UTC 时间戳和随机 suffix 的唯一 ID。相同
配置、相同 seed、相同工具版本会生成相同注入位置和 step 计划；事件时间戳和日志
文件名不参与随机计划。

## 测试

测试默认在 CUDA Tensor 上执行；使用 `--device=cpu` 显式运行 CPU 路径，此时
`tests/test_cuda.py` 会自动 skip。

```bash
# 默认 CUDA
CUDA_VISIBLE_DEVICES=1 conda run -n llmresfi python -m pytest -q
CUDA_VISIBLE_DEVICES=1 conda run -n llmresfi python -m pytest -q -m cuda

# CPU 回退
conda run -n llmresfi python -m pytest -q --device=cpu
```

本次实际结果：

- 总测试用例：123
- 通过：123
- 失败：0
- 跳过：0
- CUDA 专项测试：`6 passed`（在 GPU 1 上执行）
- CPU 回退模式：`117 passed, 6 skipped`（CUDA 专项测试自动跳过）

测试覆盖三种 dtype 的 sign/exponent/mantissa、边界 bit、NaN/Inf/0/负数、原地与
非原地、CPU、非连续切片与共享存储、调度去重、随机可复现、JSONL/CSV 日志、失败
事件、配置快照、AdamW 训练集成。CUDA 测试已在真实 RTX 4090 上完成 FP16、BF16、
FP32 位模式验证和非连续 Tensor 原地注入验证。

## 项目结构

```text
LLMResFI/
├── llmresfi/
│   ├── __init__.py
│   ├── _version.py
│   ├── bit_utils.py
│   ├── injector.py
│   ├── ids.py
│   ├── target_selector.py
│   ├── scheduler.py
│   ├── logger.py
│   ├── config.py
│   └── exceptions.py
├── examples/
│   ├── basic_bit_flip.py
│   ├── fp16_injection.py
│   ├── bf16_injection.py
│   ├── fp32_injection.py
│   ├── model_parameter_injection.py
│   └── training_loop_integration.py
├── tests/
│   ├── test_bit_utils.py
│   ├── test_injector.py
│   ├── test_target_selector.py
│   ├── test_scheduler.py
│   ├── test_logger.py
│   ├── test_config.py
│   ├── test_training_integration.py
│   └── test_cuda.py
├── configs/example_fault.yaml
├── scripts/environment_check.py
├── docs/design.md
├── docs/environment.md
├── experiments/.gitkeep
├── logs/.gitkeep
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## 当前功能与限制

已实现：

- FP16/BF16/FP32 单 bit 与多事件 bit flip，CPU/CUDA Tensor 路径。
- 精确 bit 验证、原/新数值与位模式记录、非连续 Tensor 逻辑索引语义。
- sign/exponent/mantissa 自动分类与边界校验。
- 模型参数选择、dtype/名称过滤、seeded 随机目标选择。
- 显式 step、step range、随机 step、事件预算和一次性触发 scheduler。
- JSONL/CSV/字典事件、失败事件、可选 Tensor 备份、运行元数据。
- YAML 配置、配置快照、scheduler plan 与版本记录。
- AdamW 训练示例、持久化低精度、transient compute copy 和 AMP-FP32 路径说明。

当前限制：

- 仅支持 FP16、BF16、FP32；不支持 FP64、FP8、整数、量化 Tensor 或稀疏 Tensor。
- 仅模拟 Tensor 数据损坏；不模拟地址译码、存储单元邻接、cache line、寄存器、
  Tensor Core、矩阵乘法累加器、ECC 或真实硬件错误时序。
- 第一版以单 GPU、单进程训练为主；未实现 DDP/FSDP 一致性注入和通信 hook。
- 不自动识别 GPT-2 模块语义，不提供 Transformer 层自动遍历策略。
- 自动配置 runner 尚未把 multi-bit plan、优化器状态/梯度/激活注入整合为统一流程；
  当前多 bit API 与底层训练集成已提供，可在实验中组合。
- `inplace=True` 对非叶子 autograd Tensor 会被拒绝；需要在安全阶段注入或使用
  非原地结果替换下游计算。
- GPU 上为记录精确原/新值会读取标量，可能触发同步；`execution_time_ns` 包含这些
  同步成本。若只做吞吐测试，应在实验设计中单独测量未启用日志的注入路径。

## 后续扩展计划

1. 增加 forward/backward/optimizer 阶段的算子级 hook 和 ActivationSelector。
2. 支持 GPT-2/GPT-NeoX/Llama 系列层、attention/MLP/embedding 语义目标选择。
3. 增加梯度、AdamW 一阶/二阶状态、master weight 与 compute weight 的一致注入。
4. 增加永久/间歇故障生命周期、恢复对象版本号和 checkpoint 对齐。
5. 增加不可泄漏的 blind evaluation API，使检测器只能拿到训练可观测量。
6. 增加 CUDA event 计时、异步日志队列、批量事件写入和吞吐基准。
7. 扩展多 GPU/DDP 注入矩阵，并增加真实硬件 SEU 数据对照实验。
