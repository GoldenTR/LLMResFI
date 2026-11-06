# 分布式软错误实验

本文档给出可复现的实验协议，用于研究分布式 LLM 训练中的软错误如何影响收敛、
是否能够无损恢复，以及恢复开销有多大。

## 1. 实验环境

```bash
conda activate llmresfi                 # Python 3.10 + torch 1.12.1+cu113
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 仅使用两张 Tesla V100；2 号 GPU 是不受支持的 GT 710。
export CUDA_VISIBLE_DEVICES=0,1
```

安装本包（当前工作区已完成）以及测试依赖：

```bash
pip install -e . --no-deps
pip install pytest 'numpy<2'            # torch 1.12 与 NumPy 2.x 不兼容
```

运行完整测试套件：

```bash
pytest -q                               # CPU/gloo DDP 集成测试
CUDA_VISIBLE_DEVICES=0,1 pytest -q      # 额外运行双 GPU FSDP 分片测试
```

## 2. 研究问题

1. **影响**：单个 rank 本地位翻转经过全归约/reduce-scatter 后会传播多远，
   影响会持续多久？
2. **检测**：哪些故障可通过非有限值检查、梯度范数离群值、loss 突增或通信校验和发现？
3. **无损恢复**：瞬态故障能否通过回滚恢复，使最终参数与原无故障运行逐 bit
   一致（或在容差内一致）？
4. **开销**：检测与恢复在每个 step 上带来多少额外时间、通信和内存开销？

## 3. 故障维度

| 维度 | 取值 | 重要性 |
|---|---|---|
| 后端 | `ddp`、`fsdp` | 全归约与 reduce-scatter 的差异；状态是否分片 |
| 注入点 | `before_reduce`、`after_reduce`（DDP/新版 FSDP）；`param_shard`、`grad_shard`（torch 1.12 FSDP） | rank 本地传播与非对称发散的差异 |
| 故障 rank | `inject_rank`、`inject_ranks`、`inject_rank_fraction` | 单卡故障与多 rank 相关故障的差异 |
| 梯度桶 | `bucket_index` | 早期/晚期梯度桶与层敏感性的差异 |
| 故障模型 | `single`、`double`、`sign`、`exponent`、`mantissa`、`random_value`、`nan`、`inf`、`scale` | 静默位翻转与可检测值污染的差异 |
| 时序 | `transient`、`permanent`、`intermittent` | 一次性事件与持续硬件损伤的差异 |
| 缓解策略 | `none`、`sanitize`、`step_rollback`、`checkpoint_rollback` | 仅检测、低成本缓解与无损恢复的差异 |

## 4. 核心实验

### E1. 非对称 SDC（全归约之后）

```bash
# 通过跨 rank 校验和一致性检测 -> 回滚。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20

# 同一故障但不开启验证 -> rank 0 静默发散（SDC）。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --mitigation none --steps 20
```

比较 `Injection records`、各 rank 的 loss 以及各 rank 的梯度范数。第二次运行应显示
rank 0 发生发散，而 rank 1 仍接近无故障基线。

### E2. 对称传播（全归约之前）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --injection_point before_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20
```

全归约前的 NaN 会传播为所有 rank 上的 NaN：非有限值检测器和归约校验和都会触发。
记录所需重试次数，并将恢复后的 loss 曲线与基线运行比较。

### E3. rank 比例与梯度桶敏感性

```bash
# 一半 rank，指定一个梯度桶。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model exponent --inject_rank_fraction 0.5 \
  --bucket_index 2 --verify --mitigation none --steps 50
```

遍历不同的 `bucket_index` 与 bit 区域（`sign`、`exponent`、`mantissa`）。
每条记录中的 `bit_idxs` 标识具体 bit；可按 bit 位置使用
`analysis.bit_position_sensitivity` 聚合 SDC 计数。

### E4. FSDP 分片故障（torch 1.12）

```bash
# all-gather 前的参数分片：NaN 会传播到所有 rank 的完整参数。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection param_shard --fault_model nan --mitigation step_rollback --steps 20

# 优化器更新前的梯度分片。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection grad_shard --fault_model nan --mitigation step_rollback --steps 20
```

在 torch 1.12 下，FSDP 记录包含 `sharded: true` 与 `legacy_fsdp: true`。
`param_shard` 故障在前向阶段注入；`grad_shard` 故障由
`injector.inject_distributed_now()` 在反向传播后注入。

### E5. 缓解开销

对每种模式（`none`、`sanitize`、`step_rollback`、`checkpoint_rollback`）测量：

- 每个 step 的实际耗时（排除前几个预热步骤），
- 峰值 GPU 显存（`torch.cuda.max_memory_allocated`），
- 重试次数与额外前向/反向传播次数，
- 检查点 I/O 字节数与耗时。

各模式使用相同的随机种子和故障调度。`step_rollback` 应只在故障 step 增加一次前向/反向传播；
其固定成本是快照内存（参数与优化器状态）以及快照复制时间。

### E6. 自动化采集与分析

两个分布式示例都会为每次运行写入独立的 `summary.json`。请将每种配置放入单独目录，
避免不同运行的 summary 相互覆盖：

```bash
# 无故障参考运行。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20 \
  --log_dir runs/e1/seed42/baseline --baseline

# 模型、world size、随机种子和步数均相同的故障用例。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20 \
  --log_dir runs/e1/seed42/fault

python examples/analyze_runs.py --root runs/e1/seed42
python examples/analyze_runs.py --root runs/e1/seed42 --json \
  --output runs/e1/report.json
```

`summary.json` 包含：

| 字段 | 含义 |
|---|---|
| `config`、`run_id`、`world_size`、`steps` | 解析后的实验标识与规模 |
| `records` | 所有 rank 的注入故障：rank、bucket、张量 dtype/索引、bit 索引、注入点 |
| `loss_history` | 各 rank 的已接受步骤 loss 曲线；重试尝试不会被重复追加 |
| `per_rank_metrics` | 各 rank 的最终 loss 与梯度范数 |
| `recovery_events` | 各 rank 的检测信号、动作、step 与重试元数据 |
| `step_times` | 各 rank 的已接受步骤耗时，包含故障步骤的重放时间 |
| `peak_memory_gb` | 各 rank 的 CUDA 峰值分配显存 |
| `param_signatures` | 各 rank 的带符号/绝对值/L2/加权参数校验和 |

基线通过后端、模型、设备、world size、随机种子和步数匹配。基线与故障变体应使用相同随机种子；
聚合统计时应在多个种子上重复运行。为获得干净的开销对比，基线应与故障用例使用相同的缓解模式。
若还需分离“缓解机制本身的固定成本”和“故障重放开销”，可额外加入一个
`--mitigation none` 的基线。

分析器会报告检测率与恢复率，并按故障模型、注入点、bit 区域、rank、bucket、时序和
缓解策略对结果分组。开销表报告平均已接受步骤耗时、峰值显存、相对匹配基线的百分比开销、
发生重试的唯一 step 数以及每 rank 恢复事件数。两 rank 的成功步级回滚通常产生两个事件，
但只对应一个发生重试的 step。

故障专属分组不包含基线运行，因此计数回答的是：“在经历了该故障维度的运行中，
有多少被恢复/SDC/masked/DUE？”`by_backend` 分组则包含基线，以展示参考总体。

### E7. 结果解释

| 结果 | 解释 |
|---|---|
| `recovered` | 检测触发、回滚完成，且最终 loss/参数签名在 `--loss_atol` / `--param_atol` 容差内匹配基线。 |
| `mitigated` | 检测触发且无可测偏离，但策略未执行回滚（通常为 `sanitize`）。 |
| `detected_unrecovered` | 已观察到故障，但最终状态仍与基线不同。应排查恢复错误、检查点重放或持续故障。 |
| `sdc` | 未检测到且存在可测偏离。这是危险的静默数据损坏类别，应推动检测器/冗余机制改进。 |
| `masked` | 未检测到且无可测偏离。故障可能命中无关元素、被数值误差吸收，或被粗粒度指标掩盖。 |
| `due` | 检测后运行中止，例如重试耗尽，或没有可用的快照/检查点。 |
| `no_fault` | 未产生注入记录；在将其视为实验结果前，应检查故障调度、阶段、rank 策略和 bucket 选择。 |

`param_signature` 刻意保持紧凑，以便在 LLM 实验中控制 summary 体积。它不是逐 bit 相等的证明。
`recovered` 的含义是“在配置容差内无损”；严格无损应通过比较保存的最终 state dict，或使用
第 5 节的确定性参考协议来验证。loss 曲线可能暴露最终标量所掩盖的发散，因此对失败或边界运行，
应检查 JSON 报告中的 `curve_max_diff` 与 `first_divergence_step`。

### E8. ResNet-18 / CIFAR-10 端到端实验

`examples/resnet18/run_resnet18.py` 使用真实 CIFAR-10、CIFAR 版 ResNet-18 和双卡 DDP，
验证完整训练链路中的故障影响与恢复。脚本使用确定性预处理、分布式 sampler、固定初始权重，
并在每个 `summary.json` 中保存两个 rank 的完整模型 state-dict 与优化器状态 SHA-256。

```bash
export RUN_ROOT=/home/gjj/data/lijinshu/ResNet-18

# 基线
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario baseline --baseline --steps 96 --lr 0.05 --bit_idx 27 \
  --log_dir "$RUN_ROOT/runs/baseline"

# 归约后 bit 27：单 rank 非对称 SDC（不开启检测）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario fault_sdc_after --fault_model exponent --bit_idx 27 \
  --injection_point after_reduce --inject_rank 0 --mitigation none \
  --steps 96 --lr 0.05 --log_dir "$RUN_ROOT/runs/fault_sdc_after"

# 同一故障：校验和检测 + 步级回滚
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py \
  --scenario fault_recovered_after --fault_model exponent --bit_idx 27 \
  --injection_point after_reduce --inject_rank 0 \
  --mitigation step_rollback --verify \
  --steps 96 --lr 0.05 --log_dir "$RUN_ROOT/runs/fault_recovered_after"

python examples/resnet18/analyze_resnet18.py \
  --root "$RUN_ROOT/runs" \
  --output-md "$RUN_ROOT/analysis/report.md" \
  --output-json "$RUN_ROOT/analysis/report.json"
```

本机两张 V100 的代表性结果（seed 42，96 step）：

| 场景 | Test Acc | ΔAcc | Test Loss | 严格无损 |
|---|---:|---:|---:|---|
| 基线 | 54.69% | +0.00 pp | 1.264643 | — |
| `after_reduce` bit27，无缓解 | 50.85% | -3.84 pp | 1.357679 | 否（SDC） |
| `after_reduce` bit27，检测 + 回滚 | 54.69% | +0.00 pp | 1.264643 | 是 |
| `before_reduce` bit27，无缓解 | 52.56% | -2.13 pp | 1.301676 | 否（漏检 SDC） |
| `before_reduce` bit30，检测 + 回滚 | 54.69% | +0.00 pp | 1.264643 | 是 |

严格无损的判据是模型 state-dict 与优化器状态的完整 SHA-256 都匹配基线，而不是 loss 近似。
归约前 bit 27 的扰动落在单次全归约校验和的相对容差内，因此即使最终模型已偏离基线仍会漏检；
这是可观测性边界，不是回滚失败。`inject_rank=99` 的受控无故障对照显示：仅校验和约增加
13.0% step 时间，校验和 + 检测器 + 每步快照约增加 42.6%；真实故障还会重放一个 batch。
完整配置、目录结构与结果分析见同级 `ResNet-18/README.md`。

## 5. 无损恢复校验

集成测试（`tests/test_ddp_integration.py`）实现了最严格的校验：在瞬态故障并回滚后，
重放同一 batch，并将最终参数与独立训练的无故障参考模型比较。在 CPU/gloo 设置下，
差异小于 `1e-6` 视为无损。ResNet-18 示例进一步保存完整模型与优化器 SHA-256，
可直接判定恢复运行是否与基线逐 bit 一致。

GPU 实验可采用相同流程：

1. 在故障 step 前保存 RNG 状态。
2. 执行故障 step，检测故障，恢复步级快照，并重放同一 batch。
3. 恢复保存的 RNG 状态，在初始参数完全相同的独立模型上执行一次干净参考 step。
4. 逐元素比较参数，并报告最大绝对差异。

注意事项：

- 只有当 batch 与数据顺序被重放，且所有随机操作均已设置种子/确定性时，
  `step_rollback` 才是无损的。
- `checkpoint_rollback` 在完成的 step 边界恢复模型/优化器/RNG。需要将 tracker 与数据迭代器
  回退到 `checkpoint_step`，并重放其后的每个 step；仅从当前 step 继续并不无损。
  第一次 `begin_step()` 会创建初始检查点，分布式示例会处理 tracker/历史回退，
  但真实数据集加载器还必须回退到检查点对应的数据位置。
- GPU 全归约的归约顺序可能引入微小数值差异；应使用容差，而非要求逐 bit 相等。
- 除非将快照写入 CPU/磁盘，否则快照会使模型与优化器内存占用翻倍。

## 6. 可检测性边界

- **归约后污染**：跨 rank 校验和一致性可检测任何偏离，即使变化小于 float32 归约噪声。
- **归约前污染**：标量和校验和会与 float32 累加噪声竞争，较小的尾数翻转可能漏检。
  非有限值与大幅值故障可被值/范数检测器可靠发现。
- **归约前单 bit 的精确检测**需要冗余机制（重算梯度、复制 bucket，或对复制数据执行
  第二次独立归约）。这是单次全归约的固有限制，不只是实现缺口。
- 在 ResNet-18 实测中，归约前 bit 27 的指数域扰动已造成 -2.13 pp 准确率和
  +0.0370 测试损失，但未触发校验和；bit 30 的更强扰动则可被检测并恢复。

## 7. 推荐报告方式

对每种配置报告：

- 故障位置（`rank`、`bucket_index`、`flat_idx`、`bit_idxs`），
- 检测信号与检测延迟，
- 恢复动作与重试次数，
- 相对基线的最终 loss / 梯度范数偏离，
- 相对无故障参考的最大参数差异，
- 每步耗时与峰值显存开销。

至少聚合 5 个随机种子，并报告被分类为 masked / detected / SDC / DUE 的运行比例。
机器可读报告可通过以下命令生成：

```bash
python examples/analyze_runs.py --root runs/campaign --json \
  --output runs/campaign-report.json
```

记录基线比较所使用的容差。不要仅凭 `summary.json` 声称某次运行逐 bit 无损；
当该结论很重要时，应执行完整的 state-dict 对照。
