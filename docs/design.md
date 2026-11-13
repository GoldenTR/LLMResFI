# LLMResFI v0.1 设计文档

## 1. 目标与适用范围

LLMResFI 服务于混合精度大语言模型训练中的 Tensor-level 软错误实验。第一版目标
是把“在哪一步、对哪个逻辑 Tensor 元素、翻转哪一个 IEEE-754 bit”做成可控、可复现、
可审计的实验操作，同时不把故障注入、检测和恢复耦合在同一个状态对象中。

适用对象：

- 单 GPU 或单进程 PyTorch 训练；
- GPT-2 Small 等 Transformer 参数和激活；
- FP16、BF16、FP32 Tensor；
- AdamW 等标准优化器的训练循环；
- 低精度持久化权重、FP32 master weight、autocast 临时操作数等不同数据路径。

不适用范围：

- 物理 GPU SRAM/DRAM/cache/寄存器/Tensor Core 的 SEU 复现；
- 真实故障的地址相关性和空间相关性；
- ECC、scrub、refresh、电压/温度/工艺偏差建模；
- 分布式通信、跨 rank 梯度桶、FSDP shard 的自动一致性注入；
- 检测算法或恢复算法本身。

## 2. 软件注入与真实硬件软错误的区别

LLMResFI 的 `inject_bit_flip()` 对 PyTorch Tensor 的整数位视图执行 XOR：

```text
IEEE-754 float word -> integer view -> bitwise XOR mask -> float view
```

这能精确控制一个逻辑元素的位模式，但缺少真实硬件故障的微架构上下文。真实 GPU
SEU 可能发生在寄存器堆、L1/L2 cache、DRAM、共享内存、矩阵累加器或控制逻辑中；
其影响取决于数据驻留位置、访问时序、ECC、复制、缓存替换和算子融合。因此，
LLMResFI 的数据只能解释为“软件 Tensor 数据损坏下的算法韧性”，不能直接外推为
真实物理 GPU 软错误率或微架构故障模型。

## 3. 模块边界

```text
配置/YAML ──> ExperimentConfig ──> FaultScheduler ──> should_inject(step)
                                                        │
训练程序 ──> TargetSelector ──> Tensor ──> inject_bit_flip()
                                              │
                                              ├──> 返回损坏 Tensor、值、bit pattern 元信息
                                              └──> InjectionResult
                                                        │
审计驱动 ────────────────────────────────────────> EventLogger
                                                        │
                                                 JSONL / CSV / backup / metadata
```

边界规则：

1. `injector.py` 只负责定位、翻转和验证，不写日志。
2. `logger.py` 只负责 ground truth 审计，不参与 Tensor 修改。
3. `scheduler.py` 只负责何时触发，不选择 Tensor 和 bit。
4. 检测/恢复模块不得接收 `InjectionResult`、logger、日志文件路径或注入 plan。
5. Tensor 不保存故障元信息属性；检测器不能通过 Tensor 自身读取注入位置。

## 4. Bit 表示与格式

所有 dtype 统一使用 LSB=0 的编号。

| dtype | 存储位宽 | Sign | Exponent | Mantissa | 整数视图 |
|---|---:|---|---|---|---|
| FP16 | 16 | 15 | 14–10 | 9–0 | `torch.int16` |
| BF16 | 16 | 15 | 14–7 | 6–0 | `torch.int16` |
| FP32 | 32 | 31 | 30–23 | 22–0 | `torch.int32` |

`get_float_format(dtype)` 返回不可变 `FloatFormat`，`get_bit_type(dtype, bit)` 返回
`BitType.SIGN/EXPONENT/MANTISSA`。所有 bit 位置都进行边界检查。

FP16 和 BF16 虽然都是 16 bit，但布局不同；代码通过 dtype 查表，不使用“16 bit 就
一定相同”的假设。FP32 的掩码使用有符号 two's-complement 表示，避免 Python
`1 << 31` 无法直接写入 `int32` 的溢出问题。

## 5. 元素索引与存储语义

`flat_index` 是 row-major 逻辑展平索引，等价于：

```python
tensor.flatten()[flat_index]
```

但实现不会对输入 Tensor 调用 `reshape()` 后原地修改，因为非连续 slice 的
`reshape()` 可能复制，导致修改的不是用户原始存储。实现先把 flat index 转换为
多维坐标，再使用基本整数索引得到零维 view：

```python
coordinates = unravel_row_major(flat_index, tensor.shape)
element_view = tensor[coordinates]
```

该 view 与原始存储共享数据，因此 `inplace=True` 可以正确修改 slice、transpose 和
共享存储 Tensor 的底层元素。非原地模式先 `clone()`，再在副本上执行同样逻辑；
副本不保证保留非连续 stride，但 shape、dtype、device 和逻辑值语义保持一致。

## 6. 注入验证

每次注入记录：

- 原始无符号 bit pattern；
- 损坏后无符号 bit pattern；
- 两者 XOR 是否恰好等于 `1 << bit_position`；
- 原始值和损坏后的 Python 浮点值。

若验证失败，抛 `InjectionVerificationError`；原地模式会尝试恢复原 bit，避免把
未验证状态留在训练中。NaN/Inf 不应被过滤，因为它们是合法的位级结果。

## 7. Autograd 与训练阶段

`inplace=True` 且目标为带 `grad_fn` 的非叶子 Tensor 时，工具拒绝操作。原因是通过
`.view(int_dtype)` 修改数据会增加 version counter，可能使后续 `backward()` 报
“变量被修改”。安全策略：

- 持久化叶子参数：在 `optimizer.zero_grad()` 后、forward 前原地注入；
- 激活/transient compute 副本：使用 `inplace=False`，将返回 Tensor 送入下游；
- 已消费的梯度：在 backward 完成后再修改；
- 不在 forward 和 backward 之间原地破坏仍需反向传播的激活。

注入发生在不同阶段的研究含义：

| 阶段 | 影响 |
|---|---|
| forward 前参数 | 当前 forward、backward 和 optimizer step |
| forward 后激活 | 仅影响该激活的下游使用；原地可能破坏图 |
| backward 后梯度 | 影响 optimizer step 和参数更新 |
| optimizer step 后参数 | 主要影响后续 step，直到再次被覆盖 |

## 8. 混合精度训练路径

### 8.1 FP32 参数 + autocast

标准 PyTorch AMP 保留 FP32 持久化参数。autocast 在算子调用边界把参数/输入转换为
FP16 或 BF16 临时操作数。对 FP32 参数执行 bit flip，破坏的是持久化 FP32 存储；
下一次算子仍会在其转换为低精度时传播该破坏，但这不是“持久化 FP16 权重位翻转”。

若实验问题是“低精度计算操作数被 SEU 击中”，需要：

1. 在自定义低精度 copy 上注入；
2. 在算子 wrapper/forward hook 捕获实际低精度 Tensor 后注入；或
3. 显式维护低精度 compute weight cache，并记录其版本。

### 8.2 持久化低精度参数

`model.to(torch.float16/bfloat16)` 后参数本身就是低精度 Tensor。原地注入直接改变
后续 forward 读取的权重；优化器 step 可能覆盖损坏，因此必须记录注入与 optimizer
step 的先后顺序。

### 8.3 自定义 master/compute 权重

从 FP32 master weight 执行 `.to(low_dtype)` 得到的 compute copy 可能是临时 Tensor。
非原地注入返回的 Tensor 仍可回溯 autograd 图。如果每 step 重建 copy，故障只影响
当步；如果缓存 copy，则故障持续到下一次同步。工具不假设任何框架一定具有两种权重，
也不会自动声称某种权重路径存在。

## 9. 调度语义

- step 从 0 开始；
- 同一计划 step 只能被 `should_inject()` 认领一次；
- `max_faults` 是所有 step 的总事件预算；
- 当候选 step 多于预算时，用局部 seeded RNG 选子集并按时间排序；
- 随机 step 和随机 element 不读取全局随机状态；
- `reset()` 只恢复原 plan，不重新采样。

这避免了重复调用 `should_inject()` 造成同一事件被触发多次，也避免不同模块共享
全局 RNG 导致实验结果漂移。

## 10. 日志与泄漏隔离

日志字段是评估 ground truth，至少记录 experiment/fault ID、step、Tensor 逻辑名、
shape/dtype/device、flat index、bit position、bit type、原/新值、原/新位模式、
注入时间戳、调用方参数版本（`parameter_version`）、注入阶段
（`injection_phase`）、PyTorch Tensor version counter、seed、成功标志和失败原因。
`parameter_version` 由训练程序定义（例如 optimizer step 或 checkpoint 版本），
避免仅依赖日志写入时间推断参数状态。

JSONL 使用逐事件 append 和 `allow_nan=False`；NaN/Inf 用字符串表示，保证严格 JSON
可解析。CSV 使用固定字段集合。Tensor 备份是可选项，会显式产生 CPU 拷贝和存储开销，
因此不能默认开启。

检测器接口设计原则：

- 只接收训练中自然可观测的 Tensor/标量；
- 不接收 `InjectionResult`；
- 不读取 `fault_events.*` 和 `run_metadata.json` 中的 fault 字段；
- 不从日志目录、文件名或 Tensor 属性反查故障位置；
- 评估脚本在检测输出提交后才读取 ground truth 做匹配。

## 11. 可复现性

复现要素：

1. 固定 `experiment.seed`；
2. 固定显式 injection step 或 seed 驱动的 step range；
3. 固定目标参数名、dtype、名称过滤和选择 seed；
4. 使用工具生成的 plan/log 记录实际位置；
5. 保存原始 YAML 和 resolved config；
6. 记录工具版本、Python、PyTorch、CUDA runtime、GPU 探测结果；
7. 使用唯一 experiment ID 和 append-only 日志。

随机计划不依赖全局 `random`/`numpy`/PyTorch RNG。实际训练中的数据 shuffle 和模型
初始化仍需实验代码单独设 seed。

## 12. 性能与同步

位操作在 Tensor 所在 device 完成，仅对目标零维 view 执行整数 XOR，不对整个 Tensor
做 NumPy 转换或反复 CPU 拷贝。日志写入与注入执行分离。

当前为返回精确值和位模式会读取标量；CUDA 上该读取可能同步。`execution_time_ns`
反映包含同步的端到端注入开销。若只研究吞吐，应单独 benchmark 不记录标量的路径，
或在异步日志队列中延迟记录；不要在每次无关操作后强制 `torch.cuda.synchronize()`。

## 13. 已知限制与后续扩展

已知限制包括 FP8/量化/稀疏 Tensor 未支持、无分布式一致性注入、无算子级自动 hook、
无真实物理 SEU 校准、自动 runner 对 multi-bit 与 optimizer state 支持不完整。

建议扩展顺序：算子级注入点 -> Transformer 语义 selector -> 梯度/AdamW 状态注入 ->
故障生命周期和恢复版本 -> blind detection API -> 异步日志/性能基准 -> 真实 GPU
可靠性实验对照。
