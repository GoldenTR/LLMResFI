# ResNet-18 分布式实验脚本

本目录集中存放 CIFAR-10 / ResNet-18 的双卡 DDP 故障注入与结果分析脚本。

```text
examples/resnet18/
├── run_resnet18.py       # 训练、注入、检测与回滚
└── analyze_resnet18.py   # 对比基线并生成中文报告
```

## 快速开始

在 LLMResFI 仓库根目录执行：

```bash
# 4 步双卡冒烟
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  examples/resnet18/run_resnet18.py --baseline --steps 4

# 无故障基线
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  examples/resnet18/run_resnet18.py --scenario baseline --baseline \
  --steps 96 --lr 0.05 --bit_idx 27 \
  --log_dir ../ResNet-18/runs/baseline

# 归约后故障 + 校验和 + 步级回滚
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  examples/resnet18/run_resnet18.py --scenario fault_recovered_after \
  --fault_model exponent --bit_idx 27 --injection_point after_reduce \
  --inject_rank 0 --mitigation step_rollback --verify \
  --steps 96 --lr 0.05 --log_dir ../ResNet-18/runs/fault_recovered_after

# 生成 ResNet-18 中文报告
python examples/resnet18/analyze_resnet18.py \
  --root ../ResNet-18/runs \
  --output-md ../ResNet-18/analysis/report.md \
  --output-json ../ResNet-18/analysis/report.json
```

默认实验根目录是 LLMResFI 同级目录下的 `ResNet-18/`。每个 run 写入
`summary.json`、`recovery.jsonl` 和 `training_metrics.csv`；`summary.json`
包含完整模型与优化器状态 SHA-256，用于严格无损判定。

完整实验协议与结果见同级 `ResNet-18/README.md` 和
[`docs/experiments.md`](../../docs/experiments.md)。

