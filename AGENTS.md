# 仓库开发指南

## 项目结构与模块组织

- `llmresfi/` 包含源码包，公共 API 从 `llmresfi/__init__.py` 导出。
- `llmresfi/config.py` 定义 YAML/Pydantic 配置模式；`bitflip.py`、`fault_models.py`、`scheduler.py`、`injector.py` 与 `distributed.py` 实现注入原语、调度以及 DDP/FSDP 支持。
- `examples/` 包含可运行的推理、单 GPU 训练、DDP/FSDP 示例，以及 ResNet-18/CIFAR-10 端到端实验与分析脚本。
- `tests/` 包含 pytest 测试套件；`docs/design.md` 保存详细架构参考。
- 默认实验配置位于 `config.yml`。

## 构建、测试与开发命令

- `pip install -e .` 安装本包及运行时依赖。
- `pip install -e ".[dev,datasets]"` 额外安装测试与数据集依赖。
- `pytest` 运行 `tests/` 下的测试套件。
- `python examples/run_inference.py --config config.yml --prompt "..." --new_tokens 16` 运行单 GPU 推理故障注入示例。
- `python examples/run_training.py --target gradient --stage backward` 运行 CPU 玩具训练示例。
- `torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --stage backward` 运行分布式示例。
- `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/resnet18/run_resnet18.py --baseline --steps 4` 运行 ResNet-18 双卡冒烟实验。
- `python examples/resnet18/analyze_resnet18.py --root ../ResNet-18/runs` 生成 ResNet-18 中文结果报告。

## 编码风格与命名约定

- 使用 Python 3.9+、PEP 8 与四空格缩进。
- 公共名称应与 `llmresfi/__init__.py` 保持一致；优先使用描述性的模块、函数和变量名称。
- 在适用时使用 `from __future__ import annotations` 与类型注解。
- 新增配置字段时，应同步更新 `FIConfig` 与 `config.yml`，并在 `README.md` 中记录支持的取值。

## 测试指南

- 使用 `pytest`。测试模块命名为 `test_<area>.py`，测试函数命名为 `test_<behavior>`。
- 使用 CPU 张量、小模型与固定随机种子，保持测试快速且确定。
- 修复缺陷时，应同时覆盖新增故障模型、阶段转换与配置校验。

## 提交与 Pull Request 指南

- 遵循仓库的约定式提交前缀：`feat:`、`fix:` 或 `refactor:`。
- 保持提交聚焦；行为变化时引用相关的 `docs/design.md` 或配置字段。
- Pull Request 应包含简短描述、用于验证的测试或示例命令，以及关联的 issue。

## 配置与安全建议

- 除非模型确实需要自定义远程代码，否则使用 `trust_remote_code: false`。
- 通过 `execution.random_seed` 保持带种子的可复现运行；将输出记录到 `execution.log_dir`。
