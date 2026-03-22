# multiGPT

一个面向训练系统学习的最小分布式训练仓库。

这个仓库不是“功能越多越好”的 demo，而是刻意保留较小代码体量，用来把下面这些技术路径讲清楚：

- 训练循环状态机
- TP / PP / DP / CP 的张量流与通信边界
- profiler / trace / step timeline
- checkpoint / resume / RNG 恢复
- eager attention 与 SDPA 后端切换

## 从哪里开始

如果你是第一次看这个仓库，建议按这个顺序读：

1. 教程文档：[docs/tutorial_training_system.md](/Users/taom/Repo/multiGPT/docs/tutorial_training_system.md)
2. 本次主干优化说明：[docs/do_technical_optimization.md](/Users/taom/Repo/multiGPT/docs/do_technical_optimization.md)
3. 系统契约：[docs/system_contract.md](/Users/taom/Repo/multiGPT/docs/system_contract.md)
4. 并行布局：[docs/parallel_layouts.md](/Users/taom/Repo/multiGPT/docs/parallel_layouts.md)
5. 故障手册：[docs/failure_playbook.md](/Users/taom/Repo/multiGPT/docs/failure_playbook.md)
6. 训练入口：[train.py](/Users/taom/Repo/multiGPT/train.py)
7. PP 调度：[picotron/pipeline_parallel/pipeline_parallel.py](/Users/taom/Repo/multiGPT/picotron/pipeline_parallel/pipeline_parallel.py)
8. PP 通信：[picotron/pipeline_parallel/pp_communications.py](/Users/taom/Repo/multiGPT/picotron/pipeline_parallel/pp_communications.py)

## 最小使用

查看启动命令：

```bash
./.venv/bin/python launch.py --config configs/gpt2.yaml --dry-run
```

实际启动：

```bash
./.venv/bin/python launch.py --config configs/gpt2.yaml
```

## 工具脚本

- `./tools/topology_check.sh`
  - 检查当前配置的拓扑约束
- `./tools/profile_run.sh`
  - 跑一个最小 profiler trace
- `./tools/restore_demo.sh`
  - 演示 checkpoint 保存与恢复
- `./tools/smoke_tests.sh`
  - 一键回归当前主干

## 教学导向的当前能力

- 训练 step 具备时间线摘要
- 可选 `torch.profiler`
- PP 通信具备 `comm stream + event` 基础结构
- checkpoint 支持模型、优化器、步数、token 数与 RNG 状态恢复
- attention 可在 `eager` / `sdpa` 两种后端间切换
- 支持可选 `activation checkpointing`
- 支持可选 `autocast/bf16`
- 单卡 CUDA dense 路径支持可选 `torch.compile`

## 当前有意保留的边界

- 还没有做 DCP / async checkpoint
- 还没有做 CUDA Graph / torch.compile 主线
- 还没有做多节点 topology-aware 调度
- 还没有把 CP/PP overlap 优化到 production 级别

这些不是遗漏，而是刻意留给后续 `T3 -> T8 -> L5/L6` 的深入路线。
