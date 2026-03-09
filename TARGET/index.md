# TARGET

先把总目标钉死：**不是把 multiGPT 做成“会很多 feature 的 demo”，而是做成一个可以被从张量布局、collective、stream/event、overlap、checkpoint、NCCL、PyTorch internals、C++/CUDA 一路追问到底都不虚的训练系统主干。** 但对着当前仓库，优先级必须服从代码现实：先把 **训练入口、配置约束、trace、checkpoint state contract、TP/PP/DP correctness** 做成闭环，再去做 **CP 真接线、SP、MoE、vision、custom op**。现在仓库里 TP/PP/DP/CP 都有代码雏形，但 profiler、测试、恢复、任务边界还没成体系；因此这份安排改成 **先主干证据化，再扩展能力**。Megatron 当前官方主线仍然是 **DP→TP→PP→CP**，并且 **TP 场景建议配 SP**；DCP 支持并行 save/load 与 load-time resharding，PyTorch profiler 可导出 Chrome/Perfetto trace，custom C++/CUDA op 是官方推荐路径，NCCL 调试面向 `WARN/INFO/TRACE`、`DEBUG_FILE` 和 RAS。([NVIDIA Docs][1])

## 导航

- [任务关闭标准](./01_exit_criteria.md)
- [T0a 基础设施封顶](./02_t0_mainline.md)
- [T0b 观测与时间线](./03_t1_tp.md)
- [T1 TP 深挖](./04_t2_dp.md)
- [T2 PP 异步化](./05_t3_pp.md)
- [T3 DP Bucket 证据化](./06_t4_sp.md)
- [T4 Checkpoint 契约](./07_t5_cp.md)
- [T5 CP 真接线](./08_t6_ep_moe.md)
- [T6 SP](./09_t7_vision.md)
- [T7 MoE / Vision](./10_t8_checkpoint.md)
- [T9 C++/CUDA + NCCL](./11_t9_cpp_cuda_nccl.md)
- [任务依赖图](./12_dependencies.md)
- [每天检查表](./13_daily_checklist.md)
- [面试映射表](./14_interview_mapping.md)
- [月底仓库产物](./15_month_end_deliverables.md)
- [执行原则](./16_execution_principles.md)

[1]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html?utm_source=chatgpt.com "Parallelism Strategies Guide — Megatron Core"
