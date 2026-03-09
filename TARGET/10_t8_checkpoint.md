# T7. MoE / Vision / DCP 只选一条

这一阶段不再贪多，只允许选一条真正能落地的扩展线。

**建议窗口：D28-D30**

| 方向 | 进入条件 |
| --- | --- |
| MoE | TP/PP/DP/checkpoint 主干稳定，且 MLP 抽象已经清楚 |
| Vision | trunk 已稳定，且你只做最小 patch embed → projector → GPT 路径 |
| DCP | 现有 checkpoint contract 已稳定，且你愿意先做同拓扑到异拓扑的 state dict 重构 |

| 子维度       | 任务                                                                                                                               |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Impl      | 三选一：`MoE`、`Vision trunk`、`DCP/async checkpoint`，禁止并行开三线                                                                            |
| Comm/NCCL | 只分析所选方向里的关键通信：MoE 看 dispatch，Vision 看 token layout，DCP 看 save/load/reshard                                                         |
| Trace     | 必须给出“扩展前 vs 扩展后”的时间线和内存对比                                                                                                          |
| Recovery  | 所选方向不能破坏已有 checkpoint 恢复链路                                                                                                          |
| Source    | 只读与所选方向直接相关的官方文档和当前代码入口                                                                                                            |
| Failure   | 复盘 1 个该方向特有的问题                                                                                                                    |
| 交付        | `reports/t7_extension.md`，并在开头明确写“本月只选择了哪一条扩展线，以及为什么”                                                                              |
| 通过标准      | 你能解释为什么没有同时做 MoE、Vision、DCP；并证明这是出于主干约束，而不是拖延                                                                                      |

[1]: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html?utm_source=chatgpt.com "Asynchronous Saving with Distributed Checkpoint (DCP)"
[2]: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html?utm_source=chatgpt.com "torch.distributed.checkpoint"
