# T0a. 基础设施封顶

先把仓库从“能跑”变成“能稳定复现、能解释配置、能做最小回归”。

**建议窗口：D1-D3**

| 子维度       | 任务                                                                                         |
| --------- | ------------------------------------------------------------------------------------------ |
| Impl      | 统一 `launch.py`、`train.py`、config schema；修正并行约束断言；让命令行真正使用传入 config，而不是写死默认路径                  |
| Correctness | 建最小 smoke matrix：`1卡`、`TP=2`、`PP=2`、`DP=2`、`CP=2` 中至少能明确哪些已支持、哪些未接通                           |
| Layout    | 日志输出 rank→`(dp, pp, cp, tp)` 映射、world size、group ids、每 rank tensor shape                      |
| Comm/NCCL | 给 TP/PP/DP/CP 通信入口统一包一层 tag；先不追 overlap，先保证每次 collective/P2P 都能被定位                           |
| Recovery  | 暂不做 DCP，先定义 state contract：至少 model/optimizer/RNG/step/tokens 要有一致保存边界                       |
| Source    | 读：`train.py`、`launch.py`、`process_group_manager.py`、`checkpoint.py`                           |
| Failure   | 先复盘启动类问题：world size 不匹配、shape 不整除、config 断言错、rank mapping 错位                                  |
| 交付        | `docs/system_contract.md`、`docs/run_matrix.md`、`reports/t0a_bootstrap.md`                    |
| 通过标准      | 任意一个支持的实验都能单命令启动；你能清楚说出“哪些并行真的可用，哪些只是代码存在但未接线”                                             |

[1]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.10 documentation"
[2]: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html?utm_source=chatgpt.com "torch.distributed.checkpoint"
