# T0. 主干封顶：把仓库从“能跑”变成“可测、可恢复、可解释”

## 建议窗口

D1–D3

## 子维度任务

### Impl
统一 `exp_dir`、config schema、seed、log 格式、rank 命名；修掉 launch/config/correctness/P0 bug；把每次实验都变成单命令复现。

### Comm/NCCL
给 TP/PP/DP 全部打 group 名称和 collective 包装层；日志中输出 rank→(dp,tp,pp,cp) 映射；预留 `nccl_debug.sh`。

### Overlap
暂不优化，只建立时间线骨架：forward/backward/dp_sync/tp_comm/pp_p2p 分段标记。

### Trace
接 `torch.profiler` + `record_function`，每个阶段独立 tag，导出 trace；PyTorch profiler 本身就是用来看算子耗时、内存、device activity 和执行轨迹的。([PyTorch Docs][2])

### Recovery
先接 DCP 同拓扑 save/load；存 model/optim/RNG/step；DCP 原生支持多 rank 并行读写。([PyTorch Docs][3])

### Source
读：PyTorch profiler、DCP 文档、你自己仓库 train/launch/parallel manager 入口。

### Failure
复盘 1 次最常见的启动类问题：world size 不匹配、shape 不整除、rank mapping 错位。

### 交付
`docs/system_contract.md`、`reports/t0_baseline.md`、一份 baseline trace、一次恢复演示。

### 通过标准
任意实验一键重跑；trace 可读；checkpoint 可恢复；你能完整讲一次 step 的时间线。
