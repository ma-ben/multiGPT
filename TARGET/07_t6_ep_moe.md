# T6. EP + MoE：不是又开一条线，而是把现有 engine 推到稀疏路由

## 建议窗口

D20–D23

## 子维度任务

### Impl
dense FFN → top-2 MoE；router、aux loss、expert local FFN、combine；dispatcher 独立成接口层。

### Comm/NCCL
先做 `alltoall` token dispatch；记录 token count、imbalance、每 expert 负载；Megatron MoE 明确把 EP 与 DP/TP/PP/SP/CP 组合起来，并把 token dispatch / grouped GEMM / overlap 作为优化点。([NVIDIA Docs][8])

### Overlap
拆分 router / dispatch / expert compute / combine 四段计时；先找最大段，再决定是 fuse 还是 overlap。

### Trace
做 dense vs MoE、不同 expert 数、不同 capacity/top-k 的对照。

### Recovery
MoE checkpoint：router、expert shards、optimizer 状态全保存；恢复后 token 分配统计合理。

### Source
读：Megatron MoE 指南、你仓库 MLP 入口、dispatcher 抽象。

### Failure
复盘 1 个 MoE 问题：专家负载崩坏、alltoall shape 不稳、combine scatter/gather 错。

### 交付
`reports/t6_moe_ep.md`、token path 图、expert balance 表。

### 通过标准
你能完整画出一次 token 在 router→dispatch→expert→combine 的张量和通信路径。
