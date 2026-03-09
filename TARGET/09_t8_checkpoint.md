# T8. DCP/Async checkpoint：不是收尾功能，而是把恢复从“能存档”拉到“训练系统级”

## 建议窗口

D27–D28

## 子维度任务

### Impl
全面切 DCP；同拓扑恢复、异 world size 恢复、async save、save frequency 策略。

### Comm/NCCL
标记 checkpoint 时 rank 侧 I/O 与 CPU/pinned memory 行为；DCP `async_save()` 设计目的就是 off critical path，但官方也明确指出额外内存和并发开销。([PyTorch Docs][9])

### Overlap
checkpoint 期间主训练 step 是否被阻塞，用 trace/step time 对比证明。

### Trace
sync save vs async save 的 step timeline 对比。

### Recovery
至少完成一次 `2卡存 -> 4卡恢复` 或 world-size 变更恢复；DCP 支持 load-time resharding。([PyTorch Docs][3])

### Source
读：DCP API、async checkpoint recipe、你自己的 state dict 组织。

### Failure
复盘 1 个恢复类问题：状态不完整、optimizer shard 不对、异拓扑恢复错。

### 交付
`reports/t8_checkpoint.md`、恢复脚本、恢复演示日志。

### 通过标准
你能回答“为什么这不是 torch.save 换皮、async save 真正收益在哪、代价是什么”。
