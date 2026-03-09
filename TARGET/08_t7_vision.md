# T7. Vision trunk：不是“学视觉”，而是证明 engine 对张量图更一般

## 建议窗口

D24–D26

## 子维度任务

### Impl
只做一条最小路径：patch embed/ViT encoder → projector → GPT trunk；不准多架构发散。

### Comm/NCCL
检查视觉 token 进入 TP/SP/CP 后的 shape 与 padding；如果走 CP，要清楚图像 token/文本 token 的拼接与分块。

### Overlap
重复前面同一方法：哪段 compute，哪段 comm，哪段 wait；不另起一套语言。

### Trace
dense text vs multimodal 的 time/memory 对比。

### Recovery
multimodal checkpoint 回归；visual params + projector + trunk 全恢复。

### Source
读：Megatron multimodal CP/SP API、你新加的 vision trunk。

### Failure
复盘 1 个 multimodal 问题：patch/token layout 对不上、position encoding 冲突、跨模态拼接后 CP padding 错。

### 交付
`reports/t7_vision.md`、多模态 layout 图。

### 通过标准
你能证明“这不是另一个项目，只是同一 engine 处理了更一般的 token 图”。
