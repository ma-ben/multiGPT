# 面试映射表

做完每个任务后，立刻能答的题。

| 任务 | 必须能答的问题                                                   |
| -- | --------------------------------------------------------- |
| T0a | 当前仓库里哪些并行是真的可用，哪些只是代码存在但没接通                         |
| T0b | 你的 step timeline 长什么样；为什么 trace 比 print 更可信               |
| T1 | TP 的通信点在哪；为什么 async dgrad overlap 有时看起来“开了但没效果”           |
| T2 | 1F1B 的 bubble 来自哪；为什么 `wait+synchronize` 会毁掉伪异步           |
| T3 | bucket 为什么决定 overlap 粒度；为什么 bucket 过大/过小都坏                |
| T4 | 为什么当前恢复体系还不是 DCP；它的 state contract 缺什么                    |
| T5 | 当前 CP 到底接在了哪里；为什么“有 CP 文件”不等于“有 CP feature”               |
| T6 | SP 为什么常与 TP 绑定；它到底改了哪些层的激活布局                              |
| T7 | 为什么这个月只选 MoE、Vision、DCP 中的一条，而不是三条并推                     |
| T9 | 为什么这个 op 值得写成 C++/CUDA；NCCL WARN/INFO/TRACE 分别什么时候开       |
