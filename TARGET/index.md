# TARGET

先把总目标钉死：**不是把 multiGPT 做成“会很多 feature 的 demo”，而是做成一个可以被从张量布局、collective、stream/event、overlap、checkpoint、NCCL、PyTorch internals、C++/CUDA 一路追问到底都不虚的训练系统主干。** 一个月内只围绕这根主干做纵深：先把现有 TP/PP/DP 做到可证伪、可恢复、可扩展，再沿同一方法论扩到 **SP/CP、EP/MoE、vision trunk、custom C++/CUDA op**；一年内再把它推进到多节点、可复现 benchmark、故障注入、开源级文档和 patch 贡献。Megatron 当前官方主线就是 **DP→TP→PP→CP**，并且 **TP 场景建议配 SP**；CP 是把几乎所有激活按 sequence 切开，attention 是关键特例；MoE/EP 能和 DP/TP/PP/SP/CP 组合；DCP 支持并行 save/load 与 load-time resharding，`async_save()` 的目的就是把 checkpoint 尽量挪出关键训练路径；PyTorch profiler 可导出 Chrome/Perfetto trace；custom C++/CUDA op 是官方推荐路径；NCCL 官方调试面向 `WARN/INFO/TRACE`、`DEBUG_FILE` 和 RAS。([NVIDIA Docs][1])

## 导航

- [完整总纲](../TARGET.md)
- [01 T0 主干封顶](./01_t0_mainline.md)
- [02 T1 TP 深挖](./02_t1_tp.md)
- [03 T2 DP 深挖](./03_t2_dp.md)
- [04 T3 PP 深挖](./04_t3_pp.md)
- [05 T4 SP](./05_t4_sp.md)
- [06 T5 CP](./06_t5_cp.md)
- [07 T6 EP + MoE](./07_t6_ep_moe.md)
- [08 T7 Vision trunk](./08_t7_vision.md)
- [09 T8 DCP / Async checkpoint](./09_t8_checkpoint.md)
- [10 T9 C++/CUDA + NCCL](./10_t9_cpp_cuda_nccl.md)

## 任务关闭标准

### Correctness
单测或对齐测试通过，loss 没有异常，旧功能回归通过。

### Layout
能画清楚 rank 上参数、激活、梯度、状态的分布。

### Comm
能说清使用了哪些 collective、group、tensor shape，以及触发时机。

### Overlap
trace 里能看到通信与计算交叠，不能只靠口述。

### Metrics
至少有 step time、tokens/s、max mem、comm 占比。

### Recovery
断点恢复可跑，至少保存 model、optim、RNG、step。

### Failure
至少复盘 1 个该任务相关故障。

### Source
看过对应 Megatron、PyTorch、NCCL 的源码入口并写笔记。

### Interview
能 3 分钟讲机制，10 分钟讲取舍，30 分钟讲源码路径。

## 任务依赖图

### 1. T0 主干封顶
依赖：无。

### 2. T1 TP 深挖
依赖：T0。

### 3. T2 DP 深挖
依赖：T0。

### 4. T3 PP 深挖
依赖：T0。

### 5. T4 SP
依赖：T1。

### 6. T5 CP
依赖：T4。

### 7. T6 EP+MoE
依赖：T1 + T2 + T3。

### 8. T7 Vision
依赖：T0，最好在 T4/T5 后做。

### 9. T8 DCP/Async
依赖：T0 起即可并行推进，但最好在 T1-T7 每步都回归。

### 10. T9 C++/CUDA + NCCL
依赖：T0 起可并行阅读，最终落在热路径上。

## 每天检查表

### 今日主任务
只允许一个：`Tn-子点`。

### 今日 patch
1 个核心 patch，禁止碎 patch 堆积。

### 今日证据
1 份 trace + 1 份 benchmark/日志。

### 今日源码入口
2 个函数或类：你读了什么，理解了什么。

### 今日故障
1 个现象，1 个根因，1 个修复。

### 今日回归
哪些旧实验重跑通过。

### 今日 3 分钟讲解
只讲一个机制：时间线 + 张量流 + collective。

### 今日欠债
明确写出未完成与阻塞点。

## 面试映射表

### T0
你的 step timeline 长什么样；为什么 trace 比 print 更可信。

### T1
TP 的通信点在哪；为什么 async dgrad overlap 有时看起来“开了但没效果”。

### T2
bucket 为什么决定 overlap 粒度；为什么 bucket 过大或过小都坏。

### T3
1F1B 的 bubble 来自哪；为什么 `wait+synchronize` 会毁掉伪异步。

### T4
SP 为什么常与 TP 绑定；它到底改了哪些层的激活布局。

### T5
CP 为什么不是“更大的 SP”；什么时候 TP=2+CP=2 胜过 TP=4。

### T6
EP 为什么不是另一个 DP；MoE 真正慢在 router、dispatch 还是 expert compute。

### T7
multimodal 为什么不是新系统；只是 token 图更一般还是调度也变了。

### T8
为什么 DCP 不是 `torch.save` 换皮；异拓扑恢复靠什么成立。

### T9
为什么这个 op 值得写成 C++/CUDA；NCCL WARN/INFO/TRACE 分别什么时候开。

## 月底必须存在的仓库产物

### 系统设计
`docs/system_contract.md`、`docs/parallel_layouts.md`。

### 实验报告
`reports/t1_tp_deep.md`、`t2_dp_bucket.md`、`t3_pp_async.md`、`t5_cp.md`、`t6_moe_ep.md`、`t8_checkpoint.md`。

### 故障手册
`docs/failure_playbook.md`。

### 工具脚本
`tools/profile_run.sh`、`tools/nccl_debug.sh`、`tools/topology_check.sh`、`tools/restore_demo.sh`。

### 代码能力
dense GPT、SP/CP、MoE/EP、vision trunk、1 个 custom op。

### README
项目定位、能力矩阵、运行方式、关键实验、已知限制。

## 执行原则

### 每个局部都包含全体
任何任务都同时覆盖实现、通信、overlap、trace、恢复、源码、故障、面试表达。

### 先证据后 feature
没 trace 或 benchmark 的功能，不算做完。

### 先主干后扩展
所有新任务都必须复用同一套调度、观测、恢复与报告框架。

### 先关 correctness 再追吞吐
silently wrong 的系统越快越危险。

### 每天只打一根钉子
同一天只允许一个主问题，避免“全都推进一点点”。

照这个表打，一个月后你不是“懂了很多并行名词”，而是已经有一套能被连续追问 30 分钟都不散的训练系统样本。
