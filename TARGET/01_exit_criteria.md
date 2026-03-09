# 任务关闭标准

所有任务共用，不达标不进入下一任务。

| 维度          | 关闭条件                                       |
| ----------- | ------------------------------------------ |
| Correctness | 单测/对齐测试过；loss 不异常；旧功能回归通过                  |
| Layout      | 能画清楚 rank 上参数、激活、梯度、状态的分布                  |
| Comm        | 说清用的 collective、group、tensor shape、触发时机    |
| Overlap     | trace 里能看到通信与计算交叠，不靠口述                     |
| Metrics     | 有 step time / tokens/s / max mem / comm 占比 |
| Recovery    | 断点恢复可跑；至少保存 model/optim/RNG/step           |
| Failure     | 至少复盘 1 个该任务相关故障                            |
| Source      | 看过对应 Megatron/PyTorch/NCCL 源码入口并写笔记        |
| Interview   | 能 3 分钟讲机制，10 分钟讲取舍，30 分钟讲源码路径              |
