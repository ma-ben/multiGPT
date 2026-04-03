有TP,PP,DP并行框架的基础，一些代码层面的基础知识不必重复
想要理解清楚整个技术栈从代码到硬件的性能瓶颈



必须精通
L0 训练循环与状态机:forward/backward/optimizer/AMP/checkpoint /grad accum
L1autograd(自动求导):graph、hook、saved tensor、inplace、view、backward 调度
L2 dispatcher/ATen:op如何选后端、tensor metadata、stride/layout/contiguous
L3 CUDA runtime(运行时)/stream/event/allocator/async 语义
L4 distributed(分布式):DDP/FSDP/ZeRO/NCCL/bucket /overlap /topology
L5 graph-level execution(图级执行):CUDA Graph、静态化、地址稳定、生命周期管理
必须能看懂
L6 compiler path(编译链):torch.compile/Inductor/Triton/PTX，至少知道代码怎么落到 kernel
L7 kernel execution(核执行):grid/block/warp/occupancy/divergence/coalescing
L8 memory hierarchy(存储层级):HBM/L2/shared memory/registers 对性能的影响L9 interconnect(互连):PCle/NVLink/IB 对 TP/DP/PP 的约束