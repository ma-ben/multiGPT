# 教程：从一次训练 Step 读懂 multiGPT

这篇文档不是 changelog，而是教程。

目标很直接：带你从一次最小训练 step 出发，把这个仓库里的核心技术路径串起来。看完以后，你应该能回答这些问题：

1. 训练 step 是怎么推进的？
2. PP 为什么需要单独的通信流？
3. 为什么 checkpoint 不能只存模型权重？
4. eager attention 和 SDPA 在这个仓库里分别意味着什么？

---

## 1. 先建立最小心智模型

先不要一上来盯某个并行实现。

读这个仓库，建议先把它看成 4 层：

1. **启动层**
   - 负责读配置、检查拓扑、生成 `python -m torch.distributed.run` 启动命令
   - 入口文件：[launch.py](/Users/taom/Repo/multiGPT/launch.py)

2. **训练主循环**
   - 负责设备初始化、模型构建、数据加载、单步调度、日志与恢复
   - 入口文件：[train.py](/Users/taom/Repo/multiGPT/train.py)

3. **并行调度层**
   - 负责 TP / PP / DP / CP 的具体行为
   - 重点文件：
     - [pipeline_parallel.py](/Users/taom/Repo/multiGPT/picotron/pipeline_parallel/pipeline_parallel.py)
     - [tensor_parallel.py](/Users/taom/Repo/multiGPT/picotron/tensor_parallel/tensor_parallel.py)
     - [data_parallel.py](/Users/taom/Repo/multiGPT/picotron/data_parallel/data_parallel.py)

4. **运行时与恢复层**
   - 负责 profiler、step timeline、checkpoint、RNG 恢复
   - 重点文件：
     - [monitor.py](/Users/taom/Repo/multiGPT/picotron/monitor.py)
     - [checkpoint.py](/Users/taom/Repo/multiGPT/picotron/checkpoint.py)

---

## 2. 第一件事：先跑一遍最小命令

### 2.1 看启动命令

```bash
./.venv/bin/python launch.py --config configs/gpt2.yaml --dry-run
```

你会看到类似输出：

```text
env CUDA_DEVICE_MAX_CONNECTIONS=1 ./.venv/bin/python -m torch.distributed.run --master_port 29500 --nproc_per_node 2 train.py --config configs/gpt2.yaml
```

这里建议先关注两点：

1. `launch.py` 不再把配置路径写死，而是严格透传你传入的 `--config`
2. 启动前会先检查当前实现真正依赖的拓扑约束

例如：

- `vocab_size % tp_size == 0`
- `num_heads % tp_size == 0`
- `block_size % cp_size == 0`
- `num_layers >= pp_size`

这一步属于 `T0 主干封顶` 的典型动作：  
**先把最容易 silent wrong 的入口问题挡住。**

---

## 3. 第二件事：读懂训练主循环

训练主循环在 [train.py](/Users/taom/Repo/multiGPT/train.py)。

建议按函数读，而不是从头到尾机械扫文件。

### 3.1 `resolve_cuda_device_and_dtype`

这个函数先决定：

- 绑定哪张 CUDA 卡
- 用 `nccl`
- 用 `float32` 还是 `bfloat16`

它的意义不是“代码整洁”，而是把 CUDA 运行时约束集中起来。  
后面 profiler、显存统计、fused optimizer 全都依赖这个判断。

### 3.2 `build_data_loader`

现在数据入口支持两种模式：

- `random`
  - 用随机 token 做并行与吞吐基线
- `hf`
  - 用 HuggingFace dataset 验证真实训练闭环

为什么要保留 `random` 模式？

因为训练系统调优一开始常常不是先解决“数据质量”，而是先回答：

- TP/PP/DP 路径通不通
- step timeline 是否合理
- 通信是不是把算力吃掉了

随机 token 很适合做这个阶段的基线。

### 3.3 `build_model`

模型初始化顺序是：

1. meta init
2. 并行包装
3. materialize weights
4. 移到目标 device / dtype
5. 最后再包 DP

这其实是训练系统里非常典型的一条初始化链。

它对应的思想是：

**先确定张量布局，再真正占内存。**

如果你以后继续做大模型初始化、lazy materialization、DCP，这条链路非常重要。

### 3.4 `run_one_training_step`

这个函数把 PP 和非 PP 的 step 调度收口了。

这样主循环就可以更聚焦于状态机本身：

1. `zero_grad`
2. `compute`
3. `loss_sync`
4. `optimizer_step`
5. `runtime_reset`

你在日志里看到的 step timeline，就是这个状态机的时间投影。

---

## 4. 第三件事：怎么看 step timeline

现在每个 step 会打印类似这样的摘要：

```text
Timeline: step/zero_grad=0.11ms | step/compute=17.84ms | step/loss_sync=0.10ms | step/optimizer_step=1.16ms
```

这不是 profiler 的替代品，但非常适合作为第一层观测。

你可以先用它回答一些非常基础但重要的问题：

1. 时间主要耗在 compute 还是 optimizer？
2. `loss_sync` 是否异常放大？
3. 某次代码改动后，整体 step 时间有没有明显变化？

只有当这层信息还不够时，再打开 profiler 看更细的 trace。

---

## 5. 第四件事：为什么 PP 通信要单独用 stream

这里是这次改造里最有“训练系统味道”的部分。

看文件：[pp_communications.py](/Users/taom/Repo/multiGPT/picotron/pipeline_parallel/pp_communications.py)

### 5.1 原来有什么问题

最常见的伪异步写法是：

1. 发起 `isend/irecv`
2. 马上 `wait`
3. 再 `torch.cuda.synchronize()`

这会导致一个结果：

**逻辑上你写的是异步，时间线上你得到的还是串行。**

### 5.2 现在怎么改

现在的实现引入了三个概念：

1. **默认计算流**
   - forward / backward 继续在这里运行

2. **通信流**
   - P2P send/recv 尽量在这里排队

3. **event**
   - 计算流告诉通信流：“这个 tensor 现在可以发了”
   - 通信流告诉计算流：“这个 recv tensor 现在可以用了”

### 5.3 send-only 为什么可以延迟等待

对于 `send_forward`、`send_backward` 这种 send-only 操作，当前 rank 发出去以后，不一定要立刻在 host 侧等它彻底结束。

所以现在会：

1. 把 send 放到通信流
2. 保留 tensor 引用，避免生命周期过早结束
3. 在 step 尾部统一 `drain_pipeline_communications()`

这就是一个非常典型的训练系统写法：

**把“必须现在等”的等待和“可以晚一点收口”的等待分开。**

### 5.4 为什么这还不算 production 级 overlap

因为真正的 overlap 不只是“有 stream + event”。

你还需要继续回答：

1. send 是否真的和下一段 compute 重叠了
2. recv 的等待是否还卡在关键路径上
3. bubble 是算法决定的，还是实现造成的

所以这一步是 `T3` 的基础版，不是终点。

---

## 6. 第五件事：PP 前向为什么之前会错

看文件：[pipeline_parallel.py](/Users/taom/Repo/multiGPT/picotron/pipeline_parallel/pipeline_parallel.py)

PP 最容易犯的错之一，就是把“首段输入 token id”和“中间段 hidden states”混为一谈。

在 PP 里：

- 首段拿到的是 `input_ids`
- 中间段和尾段拿到的是 `hidden_states`

如果非首段还继续做 token embedding / position embedding，就等于把语义搞错了。

这次修复后：

- 首段才会构造 `wte + wpe`
- 非首段只消费上游传来的激活

这件事非常基础，但非常关键。  
因为一旦这里错了，后面的 profiler、吞吐、checkpoint 全都只是建立在错误结果上的“伪进展”。

---

## 7. 第六件事：attention backend 为什么值得单独做成开关

看文件：[gpt2.py](/Users/taom/Repo/multiGPT/picotron/model/gpt2.py)

现在 attention 支持两种后端：

- `eager`
- `sdpa`

### 7.1 `eager`

优点：

- 数学过程最透明
- 适合教学和调试
- 容易追 autograd 与张量形状

缺点：

- 性能一般
- mask / softmax / matmul 全都显式展开

### 7.2 `sdpa`

优点：

- 使用 PyTorch 官方 `scaled_dot_product_attention`
- 更接近现代训练代码的实现方式
- 在 CUDA 上通常能用到更优的后端

缺点：

- 抽象更高
- 不如 eager 那么适合从零推导每一步

### 7.3 为什么这一步重要

因为它刚好把 `DO.md` 里的两条路线接起来了：

- `L0/L1`: 你仍然能看懂 attention 的数学结构
- `L6`: 你开始接触“同样算子，后端不同，执行路径不同”的现实问题

---

## 8. 第七件事：小规模里为什么还要加 activation checkpointing / autocast / compile

很多人会觉得这些特性只有大模型才需要，其实不完全对。

在这个仓库里，它们的价值不只是“提速”或“省显存”，更重要的是让你更早接触现代训练主干的真实形态。

### 8.1 activation checkpointing

位置：[gpt2.py](/Users/taom/Repo/multiGPT/picotron/model/gpt2.py)

它的本质是：

- 前向时不长期保存所有中间激活
- 反向需要时再重算一次前向

这会把问题直接暴露出来：

- 你到底在拿显存换什么
- 反向为什么会更慢
- autograd 图和运行时生命周期是怎么联系的

### 8.2 autocast / bf16

位置：[train.py](/Users/taom/Repo/multiGPT/train.py)

这里的实现是：

- 由训练配置决定是否开启
- 开启时直接走 CUDA autocast

它体现的不是“一个开关加速”，而是训练系统里很常见的现实：

**同一份训练代码，数值路径会随着 device 和 dtype 改变。**

### 8.3 torch.compile

位置：[train.py](/Users/taom/Repo/multiGPT/train.py)

当前实现故意只在“单卡 CUDA dense 路径”允许开启 compile。

为什么不对所有并行路径都开？

因为：

- TP/PP/DP 包装里有 hook、自定义 autograd、通信语义
- compile 的收益和风险都更复杂
- 在教学仓库里，先把 dense compile 路径讲清楚更重要

这体现的是一个很实用的工程原则：

**先在最干净的路径上吃到收益，再决定是否把复杂运行时一起卷进去。**

---

## 9. 第八件事：为什么 checkpoint 一定要存 RNG

看文件：[checkpoint.py](/Users/taom/Repo/multiGPT/picotron/checkpoint.py)

很多人第一次做恢复时只存：

- model
- optimizer
- step

这还不够。

现在 checkpoint 里还存了：

- Python `random`
- NumPy RNG
- Torch CPU RNG
- Torch CUDA RNG

原因很简单：

如果不恢复 RNG，resume 以后虽然“能接着跑”，但后续随机序列已经变了。

这会影响：

- 数据顺序
- 随机 token
- dropout
- 某些初始化和采样路径

从训练系统视角，这叫：

**表面恢复成功，实际轨迹漂移。**

所以这次 checkpoint 采取的是 correctness-first 策略：

- 每个 global rank 都保存自己的 checkpoint 文件
- 文件会稍微多一点
- 但恢复语义最直接，也最适合教学

等你以后再做 DCP / async checkpoint，再去优化存储开销和跨拓扑恢复。

---

## 10. 第九件事：怎么自己做实验

### 9.1 实验一：看随机基线

配置：

```yaml
dataset:
  loader: "random"
```

目标：

- 看 step timeline 是否合理
- 看 TP/PP/DP 路径是否通

### 9.2 实验二：切到真实数据

配置：

```yaml
dataset:
  loader: "hf"
```

目标：

- 验证 tokenizer / dataset / collate / 恢复主线

### 9.3 实验三：比较 eager 和 sdpa

配置：

```yaml
model:
  attention_backend: "eager"
```

和：

```yaml
model:
  attention_backend: "sdpa"
```

目标：

- 对比 step 时间
- 用 profiler 看 attention 区域的 trace

### 9.4 实验四：打开 activation checkpointing

配置：

```yaml
model:
  activation_checkpointing: true
```

目标：

- 观察 step 时间是否上升
- 观察显存占用是否下降
- 理解“重算换显存”这件事

### 9.5 实验五：打开 autocast

配置：

```yaml
training:
  use_autocast: true
```

目标：

- 观察 autocast 是否改变 step 时间与显存曲线
- 对比开启前后的训练日志

### 9.6 实验六：单卡 dense 路径打开 compile

前提：

- `tp=1`
- `pp=1`
- `dp=1`
- `cp=1`
- 设备是 CUDA

配置：

```yaml
training:
  use_torch_compile: true
  torch_compile_mode: "reduce-overhead"
```

目标：

- 对比 compile 前后的 step 时间
- 理解为什么当前仓库不建议在复杂并行包装路径直接 compile

### 9.7 实验七：验证恢复

先跑出一个 checkpoint：

```yaml
checkpoint:
  save_frequency: 1
```

再设置：

```yaml
checkpoint:
  load_path: "你的 checkpoint 路径"
```

目标：

- 看是否能从保存点继续推进
- 看 step、token 计数是否连续

---

## 11. 最后该怎么继续往下学

如果你已经把这篇教程跑通了，建议下一步按这个顺序推进：

1. **继续做 T3**
   - 打开 profiler
   - 观察 warmup / steady / cooldown
   - 证明 overlap 到底发生在哪

2. **继续做 T8**
   - 让 checkpoint 存更多训练状态
   - 做更严谨的恢复回归

3. **再做 L5/L6**
   - 先看 SDPA trace
   - 再试 `torch.compile`
   - 最后再考虑 CUDA Graph

一句话总结这套路线：

**先把训练系统的“时间线、张量流、恢复语义”讲清楚，再去追更深的性能优化。**
