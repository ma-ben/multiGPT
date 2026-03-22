# 并行布局说明：参数、激活、梯度到底在哪

这份文档专门回答一个问题：

**在当前实现里，参数、激活、梯度和通信分别怎么分布？**

如果这件事说不清，TP/PP/DP/CP 就只剩名词，没有系统感。

---

## 1. 全局 rank 网格

当前仓库的 rank 网格顺序固定为：

```text
DP * PP * CP * TP
```

也就是：

```python
grid = torch.arange(world_size).view(dp_size, pp_size, cp_size, tp_size)
```

这意味着一个 global rank 的坐标解释顺序始终是：

1. `dp_rank`
2. `pp_rank`
3. `cp_rank`
4. `tp_rank`

---

## 2. TP：张量并行

### 2.1 当前切分对象

TP 主要切这些层：

- `q_proj / k_proj / v_proj`
- `out_proj`
- `mlp.up_proj / mlp.down_proj`
- `wte / wpe / lm_head`

### 2.2 语义

- `ColumnParallelLinear`
  - 按输出维切
- `RowParallelLinear`
  - 按输入维切
- `VocabParallelEmbedding`
  - 按词表维切

### 2.3 代价

TP 的核心代价来自：

- forward / backward 中的 all-reduce
- 最终输出 gather

### 2.4 你在代码里应该看到什么

- 参数不再是完整矩阵，而是分片
- 某些层输出是局部分片，需要 reduce / gather

---

## 3. PP：流水线并行

### 3.1 当前切分对象

PP 按 `num_layers` 均匀切分 transformer block。

### 3.2 stage 角色

- 首段：
  - 持有 `wte / wpe`
  - 输入是 `input_ids`

- 中间段：
  - 不持有 embedding
  - 输入是上游 `hidden_states`

- 尾段：
  - 持有 `ln_f / lm_head`
  - 负责计算 loss

### 3.3 激活流

PP 前向流：

```text
input_ids -> stage0 hidden_states -> stage1 hidden_states -> ... -> logits -> loss
```

PP 反向流：

```text
dLoss/dLogits -> stageN grad -> ... -> stage0 grad
```

### 3.4 当前运行时语义

- recv 必须立即等待，因为本地计算立刻依赖结果
- send-only 允许在通信流异步推进
- step 末尾统一 `drain_pipeline_communications()`

---

## 4. DP：数据并行

### 4.1 语义

DP 不切参数和激活，只复制模型副本。

每个 DP rank：

- 拿到不同 batch
- 算出自己的梯度
- 最后在 `cp_dp_group` 上同步梯度

### 4.2 当前实现

仓库里用的是 bucket 化 DP：

- 梯度先落到 bucket
- bucket 准备就绪后再 all-reduce

这比逐参数同步更接近真实训练系统。

---

## 5. CP：上下文并行

### 5.1 当前语义

CP 按 sequence 维切分输入。

也就是：

- 每个 rank 只看到自己那一段序列
- `seq_length_per_gpu = seq_length // cp_world_size`

### 5.2 代价

CP 的难点不在切分本身，而在 attention 依赖全序列上下文。

所以当前实现里引入了 ring 通信：

- rank 间循环交换 K/V
- 每个 rank 逐块完成注意力计算

---

## 6. 参数、激活、梯度三张表

### 6.1 参数

- TP：分片
- PP：按层切分
- DP：复制
- CP：通常不切参数

### 6.2 激活

- TP：部分层输出是分片形式
- PP：stage 间传递 hidden states
- DP：各 rank 独立持有自己的 batch 激活
- CP：按 sequence 切分

### 6.3 梯度

- TP：局部梯度需要 collective
- PP：通过 backward P2P 逐段回传
- DP：最终 all-reduce
- CP：与 DP 合并在 `cp_dp_group` 上做部分同步

---

## 7. 读 trace 时该怎么映射这些布局

如果你打开 profiler，建议先这样看：

1. 看 `train/*` 标签
   - 先识别一个 step 的大阶段

2. 看 `pp/*` 标签
   - 识别 warmup / steady / cooldown

3. 再对照这份布局文档问自己：
   - 这是哪个并行维度的通信？
   - 这里传的是参数、激活还是梯度？
   - 这个等待是算法决定的，还是实现造成的？

---

## 8. 一句总结

当前这份仓库的并行布局可以压缩成一句话：

**TP 切张量，PP 切层，DP 复制模型，CP 切序列；真正的系统难点在它们的边界和等待点上。**
