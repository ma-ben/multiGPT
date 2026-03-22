# multiGPT 主干技术优化说明

## 1. 这次改了什么

这次优化不是把仓库直接改造成“大而全训练框架”，而是按 `DO.md` 的主干路线，先补最关键的四类能力：

1. **L0 训练循环可解释**
   - 把 `train.py` 拆成了设备初始化、数据加载、模型构建、优化器构建、checkpoint 恢复、单步训练几个清晰阶段。
   - 每个 step 现在都会输出 `zero_grad -> compute -> loss_sync -> optimizer_step -> runtime_reset` 的时间线摘要。

2. **L0/L4 并行正确性**
   - 修复了 PP 非首段重复注入位置编码的问题。
   - 修复了 `launch.py` 中配置路径写死、拓扑校验错位、`--debug` 布尔参数无效的问题。

3. **L3/L4 基础观测能力**
   - 引入 `StepTimer`，给日志补充人类可读的 step 级时间线。
   - 引入可选 `torch.profiler`，并在 train/PP 热路径用 `record_function` 打了阶段标签。
   - 去掉了 PP/CP 通信里不必要的全局 `cuda synchronize`，并补了 `comm stream + event` 的基础结构。

4. **L0/T8 恢复主干**
   - 训练入口现在支持同拓扑 checkpoint 恢复和周期性保存。
   - checkpoint 新增 RNG 状态恢复。

5. **L6 最小后端升级**
   - attention 支持 `eager` 与 `sdpa` 两种后端切换。
   - 单卡 CUDA dense 路径支持可选 `torch.compile`。

6. **小规模最佳实践增强**
   - 支持可选 `autocast/bf16`
   - 支持可选 activation checkpointing

---

## 2. 对应代码位置

### 2.1 启动器

文件：`launch.py`

主要变化：

- `--config` 真的会传给 `train.py`
- `--debug` 改为 `store_true`
- 拓扑检查调整为和当前代码实现一致：
  - `vocab_size % tp_size == 0`
  - `num_heads % tp_size == 0`
  - `block_size % cp_size == 0`
  - `num_layers >= pp_size`
- 用 `subprocess.run(..., check=True)` 代替 `os.system`

这一步主要服务于 `T0 主干封顶`，先把最基础的可复现性补起来。

### 2.2 训练主循环

文件：`train.py`

主要变化：

- 把训练入口整理成几个函数：
  - `resolve_cuda_device_and_dtype`
  - `build_data_loader`
  - `build_model`
  - `build_optimizer`
  - `resume_training_from_checkpoint`
  - `run_one_training_step`
- 训练日志里新增 step 时间线
- 支持 `random` 和 `hf` 两种 data loader 模式
- 支持可选 profiler
- 支持 checkpoint 保存与恢复

### 2.3 PP 正确性与可观测性

文件：`picotron/pipeline_parallel/pipeline_parallel.py`

主要变化：

- 只有首段 stage 会做 token embedding + position embedding
- 中间段与尾段只消费上游传来的 hidden states
- 在 `AFAB` 和 `1F1B` 路径上补了 `record_function` 标签
- send-only 的 P2P 操作可以先在通信流里异步推进，step 尾部再统一 drain

这一步对应 `DO.md` 里的：

- `L0`: forward/backward 状态机
- `L4`: PP 调度与通信边界
- `T3`: 让 PP 先变得正确、可解释，再谈 overlap

### 2.4 通信层最小优化

文件：

- `picotron/pipeline_parallel/pp_communications.py`
- `picotron/context_parallel/cp_communications.py`

主要变化：

- 移除不必要的全局 `torch.cuda.synchronize()`
- 给 PP 通信加入专用 `comm stream`
- 用 `event` 显式表达“计算流 -> 通信流 -> 计算流”的依赖关系

注意：这还不是“真正的 async overlap 完成版”，但它先消掉了一个最明显的伪异步反模式。下一步如果要继续深入，建议做：

1. comm stream
2. event 依赖
3. wait 点压缩
4. trace 验证 overlap

### 2.5 轻量监控模块

文件：`picotron/monitor.py`

新增了两个工具：

- `StepTimer`: 人类可读的阶段计时
- `profiling_context`: 条件式 profiler 上下文

设计目标是“少量代码，把 step timeline 先做出来”，而不是把仓库拖进复杂监控框架。

### 2.6 Attention 后端切换

文件：`picotron/model/gpt2.py`

主要变化：

- 支持 `attention_backend: eager | sdpa`
- `eager` 用来保留最原始的数学路径
- `sdpa` 用来接入 PyTorch 官方 attention 后端

这是当前代码里最轻量的一步 `L6` 落地：  
同一个 attention 结构，开始走不同执行后端。

### 2.8 小规模训练增强

文件：

- `train.py`
- `picotron/model/gpt2.py`
- `configs/gpt2.yaml`

主要变化：

- 非 PP 与 PP 路径都支持可选 `autocast`
- `GPT` 支持 activation checkpointing
- 单卡 dense 路径支持可选 `torch.compile`

这部分的设计原则是：

**尽量把“小规模能明显受益的现代训练特性”接进来，但不破坏当前分布式教学主干。**

### 2.7 Checkpoint 恢复加强

文件：`picotron/checkpoint.py`

主要变化：

- checkpoint 由“部分 rank 保存”改成“每个 global rank 一个文件”
- 新增 Python / NumPy / Torch CPU / Torch CUDA RNG 状态
- load 时会校验 topology，并把 optimizer state 迁回模型设备

这一步的取舍很明确：  
先优先保证恢复语义完整，再去优化 checkpoint 的体积和异步化。

---

## 3. 新增配置

文件：`configs/gpt2.yaml`

新增字段：

```yaml
dataset:
  loader: "random"  # random | hf

profiling:
  enabled: false
  output_dir: "tmp/profiler"
  wait: 1
  warmup: 1
  active: 2
  repeat: 1
  record_shapes: true
  profile_memory: true
  with_stack: false
```

### 3.1 `dataset.loader`

- `random`
  - 用随机 token 驱动训练
  - 更适合并行机制验证、吞吐基线和通信路径观察

- `hf`
  - 用真实 HuggingFace dataset
  - 更适合验证训练闭环、tokenizer、checkpoint 恢复

### 3.2 `profiling.enabled`

打开后仅在 `global_rank == 0` 的进程上导出 trace，避免所有 rank 同时写大量 profiler 文件。

---

## 4. 最小使用方式

### 4.1 只看启动命令

```bash
python3 launch.py --config configs/gpt2.yaml --dry-run
```

### 4.2 正常启动

```bash
python3 launch.py --config configs/gpt2.yaml
```

### 4.3 打开 profiler

把配置改成：

```yaml
profiling:
  enabled: true
```

运行后，trace 会输出到：

```text
tmp/profiler/
```

### 4.4 切到真实数据

把配置改成：

```yaml
dataset:
  loader: "hf"
```

---

## 5. 这次优化刻意没有做什么

为了控制代码量，这次**没有**继续往下做下面这些更重的主题：

1. `torch.compile / CUDA Graph`
2. Flash Attention / SDPA 替换
3. comm stream + event 的真实 overlap 改造
4. DCP / async checkpoint
5. 多节点 topology-aware 调度

原因不是这些不重要，而是当前仓库先要把 `T0 -> T3 -> T8` 的主干补稳。否则后面做得越深，返工越多。

---

## 6. 建议你下一步怎么继续

如果继续按 `DO.md` 往下推进，我建议顺序是：

1. **先做 T3**
   - 用 profiler 看 `1F1B` 的 warmup / steady / cooldown
   - 把 `pp_communications.py` 改成 `comm stream + event`
   - 证明 overlap 是真的，而不是口头上的

2. **再做 T8**
   - 把现在的 checkpoint 过渡到更完整的 state 管理
   - 补 RNG state
   - 验证恢复后的 loss 连续性

3. **最后再做 L5/L6**
   - 先接 `scaled_dot_product_attention`
   - 再试 `torch.compile`
   - 最后再看 CUDA Graph

---

## 7. 这次改动的核心原则

一句话总结：

**先让训练系统“正确、可恢复、可观察”，再去追更深的性能技术。**

这也是 `DO.md` 里最重要的一条隐含原则：  
没有证据链的性能优化，最后大概率只是把错误跑得更快。
