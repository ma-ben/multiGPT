# 故障手册：multiGPT 常见问题怎么排

这份文档的目标不是列出所有错误，而是把最常见、最值得优先判断的几类问题列成手册。

读法建议：

1. 先看现象
2. 再看最可能的根因
3. 最后按检查步骤收敛

---

## 1. `world size` 或 rank 拓扑不匹配

### 现象

- 启动后立刻报错
- `ProcessGroupManager` 的 world size assert 失败

### 常见根因

- 启动进程数和 `tp * pp * cp * dp` 不一致
- 手工改了配置，但没改启动命令

### 检查步骤

1. 先跑：

```bash
./.venv/bin/python launch.py --config configs/gpt2.yaml --dry-run
```

2. 再跑：

```bash
./tools/topology_check.sh configs/gpt2.yaml
```

---

## 2. PP 路径 loss 异常或行为不稳定

### 现象

- TP 路径正常，PP 路径 loss 异常
- 不同 PP stage 行为不一致

### 常见根因

- 非首段错误地重复做了 embedding / position embedding
- micro-batch 边界不一致
- 某个 stage 没有正确 drain send-only 通信

### 检查步骤

1. 看 rank 日志里 `rank-mapping`
2. 跑 PP smoke：

```bash
./tools/smoke_tests.sh
```

3. 打开 profiler 看 `pp/*` 标签时间线

---

## 3. trace 里看起来“一切都在等”

### 现象

- profiler 里 PP/CP 区域全是等待
- compute 和通信几乎没有交叠

### 常见根因

- 在通信后用了全局 `torch.cuda.synchronize()`
- recv 等待放在了关键路径上
- send-only 没有延迟收口

### 检查步骤

1. 先确认通信 helper 没被改回 blocking 版本
2. 运行：

```bash
./tools/profile_run.sh configs/gpt2.yaml 29520
```

3. 打开 `tmp/profiler` 对应 trace，重点看：
   - `pp/1f1b/send_fwd_recv_bwd`
   - `pp/1f1b/send_bwd_recv_fwd`
   - `step/compute`

---

## 4. checkpoint 能加载，但训练轨迹不连续

### 现象

- resume 后能继续训练
- 但行为和“同一步连续跑”明显不同

### 常见根因

- 没有恢复 RNG
- load 后 optimizer state 还在 CPU
- topology 和保存时不一致

### 检查步骤

1. 跑恢复演示：

```bash
./tools/restore_demo.sh
```

2. 检查 checkpoint 里是否包含：
   - `rng_state`
   - `topology`
   - `optimizer`

---

## 5. `torch.compile` 没生效

### 现象

- 配置里开了 `use_torch_compile: true`
- 日志仍显示“跳过 torch.compile”

### 这通常不是 bug

当前实现只建议在：

- 单卡
- CUDA
- dense
- 无 TP/PP/DP/CP 包装

这条路径上开启 compile。

### 检查步骤

1. 看配置里是否全是 `1`
2. 看设备是不是 CUDA
3. 看日志里是否打印：

```text
启用 torch.compile
```

---

## 6. 没有 CUDA 时训练会直接失败

### 现象

- 启动 `launch.py` 或 `train.py`
- 很早就报“当前训练入口只支持 CUDA”

### 原因

当前主干已经移除了 CPU 训练路径，只保留 `nccl + cuda`。
这属于预期行为，不是故障。

---

## 7. `address already in use`

### 现象

- `torch.distributed.run` 报端口占用

### 根因

- 多个实验同时使用默认 `29500`

### 解决方式

直接换端口：

```bash
./.venv/bin/python launch.py --config configs/gpt2.yaml --master-port 29511
```

或者工具脚本里传第二个参数作为端口。

---

## 8. 一句话排障原则

如果你只记一条原则，就记这一条：

**先确认拓扑和状态机正确，再看 trace，再谈性能。**

因为训练系统里最危险的情况不是“慢”，而是“错得很安静”。 
