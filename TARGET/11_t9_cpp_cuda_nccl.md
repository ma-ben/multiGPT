# T9. C++/CUDA + NCCL drill

不是附加项，是把“会用框架”抬到“会切入底层”。

**建议窗口：D29-D30**

| 子维度       | 任务                                                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Impl      | 只写一个真正进热路径的 op：`fused bias+dropout+residual` / `token permutation` / `RoPE` 三选一；PyTorch 官方把 custom C++/CUDA op 定义成推荐路径。([PyTorch Docs][1])                         |
| Comm/NCCL | 建 `nccl_debug.sh`：支持 `NCCL_DEBUG={WARN,INFO,TRACE}`、`NCCL_DEBUG_FILE`、`NCCL_DEBUG_SUBSYS`；官方说明里 `WARN` 会在错误返回前打印显式告警，`TRACE` 给可重放级调用信息，日志可定向到文件。([NVIDIA Docs][2]) |
| Overlap   | 如果 op 进入热路径，重新测一次是否改变 overlap 窗口；否则这个 op 价值很低                                                                                                                       |
| Trace     | op 前后 benchmark；至少知道它是否真的进入瓶颈段                                                                                                                                      |
| Recovery  | 新 op 不破坏 checkpoint/恢复；扩展加载流程文档齐全                                                                                                                                   |
| Source    | 读：PyTorch custom op 教程、dispatcher/registration 入口、NCCL debug 文档                                                                                                     |
| Failure   | 复盘 1 个底层问题：binding 错、ABI/编译问题、NCCL hang/timeout 分层排查                                                                                                                |
| 交付        | `cpp_ops/`、`tools/nccl_debug.sh`、`tools/topology_check.sh`、`docs/failure_playbook.md`                                                                               |
| 通过标准      | 你能从 Python 调到 C++/CUDA，再解释这个 op 为什么值得存在；同时能拿 NCCL 日志定位到错误层级                                                                                                         |

[1]: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html?utm_source=chatgpt.com "Custom C++ and CUDA Operators"
[2]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html?utm_source=chatgpt.com "Environment Variables — NCCL 2.28.9 documentation"
