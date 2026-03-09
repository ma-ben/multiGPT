# T5. CP 真接线

先把现有 CP 从“有文件”变成“真正在模型路径里生效”。

**建议窗口：D21-D24**

| 子维度       | 任务                                                                                                                                                        |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Impl      | 把 `ring_attention` 真正接进 attention 路径；明确 `apply_context_parallel` 需要改哪些模块，而不是只设环境变量                                                                        |
| Comm/NCCL | 画出 CP rank 上 token 范围；明确 attention 前后需要的交换；补一版 cp size sweep                                                                                              |
| Correctness | 先做 `CP=2` 的最小 correctness case，确认输出 shape、loss、causal mask、RoPE/position 处理正确                                                                           |
| Trace     | 至少给 3 张图：CP off/on、不同 seq len、不同 tp/cp 组合                                                                                                                 |
| Recovery  | CP 模式 checkpoint + 恢复；确保 sequence 分片状态不影响恢复 correctness                                                                                                   |
| Source    | 读：Megatron CP guide、相关 multimodal CP API、你自己的 attention 路径                                                                                                |
| Failure   | 复盘 1 个 CP 问题：attention 特殊交换没处理、seq padding/packing 错、CP 与 TP layout 冲突                                                                                     |
| 交付        | `reports/t5_cp_wiring.md`、CP/TP tradeoff 表、CP tensor layout 图                                                                                              |
| 通过标准      | 你能明确说出“当前仓库的 CP 到底接到了哪些模块，没接到哪些模块”                                                                                                                          |

[1]: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html?utm_source=chatgpt.com "Mixture of Experts — Megatron Core"
