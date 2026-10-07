# LigerEmbedding 昇腾优化最终报告

## 结论

已完成 LigerEmbedding 的 Ascend kernel 接入、调度优化和正确性验证。Forward 使用项目 Triton kernel，不再回退到 NPU 原生 embedding；Forward 去除无必要 contiguous，Backward 保留必要 contiguous；针对宽 hidden size 增加 Mouter 路径、UB 约束下的 tile 选择和 core/grid 调度。

正确性测试：pytest -q test/transformers/test_embedding.py
结果：11 passed in 6.09s
语法检查：py_compile 通过

## 最终调度

文件：src/liger_kernel/ops/backends/_ascend/ops/embedding.py

- 宽 hidden size 使用 M5 tile（BLOCK_SIZE_M=min(5, max_m)）。
- BT=2048 使用 Mouter + core multiplier 2；BT=4096/8192 使用 2D 路径 + core multiplier 6。
- Forward 通过 _launch_grid 限制实际 grid，减少无效 launch。
- Backward 大 hidden size 使用 core multiplier 8。

## Benchmark 结果

官方脚本：python benchmark/scripts/benchmark_embedding.py --overwrite。单位为 ms，token 数为 [1024, 2048, 4096, 8192]。以下为 M5 + core6 的完整回归（BENCH_M5_RAW.txt）：

| 模式 | Torch | Liger | 结论 |
|---|---|---|---|
| Forward | [0.08478, 0.04418, 0.08402, 0.16184] | [0.08466, 0.04368, 0.08380, 0.16252] | 1024/2048/4096 通过；8192 慢约 0.42% |
| Backward | [4.41076, 4.43923, 4.48648, 4.58607] | [3.66802, 3.71208, 3.80261, 3.98929] | 4/4 通过 |
| Full | [4.85318, 5.31206, 6.23862, 8.06792] | [4.12423, 4.61089, 5.56406, 7.47830] | 4/4 通过 |

当前实测通过 11/12 个 Torch 对照配置；唯一缺口是 Forward、hidden size=4096、BT=8192，差异约 0.42%。该差异接近设备运行抖动量级，但按 issue 的严格不慢于 Baseline 要求，不能宣称已经无条件满足。

## 候选优化验证

- M4：BT=1024 明显退化，撤销。
- core multiplier 8：4096/8192 改善不稳定，1024/2048 可能回退，撤销。
- Mouter 全路径：收益不足且小 token 有抖动，未采用。
- Mouter 4-core：BT=2048 约慢 18%，撤销。
- Mouter 1-core：BT=1024 约慢 8.5%，撤销。
- M5：相比 M6 在宽 hidden size 上更稳定，是当前最终实现。

## Profiling 分析

已有 profiling 结果显示主要耗时来自 embedding table 的随机行读取和 output 写回；Forward 的 kernel launch 与宽 tile 的 UB/register 压力是小 token 配置的主要影响因素；Backward 受 atomic_add 访存限制但整体稳定快于 Torch。剩余 Forward BT=8192 差异主要来自大 tile 下的访存和调度抖动，继续提高 core multiplier 未形成稳定收益。

## 后续建议

若 PR 审核将 0.42% 差异按硬门槛处理，需要在相同 NPU 空闲状态下重复多轮 benchmark，报告中使用中位数和置信区间；若仍超基线，再结合 CANN profiler 对 8192 配置做指令级访存和 core 映射分析。当前代码和报告保留了所有候选原始数据，未修改无关 profiling 产物。