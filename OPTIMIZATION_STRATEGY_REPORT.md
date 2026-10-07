# LigerEmbedding 昇腾优化策略与结果报告

> 生成日期：2026-10-07
> 项目：Huawei_Liger_Embedding_LP
> 任务：完成 liger-kernel 中 LigerEmbedding 算子的昇腾亲和支持，并验证性能

## 1. 任务目标

在 Ascend NPU 上为 LigerEmbedding 提供可用的昇腾后端，并围绕 issue #1801 完成性能优化。对照基线为 torch.nn.Embedding，测试脚本为 benchmark/scripts/benchmark_embedding.py，覆盖 Forward、Backward、Full 三种 kernel_operation_mode。严格门槛是每个配置下 Liger Kernel 耗时都不慢于 PyTorch Baseline，并附逐配置性能数据和 Profiling 分析。

## 2. 系统环境

| 项目 | 当前环境 |
| --- | --- |
| 硬件 | 华为 Ascend NPU |
| Triton-Ascend | 3.2.2 |
| CANN | 9.1.0 |
| torch-npu | 2.7.1.post8 |
| Python | 3.11 |
| 项目路径 | /root/Huawei_Liger_Embedding_LP |
| 模型与数据 | /workspace |
| Benchmark | benchmark/scripts/benchmark_embedding.py |

环境检查显示 NPU、CANN、Triton-Ascend、torch-npu 和 Python 依赖均可用，Ascend 后端可以导入并实际运行。此前 clone 的 liger-kernel 缺少可直接使用的依赖；本轮验证基于已经准备好依赖的项目环境。

## 3. PyTorch 对照测试

### 3.1 方法

使用同一组词表大小、hidden size、index 分布、dtype、设备、warmup 和重复次数。Baseline 使用 torch.nn.Embedding，候选实现使用 LigerEmbedding Ascend 后端；每个模式都在 NPU 同步后统计耗时。原始数据和对照记录见：PYTORCH_COMPARISON_REPORT.md、FINAL_BENCHMARK_RAW.txt、FINAL_CANDIDATE_RAW.txt、FINAL_NATIVE_RAW.txt、FINAL_WRAPPER_RAW.txt。

### 3.2 摘要

| 模式 | 与 PyTorch 的差距 | 结论 |
| --- | --- | --- |
| Forward | 大多数配置快于或接近 Baseline；BT=8192 约慢 0.42% | 严格门槛未完全通过 |
| Backward | 已记录配置整体快于或不慢于 Baseline | 通过 |
| Full | 已记录配置整体快于或不慢于 Baseline | 通过 |

完整 benchmark 中已有 10 组配置通过；已知回退是 Forward、BT=8192，候选版本约慢 0.42%。差值可能受到 NPU 调度和随机访存抖动影响，但在重复实验确认前不能视为通过。因此当前结论是功能和大部分性能已验证，严格的“所有配置不慢于 Baseline”尚未满足。

## 4. 算子问题

1. Embedding table 按 index 随机读行，访存难以合并；hidden size 越大，读取和写回成本越高。
2. Forward 的 kernel launch 固定开销在小 token 配置中占比明显。
3. 宽 hidden size 使用固定 tile/Block 会增加寄存器和 UB 压力，出现尾块浪费、并行度不足或调度抖动。
4. Backward 必须处理重复 index 的 atomic_add，随机写和原子竞争限制吞吐。
5. 非融合路径存在 wrapper、重复 reshape、临时 Tensor 和布局转换，增加 kernel 数量与内存流量。
6. core_mult 和 grid 映射对大 hidden size、大 BT 敏感，单一配置不能覆盖全部形状。
7. BT=8192 Forward 的小幅回退仍未消除，profiling 指向大 tile 随机读取、输出写回和调度的组合影响。

## 5. 优化方案

### 5.1 Tile/Block、core_mult 与 grid

按 token 数和 hidden size 选择 row/block 划分；宽 hidden size 使用更细的 hidden 维切分，减少尾块和寄存器/UB 压力；大 BT 调整 core_mult 与 grid，让工作均匀分布到 NPU 核心。

### 5.2 减少 Forward launch

合并索引解码、行读取和输出写回，减少阶段性 kernel。对已经是目标布局的输入跳过重复 reshape 和转换，降低小规模调用的 launch 固定成本。

### 5.3 消除 wrapper 和临时 Tensor

Ascend 路径直接进入后端实现，复用已知形状的输出和中间描述，移除重复 view/reshape、无效布局复制和不必要的临时 Tensor，减少 Python 调度与设备端分配。

### 5.4 非融合路径访存

按 row/block 组织读取和写回，使同一 tile 内 hidden 维访问连续；保持 index 和输出布局稳定，避免为适配 wrapper 进行全量搬运。

### 5.5 Backward 与指令级调优

保留 atomic_add 以保证重复 index 下的正确性，通过 block 划分减少无效加载和写回；围绕向量化连续访存、尾块 mask、core 利用率和大 hidden size 重新选择参数。每次改动都重跑完整矩阵，防止已经通过的配置回退。

## 6. Benchmark 与 Profiling

标准命令：

    cd /root/Huawei_Liger_Embedding_LP
    python benchmark/scripts/benchmark_embedding.py

测试覆盖 Forward、Backward、Full，并使用与 PyTorch 相同的输入和重复设置。Profiling 观察 kernel launch 数量、gather 随机读取带宽、输出写回、Backward atomic_add 冲突、core 利用率以及 tile 尾块和 UB/register 压力。已有分析见 PROFILING_ANALYSIS_REPORT.md 和 PROFILING_BASELINE_COMPARISON_REPORT.md；瓶颈与 benchmark 差距一致，集中在随机访存、launch 固定成本、原子累加和宽 hidden size 资源分配。

## 7. 优化结果

| 项目 | 结果 |
| --- | --- |
| Ascend 后端功能 | 已能导入并运行，Forward/Backward/Full 均有实测路径 |
| Forward | 大多数配置达到或优于 PyTorch；BT=8192 约慢 0.42% |
| Backward | 已记录配置达到或优于 PyTorch |
| Full | 已记录配置达到或优于 PyTorch |
| 已通过配置 | 10 组配置未观察到回退 |
| 严格性能门槛 | 尚未完全满足，剩余 BT=8192 Forward 回退 |
| Profiling | 已完成瓶颈定位，结论与 benchmark 差距一致 |

当前候选版本减少了非必要调度和临时 Tensor，改善了连续输出写回，并使 tile、core_mult、grid 能按形状适配。Backward 和 Full 已达到当前测试门槛；剩余问题集中在大 BT 的 Forward，而不是功能正确性或 NPU 环境。

## 8. 后续工作

当前不能宣称 issue 的严格门槛全部完成。下一轮应集中在 BT=8192 Forward：重复独立批次以区分真实回退和调度噪声；比较不同 hidden tile、core_mult、grid 的 kernel 数、带宽和端到端耗时；检查仍可合并的短 kernel、wrapper 或临时 Tensor；每次改动后重跑 Forward、Backward、Full 全部配置。只有 BT=8192 Forward 稳定不慢于 PyTorch 且 profiling 支持该结论后，才能在 PR 中声明严格门槛通过。

## 9. 提交材料建议

应提交 Ascend 后端实现、必要的注册/导出修改和对应测试。benchmark 原始日志、profiling 导出和过程报告可按仓库规范作为复现附件；torch_compile_debug、重复运行产生的临时 raw 文件和环境缓存不应作为算子源代码提交。本报告结论基于当前工作区已有记录，BT=8192 Forward 的 0.42% 回退是当前唯一已知严格门槛阻塞项。
