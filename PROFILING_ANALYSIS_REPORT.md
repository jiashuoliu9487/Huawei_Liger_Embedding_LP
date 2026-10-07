# Ascend LigerEmbedding Forward Profiling 分析报告

## 1. 分析范围

- 项目：/root/Huawei_Liger_Embedding_LP
- 算子：Ascend LigerEmbedding
- 配置：Forward，输入形状 (1, 4096)，词表大小 102400，Embedding Dim 2048，dtype bfloat16
- Profiling 输出：profiling/liger_forward_bt4096/8b51cf04e5f8_29517_20261006015447056_ascend_pt
- 采集文件：operator_details.csv、trace_view.json、analysis.db、ascend_pytorch_profiler.db

## 2. Operator 级结果

| Operator | Host Total Duration (us) | Device Self Duration (us) | Device Total Duration With AI Core (us) |
|---|---:|---:|---:|
| LigerEmbeddingFunction（第 1 次记录） | 277.89 | 0 | 12.96025 |
| embedding_forward_kernel（第 1 次记录） | 12.89 | 12.96025 | 12.96025 |
| LigerEmbeddingFunction（第 2 次记录） | 295.68 | 0 | 14.12028125 |
| embedding_forward_kernel（第 2 次记录） | 13.13 | 14.12028125 | 14.12028125 |
| aten::empty / empty_tensor / aten::view | 约 10--24 | 0 | 0 |

Profiling 记录显示，设备端 kernel 只有约 13--14 微秒，而主机侧 LigerEmbeddingFunction 约 278--296 微秒。按单次记录估算，主机调度与 launch 相关开销占总耗时的绝大部分，单纯减少 kernel 内部计算不能解决当前 Forward 慢于 Baseline 的问题。

## 3. 当前 tile 与 launch 配置

源码 src/liger_kernel/ops/backends/_ascend/ops/embedding.py 对该形状选择：

- block_m = 6
- block_n = 2048
- use_mouter = False
- core_mult = 6
- total_blocks = ceil(4096 / 6) * ceil(2048 / 2048) = 683
- launch 使用 _launch_grid(num_cores, total_blocks, core_mult)

该配置来自 wide embedding 分支。当前维度较大，沿 Embedding Dim 使用单个 2048 元素 block，沿 token 方向使用较小的 block_m，导致需要调度约 683 个 block。是否应增大 block_m、降低 core_mult 或采用更少的 block 数，需要通过对照实验确认，不能仅凭静态代码推断。

## 4. 对性能目标的影响

此前 Ascend910B2 benchmark 中，Forward 有 5 个配置慢于 torch baseline；BT=4096 是其中最慢的配置之一，Liger/Torch 约 5.902 倍。当前 profiling 解释了该差距的主要方向：NPU kernel 本体执行很短，包装、内存准备和 kernel launch 的固定成本占比过高。

## 5. 尚未完成的验证

1. 尚未对同一形状的 torch.nn.Embedding Baseline 采集同等 profiler 数据，因此暂不能把每一项 launch 开销与 Baseline 一一对应。
2. 当前 profiler 输出没有导出足够的 PMU 带宽/指令利用率指标，暂不能声称存在具体的访存带宽瓶颈或某条 NPU 指令瓶颈。
3. 修改 tile、launch grid 或非融合访存路径前，需要先建立少量候选配置的 A/B 测试，确认既能降低 BT=4096 Forward，也不会使其他 Forward、Backward、Full 配置退化。

## 6. 下一步建议

下一步应先采集同形状 PyTorch Baseline 的 Forward profile，并用相同 warmup、采样次数和同步方式比较：主机侧调用耗时、kernel 数量、设备端耗时和内存准备开销。之后再针对 block_m、core_mult 与 launch grid 做最小范围的参数实验，最后重新跑 benchmark 脚本覆盖 Forward、Backward、Full 三种模式。
