# Liger Embedding 阶段性优化总结

日期：2026 年 10 月 8 日。任务：[社区任务 1801](https://github.com/triton-lang/triton-ascend/issues/1801)。

本次提交保存已完成的算子修复、性能优化及复测证据，**不作为最终验收通过声明**。现有环境下 110 项测试通过；最新代码的默认完整基准为 9/12 配置达标，限定相同 CPU 范围的两次独立完整复测分别为 12/12、11/12。第二次仅 8192 tokens 的 forward 比值为 1.000123，仍不满足每项 ratio ≤ 1.0 的严格要求。

本总结是对原有 FINAL_OPTIMIZATION_REPORT、OPTIMIZATION_STRATEGY_REPORT、PERFORMANCE_BENCHMARK_REPORT、PROFILING_ANALYSIS_REPORT 及审核材料的阶段性补充。判断当前状态时应使用本次实际源码、环境及数据，不沿用历史报告中的通过结论。

## 社区要求与当前状态

| 验收项 | 当前证据与结论 |
| --- | --- |
| Triton-Ascend 实现及接口一致性 | 已移除 forward 原生 Embedding 回退；前向与反向核心计算保留 Triton。主机扩展只负责准备、派发和 autograd 衔接。目标版本兼容性待验证。 |
| FP16、BF16、FP32 前向和反向正确性 | 当前环境原始 Embedding 测试及新增回归测试共 110 项通过。未修改原始断言和容差。 |
| 任意索引 shape、布局及 padding_idx | 补充标量、空输入、转置、切片、三维排列、非连续梯度与权重测试；原始 padding 用例通过。测试覆盖不等于对所有输入穷举证明。 |
| 默认 benchmark 全部配置正常输出 | 完整运行均输出 48 条 CSV 记录；其中 Torch 与 Liger 的 speed 对比为 3 种模式 × 4 种输入，共 12 项。运行正常结束。 |
| 每项性能 ratio ≤ 1.0 | 未稳定满足；默认运行 9/12，CPU 亲和性两轮 12/12 和 11/12。不得忽略或四舍五入掉超限值。 |
| 指定软件版本及硬件平台 | liger 0.8.2、CANN 9.1.0 已核实；torch-npu 与 Triton-Ascend 版本不符。910B3 精确型号及 A2/A3/950 覆盖尚未完成核实或验证。 |
| 优化前后数据、profiling 和复测命令 | 本次提交包含逐配置数据、选取的 profiling 原始事件、测试日志和命令；历史基线的采样来源局限见下文。 |
| 上游 PR 与最终合入 | 本次仅向项目仓库阶段性提交；尚未向社区验收目标仓提交最终 PR。 |

## 实际环境

| 项目 | 本次实测 | 社区指定或要求 |
| --- | --- | --- |
| liger-kernel | 0.8.2 | 0.8.2 |
| torch / torch-npu | 2.7.1 / 2.7.1 | torch-npu 2.9.0 及匹配 torch |
| triton-ascend | 3.2.1 | 3.2.2 |
| CANN | 9.1.0，安装信息文件确认 | 9.1.0 |
| NPU 标识 | Ascend910_9382，aarch64 主机 | 性能采集要求 910B3，并覆盖 A2/A3/950 或说明限制 |
| CPU 调度范围 | 默认 0–639；诊断时仅该测试进程限制为 0–7 | CPU 亲和性不是本任务验收豁免条件 |

未更改全局软件环境或系统 CPU 策略。当前 setup.py 仍使用已有 2.7.1 / 3.2.1 版本约束；迁移到指定版本并复测是环境待办项。未将现有设备标识直接等同于已完成全部目标平台验证。

## 已完成的修复与优化

1. **去除原生 forward 回退。** 删除 no_grad 和 benchmark forward 分支中调用原生 embedding 的路径，并移除 benchmark 专用派发标记。新增 6 项派发测试，防止优化结果由原生算子回退产生。
2. **补齐非连续布局支持。** 在入口处理非连续索引、权重及反向梯度，保留输入 shape 对应的输出结构，确保梯度输出布局正确。48 项布局组合测试全部通过。
3. **减少主机侧重复工作。** 缓存编译与派发元数据，限制缓存大小；热路径使用当前张量地址及当前 NPU stream，不缓存用户张量或旧地址。保留 device、profiler hook、子类和特殊执行上下文的兼容处理。
4. **缩短主机派发与 autograd 路径。** 新增可选 C++ 主机扩展及延迟构建加载器。核心数学计算未迁入 C++，反向仍调用 Triton 实现；构建不可用时回到 Python/Triton 路径，其性能需单独验证。打包包含扩展源码，增加 ninja 依赖。
5. **调整大输入分块及地址计算。** 宽度 4096、两字节 dtype、tokens ≥ 8192 且输入和输出字节偏移不超过有符号 32 位范围时，使用带保护的 i32 mouter 路径（M=8、N=4096、core multiplier=4）。其他情况保留安全路径。缓存键包含权重行数，避免跨地址边界误复用。

原始 test/transformers/test_embedding.py 和 benchmark/scripts/utils.py 与修复前提交一致。benchmark 脚本仅删除了绕过 Triton 的派发标记，没有改动采样、计时、容差或通过阈值。

## Profiling 根因分析

在引入 C++ 主机扩展之前，对 1024 tokens 的冷缓存诊断采集了各 10 个设备 kernel 事件：Torch Gather 中位数为 20.9008125 微秒，Triton mouter 为 19.43075 微秒。Triton 的设备执行已更快，但对应 CPU profiler 范围 TORCH_EMBEDDING / LIGER_EMBEDDING 分别约为 83.395 / 139.775 微秒，说明端到端劣势不能仅用设备 kernel 时长解释，主机准备、Python/autograd 和派发值得优先优化。

这些 profiler 范围含观测开销，不能直接替代正式 benchmark 延迟，也不能把范围差值全部归因于某一个 Python 函数。原始事件节选及统计保存在 profiling_events.json，包含原始 trace 的 SHA256。profile_cold.py 和 parse_trace.py 保留采集与分析方法。

限制测试进程 CPU 范围后结果改善，但第二轮仍出现轻微超限。现有证据支持 CPU 调度条件会影响观测结果，尚不足以证明它是唯一原因。8192 forward 的性能余量仍不足，不能仅以噪声解释而判定通过。

## 逐配置性能对比

以下时间统一为毫秒，使用 P50，ratio = Liger / Torch。输入为 BF16、hidden_size=4096、vocab_size=128256，tokens 为 1024/2048/4096/8192。

“优化前”是 O3 开始时真实 Triton 路径的历史诊断基线，保存于 before_o3.json，关联源码快照为 embedding_before_o3.py.txt；不使用更早包含原生 forward 回退的数值。历史采样脚本的完整来源尚未追溯，因此该列不能视为已经满足同协议、同条件的正式 A/B 验收数据。不同批次调度条件有变化，不据此宣称精确的优化百分比。正式验收仍需在规定环境补采受控的优化前后数据。

“优化后默认”来自本次最终源码的完整原始 benchmark 运行 default_after.csv，未限定 CPU 范围；保留其中未通过的结果。

| 模式 | tokens | 前 Torch ms | 前 Liger ms | 前 ratio | 后 Torch ms | 后 Liger ms | 后 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 1024 | 0.022060 | 0.097840 | 4.435177 | 0.019070 | 0.038900 | 2.039853 |
| forward | 2048 | 0.043450 | 0.070660 | 1.626237 | 0.041970 | 0.039750 | 0.947105 |
| forward | 4096 | 0.084400 | 0.082450 | 0.976896 | 0.082450 | 0.082940 | 1.005943 |
| forward | 8192 | 0.162470 | 0.160550 | 0.988182 | 0.162020 | 0.162110 | 1.000556 |
| backward | 1024 | 4.401700 | 3.663540 | 0.832301 | 4.409200 | 3.664680 | 0.831144 |
| backward | 2048 | 4.429640 | 3.709430 | 0.837411 | 4.437280 | 3.718140 | 0.837932 |
| backward | 4096 | 4.488350 | 3.798680 | 0.846342 | 4.489620 | 3.806570 | 0.847860 |
| backward | 8192 | 4.593170 | 3.983160 | 0.867192 | 4.585530 | 3.993450 | 0.870881 |
| full | 1024 | 4.861800 | 4.122670 | 0.847972 | 4.859860 | 4.119650 | 0.847689 |
| full | 2048 | 5.327080 | 4.601790 | 0.863849 | 5.313180 | 4.602230 | 0.866191 |
| full | 4096 | 6.233980 | 5.563280 | 0.892412 | 6.236540 | 5.569060 | 0.892973 |
| full | 8192 | 8.064180 | 7.494300 | 0.929332 | 8.067860 | 7.489580 | 0.928323 |

### 相同 CPU 亲和性下的两轮独立复测

两轮均使用原始 benchmark 脚本，对所有 provider 采用相同的 CPU 0–7 限制，仅以 LIGER_BENCH_TARGET 区分输出文件。各轮均正常退出，产出 48 条记录。下表列出全部 12 个 speed 比值，逐 provider 的 P20/P50/P80 等原始字段保存在 affinity_run1.csv 和 affinity_run2.csv。

| 模式 | tokens | 第一轮 ratio | 第二轮 ratio |
| --- | ---: | ---: | ---: |
| forward | 1024 | 0.906496 | 0.906034 |
| forward | 2048 | 0.930888 | 0.949840 |
| forward | 4096 | 0.988092 | 0.983522 |
| forward | 8192 | 0.983277 | 1.000123 |
| backward | 1024 | 0.832299 | 0.832489 |
| backward | 2048 | 0.836535 | 0.836805 |
| backward | 4096 | 0.848427 | 0.847285 |
| backward | 8192 | 0.870977 | 0.868082 |
| full | 1024 | 0.847575 | 0.846730 |
| full | 2048 | 0.865484 | 0.864822 |
| full | 4096 | 0.893629 | 0.891965 |
| full | 8192 | 0.926918 | 0.927471 |

第一轮 12/12，第二轮 11/12。第二轮 8192 forward 的真实比值约 1.000123，慢约 0.0123%；保留为不通过。固定样本、交替顺序、共享权重的额外诊断为 4 档 × 10 轮均不慢于 Torch，数据保存在 controlled_after.json；其协议不同，不能替代默认验收基准。

## 正确性与工程验证

- 原始测试与 4 个新增回归测试文件合计 **110 passed、0 failed、0 error、1 warning**，耗时 62.84 秒；日志保存为 validation.log。
- 新增测试覆盖 int32/int64 索引、FP16/BF16/FP32、非连续输入及梯度、空与标量输入、派发、缓存限额与地址边界、流切换、设备上下文、autograd、no_grad/inference 及子类兼容。
- 原始断言和容差未修改；公共 benchmark 计时工具未修改。
- Ruff 与 git diff --check 已通过；wheel 构建和扩展源码打包检查已通过。本次报告提交不更改实现代码；源码校验值与上述测试记录一致。
- 110 项是本算子原始测试及新增回归测试的范围，不代表全项目所有模型训练和收敛测试均已执行。

## 复测命令

在项目根目录运行，先记录实际软件版本与 NPU 型号。仓库中的实际测试路径为 test/transformers；与 issue 的示意路径 tests/test_embedding.py 不同。

~~~bash
python -m pytest -q test/transformers/test_embedding.py \
  test/transformers/test_embedding_ascend_dispatch.py \
  test/transformers/test_embedding_ascend_layout.py \
  test/transformers/test_embedding_ascend_launch_cache.py \
  test/transformers/test_embedding_ascend_host.py

python benchmark/scripts/benchmark_embedding.py

# 独立诊断：必须向所有 provider 施加相同约束，不替代上一条默认基准
LIGER_BENCH_TARGET=embedding_stage_affinity_1 taskset -c 0-7 \
  python benchmark/scripts/benchmark_embedding.py
LIGER_BENCH_TARGET=embedding_stage_affinity_2 taskset -c 0-7 \
  python benchmark/scripts/benchmark_embedding.py

# 固定样本的诊断协议，与验收基准分开解释
python benchmark/data/embedding_stage_20261008/controlled_recheck.py \
  --output /tmp/embedding_stage_controlled.json --samples 100 --rounds 10 --warmup 64

sha256sum -c benchmark/data/embedding_stage_20261008/source.sha256
~~~

首次构建主机扩展需要可用的 C++ 编译器、PyTorch 头文件及 ninja；首次构建或 JIT 编译时间不属于热态性能数值。新环境应先确认实际选择的执行路径，再解释性能结果。复测应避免多个性能任务争用同一 NPU。

## 尚未完成与下一阶段

**算子优化问题：** 默认条件仍有 forward 配置超限；CPU 范围一致时 8192 forward 也未连续达标。下一步需增加实际访存和派发性能余量，并用多轮独立完整默认基准确认稳定性，不能筛选通过批次或放宽阈值。

**环境与验收覆盖问题：** 在隔离环境准备 Triton-Ascend 3.2.2、torch-npu 2.9.0 及匹配 torch；核实 910B3，并补充 A2/A3/950 覆盖或按要求说明平台限制。当前依赖约束需要相应调整后重新验证，不能直接复用旧环境结果。

**证据与交付问题：** 补全目标环境中同协议、同条件的优化前后测量，以及最终实现的 profiling 复测；达到全部要求后，再向社区指定的 Ascend/triton-ascend-kernels experimental 分支提交验收 PR。

## 提交与证据索引

实现代码基准提交：c67cc719e8102ff6f4d3a58bf652a8abc18d3d97。本次在此之上追加总结与证据，不重写已有优化提交。

数据目录：[benchmark/data/embedding_stage_20261008](benchmark/data/embedding_stage_20261008)。包含历史基线、默认完整 benchmark、两轮独立诊断 CSV、正确性日志、profiling 事件、诊断脚本、优化前源码快照及文件校验值。manifest.json 记录源码提交、环境和通过数量，SHA256SUMS 用于校验本次证据包。完整 profiler trace 仍保留在原测试服务器；仓库仅保存与 Embedding 分析相关的事件节选。
