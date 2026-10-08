"""Diagnostic only: keep the community acceptance benchmark unchanged.

Run on the Ascend host after checking its environment and current source.
Use fixed sample counts, common data/storage, and alternating provider order.
This does not turn a confidence interval or an averaged result into acceptance.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import statistics

from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

from liger_kernel.ops import LigerEmbeddingFunction
from liger_kernel.transformers.experimental.embedding import LigerEmbedding
from liger_kernel.utils import infer_device


def percentiles(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    p20, p50, p80 = torch.quantile(tensor, torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)).tolist()
    return dict(samples=len(values), p20_ms=p20, p50_ms=p50, p80_ms=p80)


def sample_forward(fn, device_interface, cache, samples, warmup):
    for _ in range(warmup):
        fn()
    device_interface.synchronize()
    starts = [device_interface.Event(enable_timing=True) for _ in range(samples)]
    ends = [device_interface.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        # Same ordering as triton.testing.do_bench, with a fixed sample count.
        cache.zero_()
        start.record()
        fn()
        end.record()
    device_interface.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def paired_summary(ratios):
    rng = random.Random(20261008)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {
        "paired_round_ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "diagnostic_bootstrap_95pct": [draws[125], draws[4874]],
        "warning": "Diagnostic interval only; official per-configuration ratio criterion remains unchanged.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    args = parser.parse_args()
    assert args.samples >= 20 and args.rounds >= 2
    assert infer_device() == "npu"
    assert LigerEmbeddingFunction.__module__ == "liger_kernel.ops.backends._ascend.ops.embedding"
    torch.set_grad_enabled(True)
    torch.manual_seed(args.seed)
    torch.npu.synchronize()
    output = Path(args.output)
    sources = [
        "src/liger_kernel/ops/backends/_ascend/ops/embedding.py",
        "src/liger_kernel/transformers/experimental/embedding.py",
        "benchmark/scripts/benchmark_embedding.py",
        "benchmark/scripts/utils.py",
        "src/liger_kernel/ops/backends/_ascend/ops/_embedding_host.cpp",
        "src/liger_kernel/ops/backends/_ascend/ops/_embedding_host.py",
    ]
    result = {
        "protocol": "diagnostic_fixed_samples_shared_storage_alternating_order",
        "parameters": vars(args),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torch-npu", "triton-ascend", "liger-kernel")
        },
        "device": torch.npu.get_device_name(0),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "TASK_QUEUE_ENABLE",
                "ASCEND_LAUNCH_BLOCKING",
                "TRITON_DEBUG",
                "TRITON_COMPILE_ONLY",
                "TRITON_REGISTER_TENSOR_MSPROF",
                "OMP_NUM_THREADS",
            )
        },
        "cases": [],
    }
    weight = torch.nn.Parameter(torch.randn((128256, 4096), device="npu", dtype=torch.bfloat16))
    with torch.device("meta"):
        native = torch.nn.Embedding(128256, 4096)
        liger = LigerEmbedding(128256, 4096)
    native.weight = weight
    liger.weight = weight
    interface = triton.runtime.driver.active.get_device_interface()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    for bt in args.tokens:
        indices = torch.randint(0, weight.shape[0], (bt // 1024, 1024), device="npu")
        torch.testing.assert_close(liger(indices), native(indices), atol=0, rtol=0)
        functions = {"torch": lambda: native(indices), "liger": lambda: liger(indices)}
        case = {"bt": bt, "rounds": []}
        for round_index in range(args.rounds):
            order = ["torch", "liger"] if round_index % 2 == 0 else ["liger", "torch"]
            measurements = {}
            for name in order:
                times = sample_forward(functions[name], interface, cache, args.samples, args.warmup)
                measurements[name] = dict(percentiles(times), raw_ms=times)
            ratio = measurements["liger"]["p50_ms"] / measurements["torch"]["p50_ms"]
            case["rounds"].append(dict(order=order, measurements=measurements, ratio=ratio))
            print(json.dumps(dict(bt=bt, round=round_index, order=order, ratio=ratio)), flush=True)
        case["summary"] = paired_summary([entry["ratio"] for entry in case["rounds"]])
        result["cases"].append(case)
        output.write_text(json.dumps(result, indent=2))
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
