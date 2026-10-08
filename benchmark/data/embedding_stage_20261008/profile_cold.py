import time

import torch
import torch_npu
import triton

from liger_kernel.transformers.experimental.embedding import LigerEmbedding

w = torch.nn.Parameter(torch.randn((128256, 4096), device="npu", dtype=torch.bfloat16))
with torch.device("meta"):
    native = torch.nn.Embedding(128256, 4096)
    liger = LigerEmbedding(128256, 4096)
native.weight = w
liger.weight = w
ids = torch.randint(0, 128256, (1, 1024), device="npu")
for _ in range(100):
    native(ids)
    liger(ids)
torch.npu.synchronize()
for name, model in [("torch", native), ("liger", liger)]:
    t = time.perf_counter()
    for _ in range(1000):
        model(ids)
    submit = time.perf_counter() - t
    torch.npu.synchronize()
    print("HOST_SUBMIT_US", name, submit * 1e3, flush=True)
cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/tmp/embedding_o3_profile_cold_20261008"),
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
) as prof:
    for step in range(12):
        for name, model in [("TORCH_EMBEDDING", native), ("LIGER_EMBEDDING", liger)]:
            cache.zero_()
            with torch.autograd.profiler.record_function(name):
                model(ids)
            torch.npu.synchronize()
        prof.step()
print("PROFILE_COMPLETE", flush=True)
