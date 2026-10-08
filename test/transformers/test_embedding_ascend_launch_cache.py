import importlib

import pytest
import torch

from triton.compiler.compiler import CompiledKernel

from liger_kernel.utils import infer_device

device = infer_device()
pytestmark = pytest.mark.skipif(device != "npu", reason="Ascend launch-cache regression")
backend = importlib.import_module("liger_kernel.ops.backends._ascend.ops.embedding") if device == "npu" else None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("offset", [0, 1])
def test_cached_launch_uses_current_tensors_and_stream(dtype, index_dtype, offset):
    backend._clear_forward_launch_cache()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    previous = None
    previous_expected = None
    for step in range(2):
        with torch.npu.stream(stream):
            storage = torch.arange(32 * 64 + offset, device="npu", dtype=torch.float32)
            storage = (storage.remainder(13) + step).to(dtype)
            weight = storage[offset:].reshape(32, 64)
            index_storage = torch.arange(8 + offset, device="npu", dtype=index_dtype).remainder(4)
            indices = index_storage[offset:].reshape(2, 4)
            expected = torch.nn.functional.embedding(indices, weight)
            output = backend.LigerEmbeddingFunction.apply(weight, indices)
        stream.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        if previous is not None:
            assert output.data_ptr() != previous.data_ptr()
            torch.testing.assert_close(previous, previous_expected, atol=0, rtol=0)
        previous, previous_expected = output, expected
    assert len(backend._FORWARD_LAUNCHERS) == 1


def test_cached_launch_preserves_profiler_hooks(monkeypatch):
    weight = torch.randn((32, 64), device="npu")
    indices = torch.arange(8, device="npu")
    backend.embedding_forward(weight, indices)
    events = []
    monkeypatch.setattr(CompiledKernel, "launch_enter_hook", lambda metadata: events.append("enter"))
    monkeypatch.setattr(CompiledKernel, "launch_exit_hook", lambda metadata: events.append("exit"))
    actual = backend.embedding_forward(weight, indices)
    torch.npu.synchronize()
    assert events == ["enter", "exit"]
    torch.testing.assert_close(actual, weight[indices], atol=0, rtol=0)


def test_cached_launch_preserves_jit_pre_run_hooks(monkeypatch):
    weight = torch.randn((32, 64), device="npu")
    indices = torch.arange(8, device="npu")
    backend.embedding_forward(weight, indices)
    calls = []
    monkeypatch.setattr(backend.embedding_forward_kernel, "pre_run_hooks", [lambda *a, **k: calls.append(True)])
    actual = backend.embedding_forward(weight, indices)
    torch.npu.synchronize()
    assert calls == [True]
    torch.testing.assert_close(actual, weight[indices], atol=0, rtol=0)


def test_launch_cache_is_bounded(monkeypatch):
    backend._clear_forward_launch_cache()
    monkeypatch.setattr(backend, "_FORWARD_LAUNCH_CACHE_SIZE", 2)
    weight = torch.randn((32, 64), device="npu")
    for count in (2, 3, 4):
        indices = torch.arange(count, device="npu")
        torch.testing.assert_close(backend.embedding_forward(weight, indices), weight[indices], atol=0, rtol=0)
        assert len(backend._FORWARD_LAUNCHERS) <= 2
        if backend._HOST_EXTENSION:
            assert backend._HOST_EXTENSION.cache_size() <= 2
