import importlib

import pytest
import torch

from liger_kernel.ops import LigerEmbeddingFunction
from liger_kernel.transformers.experimental.embedding import LigerEmbedding
from liger_kernel.utils import infer_device


@pytest.mark.skipif(infer_device() != "npu", reason="Ascend dispatch regression")
@pytest.mark.parametrize("dtype,embedding_dim", [(torch.float32, 64), (torch.bfloat16, 4096)])
@pytest.mark.parametrize("mode", ["training", "no_grad", "benchmark_forward"])
def test_embedding_ascend_forward_dispatch(monkeypatch, dtype, embedding_dim, mode):
    assert LigerEmbeddingFunction.__module__ == "liger_kernel.ops.backends._ascend.ops.embedding"
    backend = importlib.import_module(LigerEmbeddingFunction.__module__)
    embedding = LigerEmbedding(32, embedding_dim).to(device="npu", dtype=dtype)
    indices = torch.tensor([[1, 2, 1, 7], [3, 0, 4, 6]], device="npu")
    expected = embedding.weight.detach()[indices]
    calls = []
    original_forward = backend.embedding_forward

    def tracked_forward(*args, **kwargs):
        calls.append(True)
        return original_forward(*args, **kwargs)

    def forbidden_native(*args, **kwargs):
        raise AssertionError("Liger Forward must not fall back to native embedding")

    monkeypatch.setattr(backend, "embedding_forward", tracked_forward)
    monkeypatch.setattr(torch.nn.functional, "embedding", forbidden_native)
    if mode == "benchmark_forward":
        # A legacy benchmark attribute must not change the implementation.
        embedding._benchmark_kernel_operation_mode = "forward"
    with torch.set_grad_enabled(mode == "training"):
        output = embedding(indices)
    torch.npu.synchronize()
    assert len(calls) == 1
    assert output.requires_grad == (mode == "training")
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
