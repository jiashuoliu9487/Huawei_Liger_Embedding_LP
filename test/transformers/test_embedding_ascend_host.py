import gc
import importlib
import weakref

import pytest
import torch

from liger_kernel.utils import infer_device

device = infer_device()
pytestmark = pytest.mark.skipif(device != "npu", reason="Ascend native host regression")
backend = importlib.import_module("liger_kernel.ops.backends._ascend.ops.embedding") if device == "npu" else None


def test_native_autograd_is_active():
    weight = torch.randn((16, 64), device="npu", requires_grad=True)
    ids = torch.arange(4, device="npu")
    out = backend.LigerEmbeddingFunction.apply(weight, ids)
    assert backend._HOST_EXTENSION
    assert backend._HOST_EXTENSION.cache_size() > 0
    assert "HostEmbedding" in out.grad_fn.name()
    torch.testing.assert_close(out, weight[ids], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("transpose", [False, True])
def test_native_host_backward(dtype, index_dtype, transpose):
    weight = torch.randn((16, 64), device="npu", dtype=dtype)
    if transpose:
        weight = weight.t().contiguous().t()
    weight.requires_grad_()
    reference = weight.detach().clone().requires_grad_()
    ids = torch.tensor([1, 3, 1, 5], device="npu", dtype=index_dtype)
    actual = backend.LigerEmbeddingFunction.apply(weight, ids)
    expected = torch.nn.functional.embedding(ids, reference)
    grad = torch.arange(actual.numel() * 2, device="npu").remainder(5).to(dtype) - 2
    grad = grad.reshape(4, 128)[:, ::2]
    actual.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference.grad, atol=0, rtol=0)


def test_native_backward_on_side_stream():
    weight = torch.randn((16, 64), device="npu", requires_grad=True)
    reference = weight.detach().clone().requires_grad_()
    ids = torch.tensor([1, 3, 1, 5], device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        actual = backend.LigerEmbeddingFunction.apply(weight, ids)
        actual.square().sum().backward()
    stream.synchronize()
    expected = torch.nn.functional.embedding(ids, reference)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_native_without_grad(mode):
    weight = torch.randn((16, 64), device="npu", requires_grad=True)
    ids = torch.arange(4, device="npu")
    with mode():
        actual = backend.LigerEmbeddingFunction.apply(weight, ids)
        expected = weight[ids]
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_function_subclass_preserves_override():
    class OffsetEmbedding(backend.LigerEmbeddingFunction):
        @staticmethod
        def forward(ctx, weight, indices):
            return backend.LigerEmbeddingFunction.forward(ctx, weight, indices) + 3

    weight = torch.randn((16, 64), device="npu", requires_grad=True)
    reference = weight.detach().clone().requires_grad_()
    ids = torch.tensor([1, 3, 1, 5], device="npu")
    actual = OffsetEmbedding.apply(weight, ids)
    expected = torch.nn.functional.embedding(ids, reference) + 3
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference.grad, atol=0, rtol=0)


def test_native_metadata_does_not_retain_inputs():
    def run():
        weight = torch.randn((16, 64), device="npu")
        indices = torch.arange(4, device="npu")
        refs = weakref.ref(weight), weakref.ref(indices)
        output = backend.LigerEmbeddingFunction.apply(weight, indices)
        torch.npu.synchronize()
        return refs, output

    refs, output = run()
    gc.collect()
    assert refs[0]() is None and refs[1]() is None
    assert output.shape == (4, 64)


def test_native_handles_other_device_context():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two visible NPUs")
    original = torch.npu.current_device()
    other = 1 if original == 0 else 0
    weight = torch.randn((16, 64), device=f"npu:{other}", requires_grad=True)
    reference = weight.detach().clone().requires_grad_()
    indices = torch.tensor([1, 3, 1, 5], device=f"npu:{other}")
    actual = backend.LigerEmbeddingFunction.apply(weight, indices)
    actual.sum().backward()
    with torch.npu.device(other):
        expected = torch.nn.functional.embedding(indices, reference)
        expected.sum().backward()
        torch.npu.synchronize()
    assert torch.npu.current_device() == original
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference.grad, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_large_forward_uses_guarded_i32_kernel(dtype, index_dtype):
    backend._clear_forward_launch_cache()
    weight = torch.randn((32, 4096), device="npu", dtype=dtype)
    indices = torch.randint(0, 32, (8, 1024), device="npu", dtype=index_dtype)
    actual = backend.LigerEmbeddingFunction.apply(weight, indices)
    expected = torch.nn.functional.embedding(indices, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    plan = next(iter(backend._FORWARD_LAUNCHERS.values()))
    assert plan[0] is backend.embedding_forward_kernel_mouter_i32


@pytest.mark.parametrize(
    "count,dim,size,rows,expected",
    [
        (8192, 4096, 2, 262144, True),
        (8192, 4096, 2, 262145, False),
        (262145, 4096, 2, 32, False),
        (1024, 4096, 2, 32, False),
        (8192, 4096, 4, 32, False),
        (8192, 2048, 2, 32, False),
    ],
)
def test_i32_byte_offset_boundaries(count, dim, size, rows, expected):
    assert backend._use_i32_forward_offsets(count, dim, size, rows) is expected


def test_native_cache_distinguishes_vocabulary_size():
    backend._clear_forward_launch_cache()
    indices = torch.arange(4, device="npu")
    for rows in (16, 32):
        weight = torch.randn((rows, 64), device="npu")
        actual = backend.LigerEmbeddingFunction.apply(weight, indices)
        torch.testing.assert_close(actual, weight[indices], atol=0, rtol=0)
    assert len(backend._FORWARD_LAUNCHERS) == 2
    assert backend._HOST_EXTENSION.cache_size() == 2
