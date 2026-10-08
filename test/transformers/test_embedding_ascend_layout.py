import pytest
import torch

from liger_kernel.ops import LigerEmbeddingFunction
from liger_kernel.utils import infer_device


def make_indices(layout, dtype):
    base = torch.tensor([1, 3, 1, 0, 4, 2], device="npu", dtype=dtype)
    if layout == "vector":
        return base
    if layout == "matrix":
        return base.reshape(2, 3)
    if layout == "transpose":
        return base.reshape(2, 3).t()
    if layout in ("slice_vector", "slice_matrix"):
        storage = torch.tensor([1, 7, 3, 8, 1, 9, 0, 10, 4, 11, 2, 12], device="npu", dtype=dtype)
        return storage[::2] if layout == "slice_vector" else storage.reshape(2, 6)[:, ::2]
    if layout == "permute_3d":
        return base.repeat(2).reshape(2, 2, 3).permute(2, 0, 1)
    if layout == "scalar":
        return base[0]
    if layout == "empty":
        return torch.empty((2, 0, 3), device="npu", dtype=dtype)
    raise AssertionError(layout)


def make_weight(layout):
    base = torch.arange(32 * 60, device="npu", dtype=torch.float32).remainder(17).reshape(32, 60) / 8
    if layout == "transpose":
        return base.t().contiguous().t().requires_grad_()
    if layout == "slice":
        storage = torch.empty((32, 120), device="npu")
        storage[:, ::2].copy_(base)
        return storage[:, ::2].requires_grad_()
    return base.requires_grad_()


@pytest.mark.skipif(infer_device() != "npu", reason="Ascend layout regression")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "index_layout", ["vector", "matrix", "transpose", "slice_vector", "slice_matrix", "permute_3d", "scalar", "empty"]
)
@pytest.mark.parametrize("weight_layout", ["contiguous", "transpose", "slice"])
def test_embedding_ascend_layout(index_dtype, index_layout, weight_layout):
    assert LigerEmbeddingFunction.__module__ == "liger_kernel.ops.backends._ascend.ops.embedding"
    indices = make_indices(index_layout, index_dtype)
    weight = make_weight(weight_layout)
    reference_weight = weight.detach().clone().requires_grad_()
    expected = torch.nn.functional.embedding(indices, reference_weight)
    output = LigerEmbeddingFunction.apply(weight, indices)
    assert output.shape == (*indices.shape, weight.shape[1])
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    # Exercise non-contiguous upstream gradients as well as repeated indices.
    grad_storage = torch.arange(2 * output.numel(), device="npu", dtype=torch.float32).remainder(7)
    grad = grad_storage.reshape(*output.shape[:-1], 2 * output.shape[-1])[..., ::2]
    output.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-6, rtol=1e-5)
