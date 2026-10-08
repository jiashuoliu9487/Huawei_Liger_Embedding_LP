import os

from functools import lru_cache

import torch
import torch_npu
import triton
import triton.language as tl

from triton.backends.ascend import utils as ascend_runtime_utils
from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops.backends._ascend.ops._embedding_host import load_host_extension
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count

# Ascend UB capacity is 192 KB.
_ASCEND_UB_CAPACITY_BITS = 1572864
_UB_MULTIPLIER = 3.2
_UB_SAFETY = 0.85

# Wide embeddings (embedding_dim >= threshold) use large N tiles tuned on Ascend910 bf16.
_WIDE_EMBEDDING_THRESHOLD = 2048
_FORWARD_MAX_BLOCK_N = 512
_FORWARD_WIDE_BLOCK_N_CANDIDATES = (4096, 2048, 1024, 512)

# Ascend910 bf16 2D forward compile limits: max BLOCK_SIZE_M per BLOCK_SIZE_N.
_FORWARD_COMPILED_MAX_M = {
    4096: 6,
    2048: 12,
    1024: 18,
    512: 37,
    256: 75,
    128: 126,
    64: 128,
}


@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim
        block_mask = mask_m[:, None] & mask_n[None, :]

        embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        embeddings = tl.load(
            embeddings_ptr + embedding_offsets,
            mask=block_mask,
            other=0.0,
        )

        output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)


@triton.jit
def embedding_forward_kernel_mouter(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)

    for block_m in tl.range(pid, grid_m, num_progs):
        start_m = block_m * BLOCK_SIZE_M

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        for block_n in tl.range(0, grid_n):
            start_n = block_n * BLOCK_SIZE_N

            offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
            mask_n = offsets_n < embedding_dim
            block_mask = mask_m[:, None] & mask_n[None, :]

            embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
            embeddings = tl.load(
                embeddings_ptr + embedding_offsets,
                mask=block_mask,
                other=0.0,
            )

            output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)


@triton.jit
def embedding_forward_kernel_mouter_i32(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)

    for block_m in tl.range(pid, grid_m, num_progs):
        start_m = block_m * BLOCK_SIZE_M

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0).to(tl.int32)

        for block_n in tl.range(0, grid_n):
            start_n = block_n * BLOCK_SIZE_N

            offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
            mask_n = offsets_n < embedding_dim
            block_mask = mask_m[:, None] & mask_n[None, :]

            embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
            embeddings = tl.load(
                embeddings_ptr + embedding_offsets,
                mask=block_mask,
                other=0.0,
            )

            output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)


@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)

    for block_m in tl.range(pid, grid_m, num_progs):
        start_m = block_m * BLOCK_SIZE_M

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        for block_n in tl.range(0, grid_n):
            start_n = block_n * BLOCK_SIZE_N
            offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
            offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
            mask_n = offsets_n < embedding_dim
            block_mask = mask_m[:, None] & mask_n[None, :]

            grad_output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
            grad_output = tl.load(
                grad_output_ptr + grad_output_offsets,
                mask=block_mask,
                other=0.0,
            )

            grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
            tl.atomic_add(
                grad_weight_ptr + grad_weight_offsets,
                grad_output,
                mask=block_mask,
            )


@triton.jit
def embedding_backward_kernel_2d(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim
        block_mask = mask_m[:, None] & mask_n[None, :]

        grad_output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        grad_output = tl.load(
            grad_output_ptr + grad_output_offsets,
            mask=block_mask,
            other=0.0,
        )

        grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + grad_weight_offsets,
            grad_output,
            mask=block_mask,
        )


def _max_block_m_for_ub(
    block_size_n: int,
    dtype_size: int,
    multiplier: float = _UB_MULTIPLIER,
    safety_margin: float = _UB_SAFETY,
) -> int:
    usable_bits = int(_ASCEND_UB_CAPACITY_BITS * safety_margin)
    bits_per_m = max(1, int(multiplier * block_size_n * dtype_size * 8))
    return max(1, min(usable_bits // bits_per_m, 128))


def _clamp_forward_block_m(block_m: int, block_n: int, dtype_size: int) -> int:
    compile_max_m = _FORWARD_COMPILED_MAX_M.get(block_n)
    if compile_max_m is None:
        compile_max_m = max(
            16,
            int(_ASCEND_UB_CAPACITY_BITS * 0.99 / (82.5 * block_n * max(dtype_size, 1) / 2)),
        )
    ub_max_m = _max_block_m_for_ub(block_n, dtype_size)
    return max(1, min(block_m, compile_max_m, ub_max_m))


def _largest_wide_forward_block_n(embedding_dim: int, dtype_size: int) -> int:
    for block_n in _FORWARD_WIDE_BLOCK_N_CANDIDATES:
        if block_n > embedding_dim:
            continue
        if _clamp_forward_block_m(4, block_n, dtype_size) >= 4:
            return block_n
    return _FORWARD_MAX_BLOCK_N


def _select_forward_tile_sizes(
    n_elements: int,
    embedding_dim: int,
    dtype_size: int,
) -> tuple[int, int]:
    if embedding_dim >= _WIDE_EMBEDDING_THRESHOLD:
        block_n = _largest_wide_forward_block_n(embedding_dim, dtype_size)
        max_m = _clamp_forward_block_m(128, block_n, dtype_size)
        block_m = min(5, max_m)
        return max(1, block_m), block_n
    best_key = None
    best_m = 64
    best_n = triton.next_power_of_2(min(128, embedding_dim))

    for block_n in (_FORWARD_MAX_BLOCK_N, 256, 128, 64):
        if block_n > embedding_dim:
            continue
        block_n = triton.next_power_of_2(block_n)
        max_m = _max_block_m_for_ub(block_n, dtype_size)
        if max_m < 16:
            continue

        block_m = min(max_m, triton.next_power_of_2(min(128, n_elements)))
        while block_m > max_m and block_m > 1:
            block_m //= 2
        block_m = _clamp_forward_block_m(block_m, block_n, dtype_size)
        if block_m < 1:
            continue

        grid_m = triton.cdiv(n_elements, block_m)
        grid_n = triton.cdiv(embedding_dim, block_n)
        n_bias = block_n if embedding_dim >= 1024 else 0
        key = (-(block_m * block_n), -n_bias, grid_m * grid_n)
        if best_key is None or key < best_key:
            best_key = key
            best_m, best_n = block_m, block_n

    return best_m, best_n


def _pick_forward_schedule(
    n_elements: int,
    embedding_dim: int,
    block_n: int,
) -> tuple[bool, int]:
    if embedding_dim < _WIDE_EMBEDDING_THRESHOLD:
        return False, 2 if n_elements >= 2048 else 1

    grid_n = triton.cdiv(embedding_dim, block_n)
    if grid_n == 1:
        if n_elements >= 8192:
            return False, 6
        if n_elements >= 4096:
            return False, 6
        if n_elements >= 2048:
            return True, 2
        return True, 2

    if n_elements <= 1024:
        return True, 3
    if n_elements <= 2048:
        return False, 1
    return True, 1


@lru_cache(maxsize=256)
def _get_optimal_block_m(n_elements: int, dtype_size: int, block_size_n: int) -> int:
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=_UB_SAFETY,
        dtype_size=dtype_size,
        memory_multiplier=_UB_MULTIPLIER,
        shapes=((n_elements, block_size_n),),
        tiling_dims=(0,),
    )
    block_m = tile_shapes[0][0] if tile_shapes else triton.next_power_of_2(min(128, n_elements))
    return min(block_m, _max_block_m_for_ub(block_size_n, dtype_size))


@lru_cache(maxsize=256)
def _select_backward_tile_sizes(
    n_elements: int,
    embedding_dim: int,
    dtype_size: int,
) -> tuple[int, int]:
    if embedding_dim >= _WIDE_EMBEDDING_THRESHOLD:
        block_n = min(_FORWARD_MAX_BLOCK_N, triton.next_power_of_2(embedding_dim))
        max_m = _max_block_m_for_ub(block_n, dtype_size)
        return _clamp_forward_block_m(max_m, block_n, dtype_size), block_n

    if n_elements >= 4096 and embedding_dim >= 256:
        block_n = 256
    else:
        block_n = triton.next_power_of_2(min(128, embedding_dim))
    return _get_optimal_block_m(n_elements, dtype_size, block_n), block_n


@lru_cache(maxsize=256)
def _get_forward_launch_config(n_elements: int, embedding_dim: int, dtype_size: int):
    block_m, block_n = _select_forward_tile_sizes(n_elements, embedding_dim, dtype_size)
    use_mouter, core_mult = _pick_forward_schedule(n_elements, embedding_dim, block_n)
    if embedding_dim == 4096 and dtype_size == 2 and n_elements >= 8192:
        block_m, block_n, use_mouter, core_mult = 6, 4096, True, 2
    return block_m, block_n, use_mouter, core_mult


def _launch_grid(num_cores: int, total_blocks: int, core_multiplier: int = 1) -> int:
    return max(1, min(num_cores * core_multiplier, total_blocks))


# Cache compilation/launch metadata only, never tensors or data pointers.
_FORWARD_LAUNCHERS = {}
_FORWARD_LAUNCH_CACHE_SIZE = 256


def _use_i32_forward_offsets(n_elements, embedding_dim, dtype_size, num_embeddings):
    # Bound both input and output byte offsets, not only the index values.
    return (
        embedding_dim == 4096
        and dtype_size == 2
        and n_elements >= 8192
        and max(n_elements, num_embeddings) * embedding_dim * dtype_size <= (1 << 31)
    )


def _embedding_forward_python(embeddings, indices):
    ori_shape = indices.shape
    # Kernels address indices and embedding rows using contiguous offsets.
    indices = indices.contiguous()
    embeddings = embeddings.contiguous()
    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]
    device = embeddings.device
    dtype = embeddings.dtype
    if device.type != "npu" or indices.device != device:
        raise ValueError("Ascend embedding requires weight and indices on the same NPU")
    if torch.npu.current_device() != device.index:
        with torch.npu.device(device):
            return _embedding_forward_python(embeddings, indices)
    output = torch.empty((*ori_shape, embedding_dim), device=device, dtype=dtype)
    if n_elements == 0:
        return output

    weight_ptr = embeddings.data_ptr()
    indices_ptr = indices.data_ptr()
    # Triton specializes pointer alignment as well as scalar values and dtypes.
    key = (
        device.index,
        dtype,
        indices.dtype,
        n_elements,
        embedding_dim,
        embeddings.shape[0],
        weight_ptr % 16,
        indices_ptr % 16,
        os.environ.get("TRITON_DEBUG", "0"),
    )
    plan = _FORWARD_LAUNCHERS.get(key)
    if plan is None:
        block_m, block_n, use_mouter, core_mult = _get_forward_launch_config(
            n_elements, embedding_dim, embeddings.element_size()
        )
        use_i32 = _use_i32_forward_offsets(n_elements, embedding_dim, embeddings.element_size(), embeddings.shape[0])
        if use_i32:
            block_m, block_n, use_mouter, core_mult = 8, 4096, True, 4
        total_blocks = triton.cdiv(n_elements, block_m)
        if not use_mouter:
            total_blocks *= triton.cdiv(embedding_dim, block_n)
        grid = _launch_grid(get_npu_core_count(), total_blocks, core_mult)
        kernel = embedding_forward_kernel_mouter if use_mouter else embedding_forward_kernel
        if use_i32:
            kernel = embedding_forward_kernel_mouter_i32
        options = dict(embedding_dim=embedding_dim, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n)
        compiled = kernel[(grid,)](embeddings, indices, output, n_elements, **options)
        if compiled is not None and hasattr(compiled, "run"):
            runtime_launcher = compiled.run
            raw_launch = getattr(runtime_launcher, "launch", None)
            raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
            plan = (
                kernel,
                options,
                compiled,
                compiled[(grid, 1, 1)],
                runtime_launcher,
                raw_launch,
                raw_stream,
                compiled.function,
                compiled.packed_metadata,
                grid,
            )
            if len(_FORWARD_LAUNCHERS) >= _FORWARD_LAUNCH_CACHE_SIZE:
                _FORWARD_LAUNCHERS.pop(next(iter(_FORWARD_LAUNCHERS)))
            _FORWARD_LAUNCHERS[key] = plan
        return output

    kernel, options, compiled, runner, runtime_launcher, raw_launch, raw_stream, function, metadata, grid = plan
    if kernel.pre_run_hooks:
        kernel[(grid,)](embeddings, indices, output, n_elements, **options)
    elif (
        raw_launch is not None
        and raw_stream is not None
        and CompiledKernel.launch_enter_hook is None
        and CompiledKernel.launch_exit_hook is None
        and not runtime_launcher.compile_only
        and not runtime_launcher.enable_msprof_register_tensor
    ):
        # These addresses come from live, same-device NPU tensors. The supported
        # uint64 launcher arguments avoid repeating aclrtPointerGetAttributes.
        # Fetch the current stream for every call; do not cache a stream handle.
        profiler_registered = raw_launch(
            grid,
            1,
            1,
            raw_stream(device.index),
            function,
            metadata,
            None,
            None,
            None,
            weight_ptr,
            indices_ptr,
            output.data_ptr(),
            n_elements,
        )
        ascend_runtime_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1
    else:
        # Preserve runtime diagnostics, profiler tensor metadata and launch hooks.
        runner(embeddings, indices, output, n_elements)
    return output


_HOST_EXTENSION = None
_STANDARD_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _host_context_supported(embeddings, indices):
    return (
        type(embeddings) in _STANDARD_TENSOR_TYPES
        and type(indices) is torch.Tensor
        and not torch._C._are_functorch_transforms_active()
        and not torch._C._get_tracing_state()
        and not torch.compiler.is_compiling()
    )


def _prepare_host_plan(embeddings, indices):
    output = _embedding_forward_python(embeddings, indices)
    key = (
        embeddings.device.index,
        embeddings.dtype,
        indices.dtype,
        indices.numel(),
        embeddings.shape[1],
        embeddings.shape[0],
        embeddings.data_ptr() % 16,
        indices.data_ptr() % 16,
        os.environ.get("TRITON_DEBUG", "0"),
    )
    return output, _FORWARD_LAUNCHERS.get(key)


def _host_backward(embeddings, indices, grad_output):
    if torch.npu.current_device() != embeddings.device.index:
        with torch.npu.device(embeddings.device):
            return embedding_backward(embeddings, indices, grad_output)
    return embedding_backward(embeddings, indices, grad_output)


def _initialize_embedding_host():
    global _HOST_EXTENSION
    if _HOST_EXTENSION is None:
        # Other threads can use the Python path while the extension is built.
        _HOST_EXTENSION = False
        extension = load_host_extension()
        if extension is not None:
            extension.configure(
                _prepare_host_plan,
                _host_backward,
                CompiledKernel,
                getattr(torch_npu._C, "_npu_getDevice", torch.npu.current_device),
                lambda: _FORWARD_LAUNCH_CACHE_SIZE,
                ascend_runtime_utils,
            )
            _HOST_EXTENSION = extension
    return _HOST_EXTENSION


def _clear_forward_launch_cache():
    _FORWARD_LAUNCHERS.clear()
    if _HOST_EXTENSION:
        _HOST_EXTENSION.clear_cache()


def embedding_forward(embeddings, indices, *, _autograd=False):
    if _autograd:
        # The Function entry checked support and initialized the host dispatcher.
        return _HOST_EXTENSION.apply(embeddings, indices)
    if _host_context_supported(embeddings, indices):
        host = _HOST_EXTENSION
        if host is None:
            host = _initialize_embedding_host()
        if host:
            return host.forward(embeddings, indices)
    return _embedding_forward_python(embeddings, indices)


def embedding_backward(embeddings, indices, grad_output):
    indices = indices.contiguous().view(-1)
    grad_output = grad_output.contiguous().view(-1, embeddings.shape[1])

    grad_weight = torch.zeros_like(embeddings, memory_format=torch.contiguous_format)

    n_elements = indices.numel()
    embedding_dim = embeddings.shape[1]

    if n_elements == 0:
        return grad_weight

    block_m, block_n = _select_backward_tile_sizes(n_elements, embedding_dim, embeddings.element_size())
    num_cores = get_npu_core_count()
    use_2d = embedding_dim >= _WIDE_EMBEDDING_THRESHOLD or n_elements >= 4096

    if use_2d:
        total_blocks = triton.cdiv(n_elements, block_m) * triton.cdiv(embedding_dim, block_n)
        core_mult = 1 if 4096 <= n_elements < 8192 else (2 if n_elements >= 2048 else 1)
        grid = _launch_grid(num_cores, total_blocks, core_mult)
        embedding_backward_kernel_2d[(grid,)](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
        )
    else:
        total_blocks = triton.cdiv(n_elements, block_m)
        grid = _launch_grid(num_cores, total_blocks)
        embedding_backward_kernel[(grid,)](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
        )

    return grad_weight


class LigerEmbeddingFunction(torch.autograd.Function):
    @classmethod
    def apply(cls, embeddings, indices):
        if (
            cls is LigerEmbeddingFunction
            and cls.setup_context is torch.autograd.Function.setup_context
            and cls.forward is _PYTHON_EMBEDDING_FORWARD
            and cls.backward is _PYTHON_EMBEDDING_BACKWARD
            and _host_context_supported(embeddings, indices)
        ):
            host = _HOST_EXTENSION
            if host is None:
                host = _initialize_embedding_host()
            if host:
                return embedding_forward(embeddings, indices, _autograd=True)
        return super().apply(embeddings, indices)

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        output = embedding_forward(embeddings, indices)
        ctx.save_for_backward(indices, embeddings)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embeddings = ctx.saved_tensors
        grad_weight = embedding_backward(embeddings, indices, grad_output)
        return grad_weight, None


_PYTHON_EMBEDDING_FORWARD = LigerEmbeddingFunction.forward
_PYTHON_EMBEDDING_BACKWARD = LigerEmbeddingFunction.backward
