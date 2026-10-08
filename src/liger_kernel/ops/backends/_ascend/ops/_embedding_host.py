"""Optional native host dispatch for Ascend embedding; device math stays in Triton.

The extension uses PyTorch's normal compiler cache. A C++ compiler and ninja are
required. Unsupported toolchains retain the Python/Triton implementation.
Set LIGER_ASCEND_EMBEDDING_HOST=0 before import to select that reference path.
"""

import hashlib
import os
import re
import warnings

from pathlib import Path


def load_host_extension():
    if os.environ.get("LIGER_ASCEND_EMBEDDING_HOST", "1").lower() in ("0", "false", "off"):
        return None
    try:
        import torch

        from torch.utils.cpp_extension import load

        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        version = re.sub(r"[^a-zA-Z0-9_]", "_", torch.__version__)
        return load(
            name=f"liger_ascend_embedding_host_{version}_{digest}",
            sources=[str(source)],
            extra_cflags=["-O2", "-g0"],
            with_cuda=False,
            verbose=False,
        )
    except (ImportError, OSError, RuntimeError) as error:
        warnings.warn(
            f"Ascend embedding host extension unavailable; using Python/Triton launch: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
