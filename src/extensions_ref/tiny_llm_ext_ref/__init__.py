# Copyright © 2023 Apple Inc.
#
# Modified by AmagiLabs for tiny-llm-cuda, 2026: optional CUDA backend. Set TINY_LLM_BACKEND=cuda to load the
# kernels from src/tiny_llm_cuda_kernels (mx.fast.cuda_kernel) instead of the compiled Metal extension.
# Without that variable this file behaves exactly as upstream. See NOTICE.

import os
import sys
from pathlib import Path

import mlx.core as mx

if os.environ.get("TINY_LLM_BACKEND", "").strip().lower() == "cuda":
    # The reference code also imports the private module directly (`from ... import _ext`), so register the
    # shim under that name too.
    from tiny_llm_cuda_kernels import ext_shim as _ext

    sys.modules[__name__ + "._ext"] = _ext
    from tiny_llm_cuda_kernels.ext_shim import *
else:
    try:
        from ._ext import *

        current_path = Path(__file__).parent
        load_library(str(current_path))
    except ImportError:
        print("Failed to load C++/Metal extension")
