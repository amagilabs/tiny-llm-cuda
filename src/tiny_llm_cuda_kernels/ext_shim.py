# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""CUDA stand-in for the compiled Metal extension (extensions_ref.tiny_llm_ext_ref._ext).

Imported by src/extensions_ref/tiny_llm_ext_ref/__init__.py when TINY_LLM_BACKEND=cuda. It exposes the same
names the Metal extension does, implemented with mx.fast.cuda_kernel in this package. A name with no CUDA
implementation raises NotImplementedError naming the Metal kernel, so a gap fails loudly instead of silently.
"""

from . import IMPLEMENTED as _IMPLEMENTED

_KERNELS = [
    "_submit_decode_prefix_if_ordinary", "quantized_matmul", "quantized_embedding",
    "rms_norm", "rope", "_qk_norm_rope4", "swiglu", "decode_attention",
    "_decode_attention_online", "_dense_attention_prefill_mma",
    "paged_cache_update", "paged_attention",
]


def load_library(path):  # the Metal backend loads a .metallib here; nothing to load on CUDA
    return None


def _make(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(f"tiny-llm Metal kernel '{name}' has no CUDA implementation in tiny_llm_cuda_kernels")
    _stub.__name__ = name
    return _stub


for _n in _KERNELS:
    globals()[_n] = _IMPLEMENTED.get(_n) or _make(_n)

__all__ = ["load_library"] + _KERNELS
