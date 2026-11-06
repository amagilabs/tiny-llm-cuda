# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""tiny-llm weeks 2-3 kernels ported from Metal to CUDA via mx.fast.cuda_kernel.

Loaded when TINY_LLM_BACKEND=cuda (src/extensions_ref/tiny_llm_ext_ref/__init__.py);
tiny_llm_cuda_kernels/ext_shim.py re-exports whatever is implemented here.
"""

from .week2 import rms_norm, rope, _qk_norm_rope4, swiglu, _submit_decode_prefix_if_ordinary
from .quantized import quantized_matmul, quantized_embedding
from .paged import paged_cache_update, paged_attention
from .attention import decode_attention, _decode_attention_online, _dense_attention_prefill_mma

IMPLEMENTED = {
    "decode_attention": decode_attention,
    "_decode_attention_online": _decode_attention_online,
    "_dense_attention_prefill_mma": _dense_attention_prefill_mma,
    "paged_attention": paged_attention,
    "paged_cache_update": paged_cache_update,
    "quantized_matmul": quantized_matmul,
    "quantized_embedding": quantized_embedding,
    "rms_norm": rms_norm,
    "rope": rope,
    "_qk_norm_rope4": _qk_norm_rope4,
    "swiglu": swiglu,
    "_submit_decode_prefix_if_ordinary": _submit_decode_prefix_if_ordinary,
}

# Week-3 call-site patch (fewer kernel calls per decode step). Installs an import hook that patches
# tiny_llm_ref.qwen3_week3 / paged_kv_cache after they load; upstream files stay untouched.
from . import week3_patch as _week3_patch

_week3_patch.install()
