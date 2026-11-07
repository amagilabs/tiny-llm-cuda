<!-- Copyright 2026 AmagiLabs. SPDX-License-Identifier: Apache-2.0. Part of tiny-llm-cuda; see NOTICE. -->
# CUDA backend: design notes

These notes cover the CUDA backend only. For the course itself, read the original book at
https://skyzh.github.io/tiny-llm/ .

## 1. How the backend plugs in

The reference solution calls its custom kernels through one module, `extensions_ref.tiny_llm_ext_ref`. On a
Mac that module loads a compiled Metal library. With `TINY_LLM_BACKEND=cuda`, its `__init__.py` imports
`tiny_llm_cuda_kernels.ext_shim` instead, which exposes the same 12 names:

`rms_norm`, `rope`, `_qk_norm_rope4`, `swiglu`, `_submit_decode_prefix_if_ordinary`, `quantized_matmul`,
`quantized_embedding`, `paged_cache_update`, `paged_attention`, `decode_attention`,
`_decode_attention_online`, `_dense_attention_prefill_mma`.

Every kernel is a CUDA source string compiled at runtime by MLX's own CUDA backend
(`mx.fast.cuda_kernel`, NVRTC). There is no C++ build step, no CMake and no nanobind. The rest of the course
code (the `mx.*` Python in `src/tiny_llm_ref`) runs unchanged on MLX's CUDA backend.

The learner's extension (`src/extensions/tiny_llm_ext`) is not switched. It stays the exercise.

## 2. Kernel map

| file | kernels | notes |
|---|---|---|
| `week2.py` | rms_norm, rope, swiglu, _qk_norm_rope4, async prefix submit | warp shuffles replace `simd_sum` / `simd_shuffle` |
| `quantized.py` | quantized_matmul (matvec paths), quantized_embedding | W4 packed weights, group scales and biases |
| `matrix.py` | quantized_matmul matrix (prefill) path | inline-PTX `mma.sync.m16n8k16`, f16/bf16 in, f32 accumulate, 64x64 tiles |
| `paged.py` | paged_cache_update, paged_attention | decode: one warp per query head; prefill: tiled |
| `attention.py` | decode_attention, _decode_attention_online, dense prefill | split-K decode (256 keys per partition, then a merge pass) |
| `flash.py` | tensor-core prefill for dense and paged attention, bf16, head dim 128 | FlashAttention-2 style; P stays in registers as the A operand of the second MMA |

Metal to CUDA was mostly mechanical. Apple SIMD-groups are 32 wide, like a warp, and `threadgroup` memory
maps to `__shared__`. The `simdgroup_matrix` 8x8 MMA tiles were the one place where the tiling had to change.

Things that were not obvious:
- `mx.fast.cuda_kernel` compile errors show up at `mx.eval`, not when the kernel is called.
- An input named `x` gets an `x_strides` array when `ensure_row_contiguous=False`, so strided K/V views can be
  read without a copy.
- Inline-PTX `mma.sync` needs no CUDA headers under NVRTC, so `mma.h` / `wmma` are not needed.
- A custom kernel cannot write in place into one of its inputs, so `paged_cache_update` returns a new page
  pool (a full copy plus the slice overwrite).

## 3. Per-call cost: cached launch plans

A decode step makes a few hundred custom-kernel calls, so Python cost per call matters as much as the
kernels. Each hot wrapper caches a plan per (shape, dtype, static parameters) key: the kernel object, the
parameter array and the launch arguments. Full input validation runs only on a cache miss, with the same
error messages; cheap per-call checks (paged slice bounds, finite scale) still run every call.

## 4. Week-3 call-site patch (`week3_patch.py`)

The upstream course files are not edited. When the CUDA backend loads, an import hook patches a few week-3
functions right after their modules load, to make fewer kernel calls per decode step:

| part | what it does | effect per decode step (Qwen3-4B, 36 layers) |
|---|---|---|
| `qk` | q/k RMSNorm + RoPE in one `_qk_norm_rope4` call each | -72 calls |
| `qk2` | q and k in one launch | -36 calls |
| `kv` | K and V paged-cache writes in one kernel | -36 calls |
| `blk` | residual add + post-attention RMSNorm in one kernel | fewer graph nodes |
| `fold` | each block's output add folded into the next layer's input norm | fewer graph nodes |
| `mlp` (opt-in) | gate/up projections + SwiGLU in one W4 matvec launch | fewer calls and nodes |

Custom-kernel calls per decode step go from 362 to 218 with the default set.

Safety rules:
- Each patch checks a hash of the upstream function source it replaces (written against upstream fa22518).
  If the upstream code has changed, that patch does not install and the upstream function runs.
- Every patched function falls back to the upstream function outside the decode fast path.
- The default set is bitwise: logits and greedy tokens are identical with the patch on and off
  (`scripts/cuda/check_fusion.py`). `mlp` is not bitwise against MLX's `quantized_matmul`, because the
  summation order differs (about 1% relative on logits), so it is opt-in only (`scripts/cuda/check_mlp.py`).
- `mlp` caches its validated launch plan per (module, input shape, dtype) and uses an ND kernel variant,
  so decode does no reshapes. A 512-step greedy agreement check against the default set
  (`scripts/cuda/check_agree.py`) shows divergence at near-tied tokens on 2 of 4 prompts (teacher-forced
  next-token match 505-512 of 512, decode-path perplexity within 0.2), so `mlp` stays opt-in.
  On an RTX 3090 (Qwen3-4B, 4-bit, sync loop) `all` decodes about 6% faster than `1`, and about 11% faster pipelined.

Switch: `TL_PATCH_WEEK3=1` (default, the bitwise set), `all` (adds `mlp`), `0` (off), or a comma list such as
`qk,kv`. `TL_FUSE_NORM_ROPE=0` turns off a smaller norm-into-RoPE fusion in `week2.py`.

## 5. Where decode time goes

On an RTX 3090 with Qwen3-4B 4-bit, our M=1 matvec is about as fast as MLX's own `quantized_matmul`. The
gap to stock mlx-lm is mostly CPU-side graph building: the per-call binding cost of `cuda_kernel` times the
number of calls, plus the course's own Python bookkeeping for the paged cache. That is why the work above
cut calls rather than tuning kernels. See the README for current numbers.
