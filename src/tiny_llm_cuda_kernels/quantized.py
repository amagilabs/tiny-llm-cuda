# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Week 2 Day 2 W4A16 (affine 4-bit, group 128) kernels, ported from quantized_matmul.metal.

quantized_matmul  use_simdgroup=False      -> quantized_matmul_vanilla_w4a16_g128 (1 thread / output)
                  use_simdgroup and M <= 8 -> quantized_matvec_x4_fast_w4a16_g128 (1 warp / 4 outputs)
                  use_simdgroup and M > 8  -> tiled MMA path (matrix.py, step 5)
quantized_embedding                        -> quantized_embedding_w4a16_g128

Layout: a [M, N] activations, b [K, N/8] uint32 (8 nibbles per word, low nibble first),
scales/biases [K, N/128]; out [M, K] = a @ dequant(b).T.
"""

import mlx.core as mx

from .common import call, kernel, launch, launch_kw

_VANILLA_SRC = r"""
    const int M = a_shape[0];
    const int N = a_shape[1];
    const int K = b_shape[0];
    const int groups_per_row = N / 128;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || k >= K) return;
    float sum = 0.0f;
    const unsigned int* bp = b + (size_t)k * (N / 8);
    const T* ap = a + (size_t)i * N;
    for (int g = 0; g < groups_per_row; ++g) {
        const float scale = static_cast<float>(scales[k * groups_per_row + g]);
        const float bias = static_cast<float>(biases[k * groups_per_row + g]);
        for (int item = 0; item < 128; item += 8) {
            const unsigned int w = *bp++;
            #pragma unroll
            for (int v = 0; v < 8; ++v)
                sum += (static_cast<float>((w >> (4 * v)) & 0xfu) * scale + bias) * static_cast<float>(ap[v]);
            ap += 8;
        }
    }
    out[(size_t)i * K + k] = static_cast<T>(sum);
"""

# 2 warps per block, 4 outputs per warp, each lane reads 2 packed words (16 activations) per step.
_MATVEC_SRC = r"""
    constexpr int packs_per_lane = 2;
    constexpr int outputs_per_warp = 4;
    constexpr int outputs_per_block = 8;
    const int M = a_shape[0];
    const int N = a_shape[1];
    const int K = b_shape[0];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column_tiles = (K + outputs_per_block - 1) / outputs_per_block;
    const int row = blockIdx.x / column_tiles;
    const int column_base = (blockIdx.x - row * column_tiles) * outputs_per_block + warp * outputs_per_warp;
    if (row >= M || column_base >= K) return;
    const int packed_cols = N / 8;
    const int groups_per_row = N / 128;
    const T* arow = a + (size_t)row * N;
    float sums[outputs_per_warp] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int packed_col = lane * packs_per_lane; packed_col < packed_cols; packed_col += 32 * packs_per_lane) {
        const int group = packed_col / 16;
        float sa[16];
        float asum = 0.0f;
        #pragma unroll
        for (int p = 0; p < packs_per_lane; ++p) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float x = static_cast<float>(arow[(packed_col + p) * 8 + v]);
                asum += x;
                // four nibbles share one uint16: pre-divide so masks replace shifts
                sa[p * 8 + v] = x / static_cast<float>(1 << ((v & 3) * 4));
            }
        }
        #pragma unroll
        for (int o = 0; o < outputs_per_warp; ++o) {
            const int column = column_base + o;
            if (column >= K) continue;
            const int pi = column * groups_per_row + group;
            const float scale = static_cast<float>(scales[pi]);
            const float bias = static_cast<float>(biases[pi]);
            const unsigned short* packed =
                reinterpret_cast<const unsigned short*>(b + (size_t)column * packed_cols + packed_col);
            float qd = 0.0f;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int w = packed[q];
                const int l = q * 4;
                qd += sa[l] * static_cast<float>(w & 0x000fu) + sa[l + 1] * static_cast<float>(w & 0x00f0u) +
                      sa[l + 2] * static_cast<float>(w & 0x0f00u) + sa[l + 3] * static_cast<float>(w & 0xf000u);
            }
            sums[o] += scale * qd + bias * asum;
        }
    }
    #pragma unroll
    for (int o = 0; o < outputs_per_warp; ++o) sums[o] = tl_warp_sum(sums[o]);
    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < outputs_per_warp; ++o) {
            const int column = column_base + o;
            if (column < K) out[(size_t)row * K + column] = static_cast<T>(sums[o]);
        }
    }
"""

# Fused MLP front half for decode: swiglu(a @ dequant(bg).T, a @ dequant(bu).T) in one launch.
# Per output it is _MATVEC_SRC verbatim (same lane split, same per-lane order, same warp sum), run once on the
# gate weights and once on the up weights with the activations loaded once; each sum is rounded to T, then
# week2's swiglu formula. So it is bitwise equal to quantized_matvec(gate), quantized_matvec(up), swiglu.
_GATE_UP_SRC = r"""
    constexpr int packs_per_lane = 2;
    constexpr int outputs_per_warp = 4;
    constexpr int outputs_per_block = 8;
    const int M = a_shape[0];
    const int N = a_shape[1];
    const int K = bg_shape[0];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column_tiles = (K + outputs_per_block - 1) / outputs_per_block;
    const int row = blockIdx.x / column_tiles;
    const int column_base = (blockIdx.x - row * column_tiles) * outputs_per_block + warp * outputs_per_warp;
    if (row >= M || column_base >= K) return;
    const int packed_cols = N / 8;
    const int groups_per_row = N / 128;
    const T* arow = a + (size_t)row * N;
    float gsums[outputs_per_warp] = {0.0f, 0.0f, 0.0f, 0.0f};
    float usums[outputs_per_warp] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int packed_col = lane * packs_per_lane; packed_col < packed_cols; packed_col += 32 * packs_per_lane) {
        const int group = packed_col / 16;
        float sa[16];
        float asum = 0.0f;
        #pragma unroll
        for (int p = 0; p < packs_per_lane; ++p) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float x = static_cast<float>(arow[(packed_col + p) * 8 + v]);
                asum += x;
                sa[p * 8 + v] = x / static_cast<float>(1 << ((v & 3) * 4));
            }
        }
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
            const unsigned int* b = which == 0 ? bg : bu;
            const T* scales = which == 0 ? sg : su;
            const T* biases = which == 0 ? zg : zu;
            float* sums = which == 0 ? gsums : usums;
            #pragma unroll
            for (int o = 0; o < outputs_per_warp; ++o) {
                const int column = column_base + o;
                if (column >= K) continue;
                const int pi = column * groups_per_row + group;
                const float scale = static_cast<float>(scales[pi]);
                const float bias = static_cast<float>(biases[pi]);
                const unsigned short* packed =
                    reinterpret_cast<const unsigned short*>(b + (size_t)column * packed_cols + packed_col);
                float qd = 0.0f;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const unsigned int w = packed[q];
                    const int l = q * 4;
                    qd += sa[l] * static_cast<float>(w & 0x000fu) + sa[l + 1] * static_cast<float>(w & 0x00f0u) +
                          sa[l + 2] * static_cast<float>(w & 0x0f00u) + sa[l + 3] * static_cast<float>(w & 0xf000u);
                }
                sums[o] += scale * qd + bias * asum;
            }
        }
    }
    #pragma unroll
    for (int o = 0; o < outputs_per_warp; ++o) {
        gsums[o] = tl_warp_sum(gsums[o]);
        usums[o] = tl_warp_sum(usums[o]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < outputs_per_warp; ++o) {
            const int column = column_base + o;
            if (column >= K) continue;
            const float g = static_cast<float>(static_cast<T>(gsums[o]));
            const float u = static_cast<float>(static_cast<T>(usums[o]));
            out[(size_t)row * K + column] = static_cast<T>((g / (1.0f + expf(-g))) * u);
        }
    }
"""

_EMBED_SRC = r"""
    const long tokens = indices_shape[0];
    const int dim = weight_shape[1] * 8;
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * dim) return;
    const long token = index / dim;
    const int column = index - token * dim;
    const long row = static_cast<long>(indices[token]);
    const int packed_cols = dim / 8;
    const int groups_per_row = dim / 128;
    const unsigned int packed = weight[row * packed_cols + column / 8];
    const int shift = (column % 8) * 4;
    const float q = static_cast<float>((packed >> shift) & 0xfu);
    const float scale = static_cast<float>(scales[row * groups_per_row + column / 128]);
    const float bias = static_cast<float>(biases[row * groups_per_row + column / 128]);
    out[index] = static_cast<T>(q * scale + bias);
"""


def _check_qmm(scales, biases, group_size, bits, a, b, transpose_b):
    if scales.dtype not in (mx.float16, mx.bfloat16):
        raise RuntimeError("quantized_matmul: scales must be float16 or bfloat16")
    if scales.dtype != biases.dtype:
        raise RuntimeError("quantized_matmul: scales and biases must be the same dtype")
    if b.dtype != mx.uint32:
        raise RuntimeError("quantized_matmul: b must be uint32")
    if a.dtype != scales.dtype:
        raise RuntimeError("quantized_matmul: a must be the same dtype as scales")
    if a.ndim != 2:
        raise RuntimeError("quantized_matmul: a must be a 2D array")
    if b.ndim != 2:
        raise RuntimeError("quantized_matmul: b must be a 2D array")
    if bits != 4:
        raise RuntimeError("quantized_matmul: bits must be 4")
    if group_size != 128:
        raise RuntimeError("quantized_matmul: group_size must be 128")
    if not transpose_b:
        raise RuntimeError("quantized_matmul: b must be transposed")
    if scales.shape != biases.shape:
        raise RuntimeError("quantized_matmul: scales and biases must have the same shape")
    if b.shape[0] != scales.shape[0]:
        raise RuntimeError("quantized_matmul: b must have the same number of rows as scales")
    if a.shape[1] % group_size != 0:
        raise RuntimeError("quantized_matmul: a columns must be divisible by group_size")
    if scales.shape[1] != a.shape[1] // group_size:
        raise RuntimeError("quantized_matmul: scales must have one column per input group")
    if b.shape[1] != a.shape[1] // 8:
        raise RuntimeError("quantized_matmul: a must have the same number of columns as b")


def quantized_matmul_vanilla(scales, biases, a, b, stream=None):
    M, _ = a.shape
    K = b.shape[0]
    x = 16 if M <= 16 else 32
    y = 256 // x
    return launch(
        kernel("tl_qmm_vanilla_w4a16_g128", ["scales", "biases", "a", "b"], ["out"], _VANILLA_SRC),
        [scales, biases, a, b], ((M + x - 1) // x, (K + y - 1) // y), (x, y),
        [(M, K)], [a.dtype], template=[("T", a.dtype)], stream=stream,
    )[0]


def quantized_matvec(scales, biases, a, b, stream=None):
    M, _ = a.shape
    K = b.shape[0]
    column_tiles = (K + 7) // 8
    return launch(
        kernel("tl_qmv_x4_fast_w4a16_g128", ["scales", "biases", "a", "b"], ["out"], _MATVEC_SRC),
        [scales, biases, a, b], (M * column_tiles,), (64,),
        [(M, K)], [a.dtype], template=[("T", a.dtype)], stream=stream,
    )[0]


# (shapes, dtypes, flags) of calls that passed _check_qmm -> ("zeros"|"tiled"|"vanilla", None) or ("call", (k, kw))
_QMM_PLANS = {}


def _qmm_plan(scales, biases, group_size, bits, a, b, transpose_b, use_simdgroup):
    _check_qmm(scales, biases, group_size, bits, a, b, transpose_b)
    M = a.shape[0]
    K = b.shape[0]
    if M == 0 or K == 0:
        return ("zeros", None)
    if use_simdgroup and M <= 8:
        column_tiles = (K + 7) // 8
        return ("call", (kernel("tl_qmv_x4_fast_w4a16_g128", ["scales", "biases", "a", "b"], ["out"], _MATVEC_SRC),
                         launch_kw((M * column_tiles,), (64,), [(M, K)], [a.dtype], template=[("T", a.dtype)])))
    if use_simdgroup:
        try:
            from .matrix import quantized_matmul_tiled  # noqa: F401
            return ("tiled", None)
        except ImportError:
            pass
    return ("vanilla", None)


def quantized_matmul(scales, biases, group_size, bits, a, b, transpose_b=False,
                     use_simdgroup=True, use_split_k=False, stream=None):
    key = (scales.shape, scales.dtype, biases.shape, biases.dtype, a.shape, a.dtype, b.shape, b.dtype,
           group_size, bits, bool(transpose_b), bool(use_simdgroup))
    plan = _QMM_PLANS.get(key)
    if plan is None:
        plan = _QMM_PLANS[key] = _qmm_plan(scales, biases, group_size, bits, a, b, transpose_b, use_simdgroup)
    route, data = plan
    if route == "call":
        return call(data[0], [scales, biases, a, b], data[1], stream)[0]
    if route == "tiled":
        from .matrix import quantized_matmul_tiled
        return quantized_matmul_tiled(scales, biases, a, b, use_split_k, stream=stream)
    if route == "zeros":
        return mx.zeros((a.shape[0], b.shape[0]), dtype=a.dtype)
    return quantized_matmul_vanilla(scales, biases, a, b, stream=stream)


_EMBED_PLANS = {}


def _embed_plan(indices, scales, biases, weight, group_size, bits):
    if indices.dtype not in (mx.int32, mx.uint32) or weight.dtype != mx.uint32:
        raise RuntimeError("quantized_embedding: indices and weight must use 32-bit integers")
    if scales.dtype != biases.dtype or scales.dtype not in (mx.float16, mx.bfloat16):
        raise RuntimeError("quantized_embedding: scales and biases must have the same 16-bit dtype")
    if group_size != 128 or bits != 4 or scales.shape != biases.shape:
        raise RuntimeError("quantized_embedding: expected 4-bit weights with group size 128")
    dim = weight.shape[1] * 8
    if scales.shape[0] != weight.shape[0] or scales.shape[1] != dim // group_size:
        raise RuntimeError("quantized_embedding: incompatible parameter shapes")
    out_shape = tuple(indices.shape) + (dim,)
    tokens = indices.size
    total = tokens * dim
    if total == 0:
        return (None, None, out_shape, tokens)
    threads = 256
    kw = launch_kw(((total + threads - 1) // threads,), (threads,), [(total,)], [scales.dtype],
                   template=[("T", scales.dtype)])
    return (kernel("tl_qembed_w4a16_g128", ["indices", "scales", "biases", "weight"], ["out"], _EMBED_SRC), kw,
            out_shape, tokens)


def quantized_embedding(indices, scales, biases, weight, group_size, bits, stream=None):
    key = (indices.shape, indices.dtype, scales.shape, scales.dtype, biases.shape, biases.dtype,
           weight.shape, weight.dtype, group_size, bits)
    plan = _EMBED_PLANS.get(key)
    if plan is None:
        plan = _EMBED_PLANS[key] = _embed_plan(indices, scales, biases, weight, group_size, bits)
    k, kw, out_shape, tokens = plan
    if k is None:
        return mx.zeros(out_shape, dtype=scales.dtype)
    return call(k, [indices.reshape(tokens), scales, biases, weight], kw, stream)[0].reshape(out_shape)


__all__ = ["quantized_matmul", "quantized_embedding"]


# ---------------------------------------------------------------- fused gate/up + swiglu (week-3 call-site patch)

_GATE_UP_PLANS = {}


def quantized_gate_up_swiglu(a, wg, sg, zg, wu, su, zu, stream=None):
    """swiglu(a @ dequant(wg).T, a @ dequant(wu).T) for a [M, N], M <= 8, W4A16 g128, in one launch.

    Bitwise equal to swiglu(quantized_matvec(gate), quantized_matvec(up)) with week2's swiglu. Returns None
    when the shapes/dtypes are outside that path, so the caller keeps its own code."""
    key = (a.shape, a.dtype, wg.shape, wg.dtype, sg.shape, sg.dtype, zg.shape, zg.dtype,
           wu.shape, wu.dtype, su.shape, su.dtype, zu.shape, zu.dtype)
    plan = _GATE_UP_PLANS.get(key)
    if plan is None:
        ok = (a.ndim == 2 and wg.ndim == 2 and 0 < a.shape[0] <= 8 and a.dtype in (mx.bfloat16, mx.float16)
              and wg.dtype == mx.uint32 and wu.dtype == mx.uint32 and wg.shape == wu.shape
              and a.shape[1] % 128 == 0 and wg.shape[1] == a.shape[1] // 8 and wg.shape[0] > 0
              and sg.shape == zg.shape == su.shape == zu.shape == (wg.shape[0], a.shape[1] // 128)
              and all(t.dtype == a.dtype for t in (sg, zg, su, zu)) and mx.default_device() == mx.gpu)
        if ok:
            M, K = a.shape[0], wg.shape[0]
            k = kernel("tl_qmv_gate_up_swiglu_w4a16_g128", ["a", "bg", "sg", "zg", "bu", "su", "zu"], ["out"],
                       _GATE_UP_SRC)
            plan = (k, launch_kw((M * ((K + 7) // 8),), (64,), [(M, K)], [a.dtype], template=[("T", a.dtype)]))
        else:
            plan = False
        _GATE_UP_PLANS[key] = plan
    if plan is False:
        return None
    return call(plan[0], [a, wg, sg, zg, wu, su, zu], plan[1], stream)[0]


# Same kernel body, but M and N come from the grid and the packed weight instead of a_shape, so the
# activations can be any [..., N] (decode's [1, 1, N]) and the output any [..., K]: no reshape nodes.
_GATE_UP_HEAD = """    const int M = a_shape[0];
    const int N = a_shape[1];
    const int K = bg_shape[0];
"""
assert _GATE_UP_SRC.count(_GATE_UP_HEAD) == 1
_GATE_UP_ND_SRC = _GATE_UP_SRC.replace(_GATE_UP_HEAD, """    const int K = bg_shape[0];
    const int N = bg_shape[1] * 8;
    const int M = gridDim.x / ((K + 7) / 8);
""")


def gate_up_swiglu_plan(x, wg, sg, zg, wu, su, zu):
    """Validated launch plan for x [..., N] with rows = prod(x.shape[:-1]) in 1..8: (kernel, launch kw), output
    shaped x.shape[:-1] + (K,). None outside the path. Callers cache it (week3_patch keys it on the weight
    arrays and x.shape/dtype), so the decode step does no shape checks and no reshapes.
    Bitwise equal to quantized_gate_up_swiglu(x.reshape(rows, N), ...).reshape(..., K)."""
    shape = tuple(x.shape)
    if not shape or mx.default_device() != mx.gpu:
        return None
    N = shape[-1]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    ok = (0 < rows <= 8 and x.dtype in (mx.bfloat16, mx.float16) and wg.ndim == 2
          and wg.dtype == mx.uint32 and wu.dtype == mx.uint32 and wg.shape == wu.shape
          and N % 128 == 0 and wg.shape[1] == N // 8 and wg.shape[0] > 0
          and sg.shape == zg.shape == su.shape == zu.shape == (wg.shape[0], N // 128)
          and all(t.dtype == x.dtype for t in (sg, zg, su, zu)))
    if not ok:
        return None
    K = wg.shape[0]
    k = kernel("tl_qmv_gate_up_swiglu_w4a16_g128_nd", ["a", "bg", "sg", "zg", "bu", "su", "zu"], ["out"],
               _GATE_UP_ND_SRC)
    return k, launch_kw((rows * ((K + 7) // 8),), (64,), [shape[:-1] + (K,)], [x.dtype],
                        template=[("T", x.dtype)])
