# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""BF16 / D=128 FlashAttention-2 style prefill on tensor cores (L > 8).

dense_prefill_wmma            -> _dense_attention_prefill_mma (week 2 day 5) fast path
paged_attention_mma_bf16_d128 -> paged_attention MMA path (week 3 day 4/5)

Block = 64 query rows of one (batch, head), 4 warps x 16 rows. Per 32-key tile each warp computes
S = Q K^T (mma.sync m16n8k16, f32 acc), applies scale*log2(e) (+ mask*log2(e)) and the causal
limit, does the exp2 online softmax per row with quad shuffles, rounds P to BF16 and accumulates
O += P V on tensor cores again (P stays in registers as the A operand, FA2 style).
"""

import mlx.core as mx

from .common import kernel, launch, f32

_HEADER = r"""
#define TL_NEG_INF (-__int_as_float(0x7f800000))
__device__ __forceinline__ unsigned int tl_ld32(const unsigned short* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 16);
}
__device__ __forceinline__ void tl_mma_bf16(float* c, unsigned int a0, unsigned int a1, unsigned int a2,
                                            unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
"""

_BODY = r"""
    constexpr int D = 128;
    constexpr int BQ = 64;
    constexpr int BK = 32;
    constexpr int LDK = D + 8;
    constexpr int LDV = BK + 8;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ __align__(16) unsigned short Ks[BK * LDK];
    __shared__ __align__(16) unsigned short Vt[D * LDV];
    __shared__ int physical_pages[BK];
    auto pack = [](float lo, float hi) -> unsigned int {
        T x = static_cast<T>(lo);
        T y = static_cast<T>(hi);
        return (unsigned int)(*reinterpret_cast<unsigned short*>(&x)) |
               ((unsigned int)(*reinterpret_cast<unsigned short*>(&y)) << 16);
    };
    const int q_rows = q_shape[0];
    const int L = q_shape[1];
    const int n = blockIdx.y;
    const int query_block = blockIdx.x;
    if (n >= q_rows) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int batch = n / NUM_HEADS;
    const int query_head = n - batch * NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const float scale_log2 = params[0] * LOG2_E;
@SETUP@
    const int warp_row = query_block * BQ + warp * 16;
    const int qpos[2] = {warp_row + g, warp_row + g + 8};
    const unsigned short* q16 = reinterpret_cast<const unsigned short*>(q) + (long)n * L * D;
    unsigned int qf[8][4];
    #pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
        const int col = kc * 16 + t * 2;
        qf[kc][0] = qpos[0] < L ? tl_ld32(q16 + (long)qpos[0] * D + col) : 0u;
        qf[kc][1] = qpos[1] < L ? tl_ld32(q16 + (long)qpos[1] * D + col) : 0u;
        qf[kc][2] = qpos[0] < L ? tl_ld32(q16 + (long)qpos[0] * D + col + 8) : 0u;
        qf[kc][3] = qpos[1] < L ? tl_ld32(q16 + (long)qpos[1] * D + col + 8) : 0u;
    }
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] = 0.0f; o[i][1] = 0.0f; o[i][2] = 0.0f; o[i][3] = 0.0f; }
    float row_max[2] = {TL_NEG_INF, TL_NEG_INF};
    float row_sum[2] = {0.0f, 0.0f};
    const int total_tiles = (context + BK - 1) / BK;
    int tile_limit = total_tiles;
    if (IS_CAUSAL) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_key = last_query + (context - L);
        tile_limit = min(max((last_key + 1 + BK - 1) / BK, 0), total_tiles);
    }
    for (int tile = 0; tile < tile_limit; ++tile) {
        const int tile_start = tile * BK;
@TILE_SETUP@
        for (int idx = tid; idx < BK * (D / 2); idx += 128) {
            const int key = idx / (D / 2);
            const int d = (idx - key * (D / 2)) * 2;
            const unsigned short* src = @KPTR@;
            *reinterpret_cast<unsigned int*>(&Ks[key * LDK + d]) = src != nullptr ? tl_ld32(src + d) : 0u;
        }
        for (int idx = tid; idx < BK * D; idx += 128) {
            const int key = idx / D;
            const int d = idx - key * D;
            const unsigned short* src = @VPTR@;
            Vt[d * LDV + key] = src != nullptr ? src[d] : (unsigned short)0;
        }
        __syncthreads();
        float s[4][4];
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) { s[ni][0] = 0.0f; s[ni][1] = 0.0f; s[ni][2] = 0.0f; s[ni][3] = 0.0f; }
        #pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const unsigned short* kp = &Ks[(ni * 8 + g) * LDK + kc * 16 + t * 2];
                tl_mma_bf16(s[ni], qf[kc][0], qf[kc][1], qf[kc][2], qf[kc][3],
                            *reinterpret_cast<const unsigned int*>(kp), *reinterpret_cast<const unsigned int*>(kp + 8));
            }
        }
        float tile_max[2] = {TL_NEG_INF, TL_NEG_INF};
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const int row = qpos[h];
                const int key_local = ni * 8 + t * 2 + (e & 1);
                const int key = tile_start + key_local;
                bool valid = row < L && key < context && @KEY_OK@;
                if (IS_CAUSAL) valid = valid && key <= row + (context - L);
                float v = TL_NEG_INF;
                if (valid) {
                    v = s[ni][e] * scale_log2;
@MASK@
                }
                s[ni][e] = v;
                tile_max[h] = fmaxf(tile_max[h], v);
            }
        }
        float alpha[2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 1));
            tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 2));
            const float new_max = fmaxf(row_max[h], tile_max[h]);
            const bool finite_row = new_max != TL_NEG_INF;
            alpha[h] = row_max[h] == TL_NEG_INF || !finite_row ? 0.0f : exp2f(row_max[h] - new_max);
            row_max[h] = new_max;
        }
        float tile_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float p = s[ni][e] == TL_NEG_INF || row_max[h] == TL_NEG_INF ? 0.0f : exp2f(s[ni][e] - row_max[h]);
                s[ni][e] = p;
                tile_sum[h] += p;
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            tile_sum[h] += __shfl_xor_sync(0xffffffffu, tile_sum[h], 1);
            tile_sum[h] += __shfl_xor_sync(0xffffffffu, tile_sum[h], 2);
            row_sum[h] = alpha[h] * row_sum[h] + tile_sum[h];
        }
        #pragma unroll
        for (int ni = 0; ni < 16; ++ni) {
            o[ni][0] *= alpha[0]; o[ni][1] *= alpha[0];
            o[ni][2] *= alpha[1]; o[ni][3] *= alpha[1];
        }
        #pragma unroll
        for (int kc = 0; kc < 2; ++kc) {
            const unsigned int a0 = pack(s[2 * kc][0], s[2 * kc][1]);
            const unsigned int a1 = pack(s[2 * kc][2], s[2 * kc][3]);
            const unsigned int a2 = pack(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            const unsigned int a3 = pack(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            #pragma unroll
            for (int ni = 0; ni < 16; ++ni) {
                const unsigned short* vp = &Vt[(ni * 8 + g) * LDV + kc * 16 + t * 2];
                tl_mma_bf16(o[ni], a0, a1, a2, a3,
                            *reinterpret_cast<const unsigned int*>(vp), *reinterpret_cast<const unsigned int*>(vp + 8));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = qpos[h];
        if (row >= L) continue;
        const float inv = row_sum[h] == 0.0f ? 0.0f : 1.0f / row_sum[h];
        T* dst = out + ((long)n * L + row) * D;
        #pragma unroll
        for (int ni = 0; ni < 16; ++ni) {
            const int d = ni * 8 + t * 2;
            dst[d] = static_cast<T>(o[ni][h * 2] * inv);
            dst[d + 1] = static_cast<T>(o[ni][h * 2 + 1] * inv);
        }
    }
"""

_DENSE_SRC = (_BODY
    .replace("@SETUP@", r"""
    const int context = k_shape[1];
    const long kv_row = (long)batch * NUM_KV_HEADS + kv_head;
    const unsigned short* kbase = reinterpret_cast<const unsigned short*>(k) + kv_row * context * D;
    const unsigned short* vbase = reinterpret_cast<const unsigned short*>(v) + kv_row * context * D;
    (void)physical_pages;
""")
    .replace("@TILE_SETUP@", "")
    .replace("@KPTR@", "tile_start + key < context ? kbase + (long)(tile_start + key) * D : nullptr")
    .replace("@VPTR@", "tile_start + key < context ? vbase + (long)(tile_start + key) * D : nullptr")
    .replace("@KEY_OK@", "true")
    .replace("@MASK@", r"""
                    if (HAS_MASK) v += mask[((long)n * L + row) * context + key] * LOG2_E;
"""))

_PAGED_SRC = (_BODY
    .replace("@SETUP@", r"""
    const int context = context_lens[batch];
    const int page_size = key_pages_shape[2];
    const int max_pages = block_table_shape[1];
    const bool single_page_tile = page_size >= BK && page_size % BK == 0;
    const unsigned short* kpages = reinterpret_cast<const unsigned short*>(key_pages);
    const unsigned short* vpages = reinterpret_cast<const unsigned short*>(value_pages);
""")
    .replace("@TILE_SETUP@", r"""
        const int tile_logical_page = tile_start / page_size;
        const int tile_page_slot = tile_start - tile_logical_page * page_size;
        if (single_page_tile) {
            if (tid == 0)
                physical_pages[0] = tile_start < context && tile_logical_page < max_pages
                    ? block_table[batch * max_pages + tile_logical_page] : -1;
        } else if (tid < BK) {
            const int key = tile_start + tid;
            const int logical_page = key / page_size;
            physical_pages[tid] = key < context && logical_page < max_pages
                ? block_table[batch * max_pages + logical_page] : -1;
        }
        __syncthreads();
        auto row_ptr = [&](const unsigned short* pages, int key) -> const unsigned short* {
            const int global_key = tile_start + key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile ? tile_page_slot + key : global_key - logical_page * page_size;
            const int pid = single_page_tile ? physical_pages[0] : physical_pages[key];
            return pid >= 0 ? pages + (((long)pid * NUM_KV_HEADS + kv_head) * page_size + slot) * D : nullptr;
        };
""")
    .replace("@KPTR@", "row_ptr(kpages, key)")
    .replace("@VPTR@", "row_ptr(vpages, key)")
    .replace("@KEY_OK@", "(single_page_tile ? physical_pages[0] : physical_pages[key_local]) >= 0")
    .replace("@MASK@", ""))

_DENSE_IN = ["q", "k", "v", "mask", "params"]
_PAGED_IN = ["q", "key_pages", "value_pages", "block_table", "context_lens", "params"]


def dense_prefill_wmma(query, key, value, mask, scale, is_causal, has_mask, num_heads, num_kv_heads, stream=None):
    q_rows, length, _ = query.shape
    return launch(
        kernel("tl_dense_prefill_mma_bf16_d128", _DENSE_IN, ["out"], _DENSE_SRC, header=_HEADER),
        [query, key, value, mask, f32(scale)], ((length + 63) // 64, q_rows), (128,), [query.shape],
        [query.dtype],
        template=[("T", query.dtype), ("NUM_HEADS", int(num_heads)), ("NUM_KV_HEADS", int(num_kv_heads)),
                  ("IS_CAUSAL", int(bool(is_causal))), ("HAS_MASK", int(bool(has_mask)))],
        stream=stream,
    )[0]


def paged_attention_mma_bf16_d128(query, key_pages, value_pages, block_table, context_lens, scale, is_causal,
                                  num_kv_heads, num_heads, stream=None):
    q_rows, length, _ = query.shape
    return launch(
        kernel("tl_paged_attention_mma_bf16_d128", _PAGED_IN, ["out"], _PAGED_SRC, header=_HEADER),
        [query, key_pages, value_pages, block_table, context_lens, f32(scale)],
        ((length + 63) // 64, q_rows), (128,), [query.shape], [query.dtype],
        template=[("T", query.dtype), ("NUM_HEADS", int(num_heads)), ("NUM_KV_HEADS", int(num_kv_heads)),
                  ("IS_CAUSAL", int(bool(is_causal)))],
        stream=stream,
    )[0]


__all__ = ["dense_prefill_wmma", "paged_attention_mma_bf16_d128"]
