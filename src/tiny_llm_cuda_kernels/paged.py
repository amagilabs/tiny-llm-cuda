# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Week 3 paged KV-cache kernels, ported from paged_attention.metal.

paged_cache_update: the Metal primitive aliases the page buffer and writes only the
appended slice. mx.fast.cuda_kernel cannot alias an input into an output, so the CUDA
port is a single pass that copies the page tensor and overwrites the slice
(pages [P, H, page_size, D], values [1, H, length, D]).
"""

import mlx.core as mx

from .common import call, kernel, launch, launch_kw, i32, f32

_UPDATE_SRC = r"""
    const int heads = pages_shape[1];
    const int page_size = pages_shape[2];
    const int head_dim = pages_shape[3];
    const int length = values_shape[2];
    const int page_id = params[0];
    const int start = params[1];
    const long total = (long)pages_shape[0] * heads * page_size * head_dim;
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int dim = index % head_dim;
    long rest = index / head_dim;
    const int slot = rest % page_size;
    rest /= page_size;
    const int head = rest % heads;
    const int page = rest / heads;
    const int token = slot - start;
    if (page == page_id && token >= 0 && token < length) {
        out[index] = values[((long)head * length + token) * head_dim + dim];
    } else {
        out[index] = pages[index];
    }
"""


# (pages.shape, pages.dtype, values.shape, values.dtype) -> (kernel, launch kwargs, page_size, length) or None
_UPDATE_PLANS = {}


def _update_plan(pages, values):
    if pages.dtype not in (mx.float32, mx.bfloat16) or values.dtype != pages.dtype:
        raise RuntimeError("paged_cache_update: pages and values must have the same float32 or bfloat16 dtype")
    if pages.ndim != 4 or values.ndim != 4 or values.shape[0] != 1:
        raise RuntimeError("paged_cache_update: expected pages [P, H, page_size, D] and values [1, H, length, D]")
    if values.shape[1] != pages.shape[1] or values.shape[3] != pages.shape[3]:
        raise RuntimeError("paged_cache_update: values must match the page head count and head dimension")
    total = pages.size
    empty = total == 0 or values.size == 0
    threads = 256
    kw = launch_kw(((total + threads - 1) // threads,), (threads,), [pages.shape], [pages.dtype])
    return (kernel("tl_paged_cache_update", ["pages", "values", "params"], ["out"], _UPDATE_SRC), kw,
            pages.shape[0], pages.shape[2], values.shape[2], empty)


def paged_cache_update(pages, values, page_id, start, stream=None):
    key = (pages.shape, pages.dtype, values.shape, values.dtype)
    plan = _UPDATE_PLANS.get(key)
    if plan is None:
        plan = _UPDATE_PLANS[key] = _update_plan(pages, values)
    k, kw, num_pages, page_size, length, empty = plan
    if page_id < 0 or page_id >= num_pages or start < 0 or start + length > page_size:
        raise RuntimeError("paged_cache_update: destination slice is outside page storage")
    if empty:
        return pages
    return call(k, [pages, values, i32(page_id, start)], kw, stream)[0]


# Fused K+V update: one launch writes both pools (same index math as _UPDATE_SRC, two loads/stores per
# thread). Called by the week-3 call-site patch (week3_patch.py) from TinyKvPagedPool.write_page_slice,
# which upstream does as two paged_cache_update calls. Pure copy, so bit-identical to the pair.
_UPDATE_KV_SRC = r"""
    const int heads = key_pages_shape[1];
    const int page_size = key_pages_shape[2];
    const int head_dim = key_pages_shape[3];
    const int length = keys_shape[2];
    const int page_id = params[0];
    const int start = params[1];
    const long total = (long)key_pages_shape[0] * heads * page_size * head_dim;
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int dim = index % head_dim;
    long rest = index / head_dim;
    const int slot = rest % page_size;
    rest /= page_size;
    const int head = rest % heads;
    const int page = rest / heads;
    const int token = slot - start;
    if (page == page_id && token >= 0 && token < length) {
        const long src = ((long)head * length + token) * head_dim + dim;
        out_k[index] = keys[src];
        out_v[index] = values[src];
    } else {
        out_k[index] = key_pages[index];
        out_v[index] = value_pages[index];
    }
"""

_UPDATE_KV_PLANS = {}


def _update_kv_plan(key_pages, value_pages, keys, values):
    _update_plan(key_pages, keys)  # same validation as the single-pool kernel
    if value_pages.shape != key_pages.shape or value_pages.dtype != key_pages.dtype:
        raise RuntimeError("paged_cache_update_kv: key and value pools must have the same shape and dtype")
    if values.shape != keys.shape or values.dtype != keys.dtype:
        raise RuntimeError("paged_cache_update_kv: keys and values must have the same shape and dtype")
    total = key_pages.size
    empty = total == 0 or keys.size == 0
    threads = 256
    kw = launch_kw(((total + threads - 1) // threads,), (threads,), [key_pages.shape, value_pages.shape],
                   [key_pages.dtype, value_pages.dtype])
    k = kernel("tl_paged_cache_update_kv", ["key_pages", "value_pages", "keys", "values", "params"],
               ["out_k", "out_v"], _UPDATE_KV_SRC)
    return (k, kw, key_pages.shape[0], key_pages.shape[2], keys.shape[2], empty)


def paged_cache_update_kv(key_pages, value_pages, keys, values, page_id, start, stream=None):
    """Write keys/values [1, H, length, D] into slot `start` of page `page_id` of both pools in one kernel.

    Returns (new_key_pages, new_value_pages), each equal to paged_cache_update on its own pool."""
    key = (key_pages.shape, key_pages.dtype, value_pages.shape, value_pages.dtype, keys.shape, keys.dtype,
           values.shape, values.dtype)
    plan = _UPDATE_KV_PLANS.get(key)
    if plan is None:
        plan = _UPDATE_KV_PLANS[key] = _update_kv_plan(key_pages, value_pages, keys, values)
    k, kw, num_pages, page_size, length, empty = plan
    if page_id < 0 or page_id >= num_pages or start < 0 or start + length > page_size:
        raise RuntimeError("paged_cache_update_kv: destination slice is outside page storage")
    if empty:
        return key_pages, value_pages
    out = call(k, [key_pages, value_pages, keys, values, i32(page_id, start)], kw, stream)
    return out[0], out[1]


_ATTN_HEADER = r"""
#define TL_NEG_INF (-__int_as_float(0x7f800000))
"""

# One block (32 warps) per query row. Warps stride over the visible keys with an
# online softmax; partials are merged through a 32x32 transpose in shared memory.
_DECODE_SRC = r"""
    constexpr int MAX_VALUES_PER_LANE = 4;
    constexpr int WARPS = 32;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ float partial_outputs[WARPS * 32];
    __shared__ float partial_maxima[WARPS];
    __shared__ float partial_sums[WARPS];
    const int N = q_shape[0];
    const int L = q_shape[1];
    const int D = q_shape[2];
    const int page_size = key_pages_shape[2];
    const int max_pages = block_table_shape[1];
    const int query_index = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (query_index >= N * L) return;
    const float scale_log2 = params[0] * LOG2_E;
    const int dimension = FIXED_D == 0 ? D : FIXED_D;
    const int n = query_index / L;
    const int query_position = query_index - n * L;
    const int batch = n / NUM_HEADS;
    const int query_head = n - batch * NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const int context = context_lens[batch];
    const int values_per_lane = (dimension + 31) / 32;

    float accumulator[MAX_VALUES_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float query_values[MAX_VALUES_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float max_score = -1e30f;
    float sum = 0.0f;
    #pragma unroll
    for (int item = 0; item < MAX_VALUES_PER_LANE; item++) {
        const int dim = lane * values_per_lane + item;
        if (item < values_per_lane && dim < dimension)
            query_values[item] = static_cast<float>(q[(long)query_index * D + dim]) * scale_log2;
    }
    int visible_context = context;
    if (IS_CAUSAL) visible_context = min(max(context - L + query_position + 1, 0), context);
    const int visible_pages = min(max_pages, (visible_context + page_size - 1) / page_size);
    for (int logical_page = 0; logical_page < visible_pages; logical_page++) {
        const int page_id = block_table[batch * max_pages + logical_page];
        if (page_id < 0) continue;
        const int page_start = logical_page * page_size;
        const int live_slots = min(page_size, visible_context - page_start);
        for (int slot = warp; slot < live_slots; slot += WARPS) {
            const long page_offset = (((long)page_id * NUM_KV_HEADS + kv_head) * page_size + slot) * D;
            float partial = 0.0f;
            #pragma unroll
            for (int item = 0; item < MAX_VALUES_PER_LANE; item++) {
                const int dim = lane * values_per_lane + item;
                if (item < values_per_lane && dim < dimension)
                    partial += query_values[item] * static_cast<float>(key_pages[page_offset + dim]);
            }
            const float score = tl_warp_sum(partial);
            const float new_max = fmaxf(max_score, score);
            const float old_factor = exp2f(max_score - new_max);
            const float score_factor = exp2f(score - new_max);
            sum = sum * old_factor + score_factor;
            #pragma unroll
            for (int item = 0; item < MAX_VALUES_PER_LANE; item++) {
                const int dim = lane * values_per_lane + item;
                if (item < values_per_lane && dim < dimension)
                    accumulator[item] = accumulator[item] * old_factor +
                        score_factor * static_cast<float>(value_pages[page_offset + dim]);
            }
            max_score = new_max;
        }
    }
    if (lane == 0) {
        partial_maxima[warp] = max_score;
        partial_sums[warp] = sum;
    }
    __syncthreads();
    const float partial_max = partial_maxima[lane];
    const float global_max = tl_warp_max(partial_max);
    const float factor = exp2f(partial_max - global_max);
    const float global_sum = tl_warp_sum(partial_sums[lane] * factor);
    #pragma unroll
    for (int item = 0; item < MAX_VALUES_PER_LANE; item++) {
        if (item >= values_per_lane) break;
        partial_outputs[lane * WARPS + warp] = accumulator[item];
        __syncthreads();
        accumulator[item] = tl_warp_sum(partial_outputs[warp * WARPS + lane] * factor);
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (int item = 0; item < MAX_VALUES_PER_LANE; item++) {
            const int dim = warp * values_per_lane + item;
            if (item < values_per_lane && dim < dimension)
                out[(long)query_index * D + dim] =
                    global_sum == 0.0f ? static_cast<T>(0.0f) : static_cast<T>(accumulator[item] / global_sum);
        }
    }
"""

# Block = 16 query rows x 32 keys (one warp per query row); K/V tiles staged in shared memory.
_SCALAR_SRC = r"""
    constexpr int BQ = 16;
    constexpr int BK = 32;
    constexpr int MAX_D = 128;
    constexpr int OUTPUTS_PER_LANE = MAX_D / BK;
    constexpr int THREADS = BQ * BK;
    __shared__ float q_tile[BQ * MAX_D];
    __shared__ float kv_tile[BK * MAX_D];
    __shared__ float probabilities[BQ * BK];
    __shared__ int physical_pages[BK];
    const int N = q_shape[0];
    const int L = q_shape[1];
    const int D = q_shape[2];
    const int page_size = key_pages_shape[2];
    const int max_pages = block_table_shape[1];
    const float scale = params[0];
    const int n = blockIdx.x;
    const int query_block = blockIdx.y;
    const int query_row_in_block = threadIdx.y;
    const int lane = threadIdx.x;
    const int query_row = query_block * BQ + query_row_in_block;
    const int thread_idx = query_row_in_block * BK + lane;
    const bool query_valid = n < N && query_row < L;
    const int batch = n / NUM_HEADS;
    const int query_head = n - batch * NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const int context = context_lens[batch];

    for (int idx = thread_idx; idx < BQ * MAX_D; idx += THREADS) {
        const int row = idx / MAX_D;
        const int dim = idx - row * MAX_D;
        const int global_row = query_block * BQ + row;
        q_tile[idx] = global_row < L && dim < D ? static_cast<float>(q[((long)n * L + global_row) * D + dim]) : 0.0f;
    }
    __syncthreads();

    float output_fragment[OUTPUTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = TL_NEG_INF;
    float running_sum = 0.0f;
    const int total_kv_tiles = (context + BK - 1) / BK;
    int kv_tile_limit = total_kv_tiles;
    if (IS_CAUSAL) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_visible_key = last_query + (context - L);
        kv_tile_limit = min(max((last_visible_key + 1 + BK - 1) / BK, 0), total_kv_tiles);
    }
    const bool single_page_tile = page_size >= BK && page_size % BK == 0;

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        const int tile_start = tile_idx * BK;
        const int tile_logical_page = tile_start / page_size;
        const int tile_page_slot = tile_start - tile_logical_page * page_size;
        if (single_page_tile) {
            if (thread_idx == 0)
                physical_pages[0] = tile_start < context && tile_logical_page < max_pages
                    ? block_table[batch * max_pages + tile_logical_page] : -1;
        } else if (thread_idx < BK) {
            const int key = tile_start + thread_idx;
            const int logical_page = key / page_size;
            physical_pages[thread_idx] = key < context && logical_page < max_pages
                ? block_table[batch * max_pages + logical_page] : -1;
        }
        __syncthreads();
        for (int idx = thread_idx; idx < BK * MAX_D; idx += THREADS) {
            const int key = idx / MAX_D;
            const int dim = idx - key * MAX_D;
            const int global_key = tile_start + key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile ? tile_page_slot + key : global_key - logical_page * page_size;
            const int page_id = single_page_tile ? physical_pages[0] : physical_pages[key];
            const long page_offset = (((long)page_id * NUM_KV_HEADS + kv_head) * page_size + slot) * D;
            kv_tile[idx] = page_id >= 0 && dim < D ? static_cast<float>(key_pages[page_offset + dim]) : 0.0f;
        }
        __syncthreads();

        const int tile_key = lane;
        const int key = tile_start + tile_key;
        const int page_id = single_page_tile ? physical_pages[0] : physical_pages[tile_key];
        bool valid = query_valid && key < context && page_id >= 0;
        if (IS_CAUSAL) valid = valid && key <= query_row + (context - L);
        float score = TL_NEG_INF;
        if (valid) {
            score = 0.0f;
            for (int dim = 0; dim < D; dim++)
                score += q_tile[query_row_in_block * MAX_D + dim] * kv_tile[tile_key * MAX_D + dim];
            score *= scale;
        }
        const float tile_max = tl_warp_max(score);
        const float new_max = fmaxf(running_max, tile_max);
        const bool finite_row = new_max != TL_NEG_INF;
        const float previous_scale = running_max == TL_NEG_INF || !finite_row ? 0.0f : expf(running_max - new_max);
        const float probability = score == TL_NEG_INF || !finite_row ? 0.0f : expf(score - new_max);
        running_max = new_max;
        running_sum = previous_scale * running_sum + tl_warp_sum(probability);
        probabilities[query_row_in_block * BK + lane] = probability;
        __syncthreads();
        for (int idx = thread_idx; idx < BK * MAX_D; idx += THREADS) {
            const int value_key = idx / MAX_D;
            const int dim = idx - value_key * MAX_D;
            const int global_key = tile_start + value_key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile ? tile_page_slot + value_key : global_key - logical_page * page_size;
            const int pid = single_page_tile ? physical_pages[0] : physical_pages[value_key];
            const long page_offset = (((long)pid * NUM_KV_HEADS + kv_head) * page_size + slot) * D;
            kv_tile[idx] = pid >= 0 && dim < D ? static_cast<float>(value_pages[page_offset + dim]) : 0.0f;
        }
        __syncthreads();
        if (query_valid) {
            #pragma unroll
            for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
                const int dim = lane + output_idx * BK;
                if (dim < D) {
                    float partial = 0.0f;
                    for (int value_key = 0; value_key < BK; value_key++)
                        partial += probabilities[query_row_in_block * BK + value_key] * kv_tile[value_key * MAX_D + dim];
                    output_fragment[output_idx] = previous_scale * output_fragment[output_idx] + partial;
                }
            }
        }
        __syncthreads();
    }
    if (query_valid) {
        #pragma unroll
        for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
            const int dim = lane + output_idx * BK;
            if (dim < D)
                out[((long)n * L + query_row) * D + dim] =
                    running_sum == 0.0f ? static_cast<T>(0.0f) : static_cast<T>(output_fragment[output_idx] / running_sum);
        }
    }
"""

_ATTN_INPUTS = ["q", "key_pages", "value_pages", "block_table", "context_lens", "params"]


def _check_paged_attention(q, key_pages, value_pages, block_table, context_lens, num_kv_heads, num_heads):
    if q.dtype not in (mx.float32, mx.bfloat16) or key_pages.dtype != q.dtype or value_pages.dtype != q.dtype:
        raise RuntimeError("paged_attention: q, key_pages, and value_pages must have the same float32 or bfloat16 dtype")
    if block_table.dtype != mx.int32 or context_lens.dtype != mx.int32:
        raise RuntimeError("paged_attention: block_table and context_lens must be int32")
    if q.ndim != 3:
        raise RuntimeError("paged_attention: q must be 3D [B * H_q, L, D]")
    if key_pages.ndim != 4 or value_pages.ndim != 4:
        raise RuntimeError("paged_attention: page tensors must be 4D [P, H_kv, page_size, D]")
    if block_table.ndim != 2 or context_lens.ndim != 1:
        raise RuntimeError("paged_attention: block_table must be 2D and context_lens must be 1D")
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError("paged_attention: num_heads must be divisible by num_kv_heads")
    if q.shape[0] % num_heads != 0:
        raise RuntimeError("paged_attention: q.shape[0] must be divisible by num_heads")
    if key_pages.shape != value_pages.shape:
        raise RuntimeError("paged_attention: key_pages and value_pages must have the same shape")
    if key_pages.shape[1] != num_kv_heads:
        raise RuntimeError("paged_attention: page tensor head count must equal num_kv_heads")
    if q.shape[2] != key_pages.shape[3]:
        raise RuntimeError("paged_attention: q and page tensors must have the same head dimension")
    if block_table.shape[0] != context_lens.shape[0]:
        raise RuntimeError("paged_attention: block_table and context_lens batch sizes must match")
    if q.shape[0] // num_heads != block_table.shape[0]:
        raise RuntimeError("paged_attention: q batch size must match block_table batch size")
    if q.shape[2] <= 0 or q.shape[2] > 128:
        raise RuntimeError("paged_attention: head dimension must be in the range [1, 128]")


def _direct_kw(q, num_kv_heads, num_heads, is_causal):
    N, L, D = q.shape
    template = [("T", q.dtype), ("NUM_HEADS", int(num_heads)), ("NUM_KV_HEADS", int(num_kv_heads)),
                ("IS_CAUSAL", int(bool(is_causal)))]
    if L <= 8:
        fixed = 128 if (q.dtype == mx.bfloat16 and D == 128) else 0
        return (kernel("tl_paged_attention_decode", _ATTN_INPUTS, ["out"], _DECODE_SRC, header=_ATTN_HEADER),
                launch_kw((N * L,), (32 * 32,), [q.shape], [q.dtype], template=template + [("FIXED_D", fixed)]))
    return (kernel("tl_paged_attention_scalar", _ATTN_INPUTS, ["out"], _SCALAR_SRC, header=_ATTN_HEADER),
            launch_kw((N, (L + 15) // 16), (32, 16), [q.shape], [q.dtype], template=template))


_DIRECT_PLANS = {}


def paged_attention_direct(q, key_pages, value_pages, block_table, context_lens, scale, is_causal,
                           num_kv_heads, num_heads, stream=None):
    """Decode kernel for L <= 8, scalar tiled online-softmax kernel otherwise."""
    key = (q.shape, q.dtype, num_kv_heads, num_heads, bool(is_causal))
    plan = _DIRECT_PLANS.get(key)
    if plan is None:
        plan = _DIRECT_PLANS[key] = _direct_kw(q, num_kv_heads, num_heads, is_causal)
    k, kw = plan
    return call(k, [q, key_pages, value_pages, block_table, context_lens, f32(scale)], kw, stream)[0]


# Shapes/dtypes/head counts that already passed _check_paged_attention -> route ("empty", "mma" or "direct").
_ATTN_ROUTES = {}


def _attn_route(query, key_pages, value_pages, block_table, context_lens, num_kv_heads, num_heads):
    _check_paged_attention(query, key_pages, value_pages, block_table, context_lens, num_kv_heads, num_heads)
    if query.size == 0:
        return "empty"
    if query.shape[1] > 8 and query.dtype == mx.bfloat16 and query.shape[2] == 128:
        try:
            from .flash import paged_attention_mma_bf16_d128  # noqa: F401
            return "mma"
        except ImportError:
            pass
    return "direct"


def paged_attention(query, key_pages, value_pages, block_table, context_lens, scale=1.0, is_causal=False,
                    num_kv_heads=None, num_heads=None, stream=None):
    if num_kv_heads is None or num_heads is None:
        raise RuntimeError("paged_attention: num_kv_heads and num_heads are required")
    key = (query.shape, query.dtype, key_pages.shape, key_pages.dtype, value_pages.shape, value_pages.dtype,
           block_table.shape, block_table.dtype, context_lens.shape, context_lens.dtype, num_kv_heads, num_heads)
    route = _ATTN_ROUTES.get(key)
    if route is None:
        route = _ATTN_ROUTES[key] = _attn_route(query, key_pages, value_pages, block_table, context_lens,
                                                 num_kv_heads, num_heads)
    if route == "direct":
        return paged_attention_direct(query, key_pages, value_pages, block_table, context_lens, scale, is_causal,
                                      num_kv_heads, num_heads, stream=stream)
    if route == "empty":
        return mx.zeros(query.shape, dtype=query.dtype)
    from .flash import paged_attention_mma_bf16_d128
    return paged_attention_mma_bf16_d128(query, key_pages, value_pages, block_table, context_lens,
                                         scale, is_causal, num_kv_heads, num_heads, stream=stream)


__all__ = ["paged_cache_update", "paged_attention"]
