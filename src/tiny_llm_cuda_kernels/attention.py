# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Week 2 dense attention kernels, ported from week2_kernels.metal.

decode_attention             -> week2_decode_attention (32 warps per query row, any float dtype, D <= 256)
_decode_attention_online     -> week2_decode_online_bf16_d128 (8 warps, strided K/V), and for
                                256 < S <= 32768 the split-K pair week2_decode_split_bf16_d128 +
                                week2_decode_split_merge_bf16_d128
_dense_attention_prefill_mma -> tiled online-softmax prefill, BF16 D128 (scalar FMA tiles here;
                                the Metal version uses simdgroup_matrix)

Layouts: q [B*Hq, L, D], k/v [B*Hkv, S, D], mask [B*Hq, L, S] float32 (or a 1-element dummy).
"""

import math

import mlx.core as mx

from .common import call, kernel, launch, launch_kw, f32, require_float

_HEADER = r"""
#define TL_NEG_INF (-__int_as_float(0x7f800000))
"""

_DECODE_SRC = r"""
    constexpr int MAX_VALUES_PER_LANE = 8;
    constexpr int WARPS = 32;
    __shared__ float partial_accumulators[WARPS * 256];
    __shared__ float partial_maxima[WARPS];
    __shared__ float partial_sums[WARPS];
    __shared__ float partial_factors[WARPS];
    const int q_rows = q_shape[0];
    const int length = q_shape[1];
    const int dim = q_shape[2];
    const int context = k_shape[1];
    const float scale = params[0];
    const int query_index = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (query_index >= q_rows * length) return;
    const int query_row = query_index / length;
    const int query_position = query_index % length;
    const int batch = query_row / NUM_HEADS;
    const int query_head = query_row % NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const long kv_row = (long)batch * NUM_KV_HEADS + kv_head;
    float accumulator[MAX_VALUES_PER_LANE];
    float query_values[MAX_VALUES_PER_LANE];
    #pragma unroll
    for (int i = 0; i < MAX_VALUES_PER_LANE; ++i) { accumulator[i] = 0.0f; query_values[i] = 0.0f; }
    float max_score = -1e30f;
    float sum = 0.0f;
    const int values_per_lane = (dim + 31) / 32;
    #pragma unroll
    for (int item = 0; item < MAX_VALUES_PER_LANE; ++item) {
        const int d = lane + item * 32;
        if (item < values_per_lane && d < dim)
            query_values[item] = static_cast<float>(q[(long)query_index * dim + d]) * scale;
    }
    for (int position = warp; position < context; position += WARPS) {
        if (IS_CAUSAL && position > context - length + query_position) continue;
        const T* krow = k + (kv_row * context + position) * dim;
        const T* vrow = v + (kv_row * context + position) * dim;
        float partial = 0.0f;
        #pragma unroll
        for (int item = 0; item < MAX_VALUES_PER_LANE; ++item) {
            const int d = lane + item * 32;
            if (item < values_per_lane && d < dim) partial += query_values[item] * static_cast<float>(krow[d]);
        }
        float score = tl_warp_sum(partial);
        if (HAS_MASK) score += mask[(long)query_index * context + position];
        const float new_max = fmaxf(max_score, score);
        const float old_factor = expf(max_score - new_max);
        const float score_factor = expf(score - new_max);
        sum = sum * old_factor + score_factor;
        #pragma unroll
        for (int item = 0; item < MAX_VALUES_PER_LANE; ++item) {
            const int d = lane + item * 32;
            if (item < values_per_lane && d < dim)
                accumulator[item] = accumulator[item] * old_factor + score_factor * static_cast<float>(vrow[d]);
        }
        max_score = new_max;
    }
    if (lane == 0) {
        partial_maxima[warp] = max_score;
        partial_sums[warp] = sum;
    }
    #pragma unroll
    for (int item = 0; item < MAX_VALUES_PER_LANE; ++item) {
        const int d = lane + item * 32;
        if (item < values_per_lane && d < dim) partial_accumulators[warp * dim + d] = accumulator[item];
    }
    __syncthreads();
    if (warp == 0) {
        const float global_max = tl_warp_max(partial_maxima[lane]);
        partial_factors[lane] = expf(partial_maxima[lane] - global_max);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float global_sum = 0.0f;
        for (int group = 0; group < WARPS; ++group) global_sum += partial_sums[group] * partial_factors[group];
        partial_sums[0] = global_sum;
    }
    __syncthreads();
    if (threadIdx.x < dim) {
        float value_sum = 0.0f;
        for (int group = 0; group < WARPS; ++group)
            value_sum += partial_accumulators[group * dim + threadIdx.x] * partial_factors[group];
        out[(long)query_index * dim + threadIdx.x] = static_cast<T>(value_sum / partial_sums[0]);
    }
"""

# Shared body of the online and split decode kernels (8 warps, D = 128, 4 adjacent dims per lane).
_ONLINE_BODY = r"""
    constexpr int dim = 128;
    constexpr int WARPS = 8;
    constexpr int VPL = 4;
    __shared__ float partial_accumulators[WARPS * dim];
    __shared__ float partial_maxima[WARPS];
    __shared__ float partial_sums[WARPS];
    __shared__ float partial_factors[WARPS];
    const int q_rows = q_shape[0];
    const int length = q_shape[1];
    const int context = k_shape[1];
    const float scale = params[0];
    const int query_index = blockIdx.x;
    const int partition = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (query_index >= q_rows * length) return;
    const int query_row = query_index / length;
    const int query_position = query_index % length;
    const int batch = query_row / NUM_HEADS;
    const int query_head = query_row % NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const long kv_row = (long)batch * NUM_KV_HEADS + kv_head;
    const long ks0 = k_strides[0], ks1 = k_strides[1], ks2 = k_strides[2];
    const long vs0 = v_strides[0], vs1 = v_strides[1], vs2 = v_strides[2];
    float accumulator[VPL] = {0.0f, 0.0f, 0.0f, 0.0f};
    float query_values[VPL];
    #pragma unroll
    for (int item = 0; item < VPL; ++item)
        query_values[item] = static_cast<float>(q[(long)query_index * dim + lane * VPL + item]) * scale;
    float max_score = TL_NEG_INF;
    float sum = 0.0f;
    const int begin = PARTITION_SIZE == 0 ? 0 : partition * PARTITION_SIZE;
    const int end = PARTITION_SIZE == 0 ? context : min(context, (partition + 1) * PARTITION_SIZE);
    for (int position = begin + warp; position < end; position += WARPS) {
        if (IS_CAUSAL && position > context - length + query_position) continue;
        const T* krow = k + kv_row * ks0 + (long)position * ks1;
        const T* vrow = v + kv_row * vs0 + (long)position * vs1;
        float partial = 0.0f;
        #pragma unroll
        for (int item = 0; item < VPL; ++item)
            partial += query_values[item] * static_cast<float>(krow[(long)(lane * VPL + item) * ks2]);
        float score = tl_warp_sum(partial);
        if (HAS_MASK) score += mask[(long)query_index * context + position];
        const float new_max = fmaxf(max_score, score);
        const float old_factor = max_score == TL_NEG_INF ? 0.0f : expf(max_score - new_max);
        const float score_factor = score == TL_NEG_INF ? 0.0f : expf(score - new_max);
        sum = sum * old_factor + score_factor;
        #pragma unroll
        for (int item = 0; item < VPL; ++item)
            accumulator[item] = accumulator[item] * old_factor +
                score_factor * static_cast<float>(vrow[(long)(lane * VPL + item) * vs2]);
        max_score = new_max;
    }
    if (lane == 0) {
        partial_maxima[warp] = max_score;
        partial_sums[warp] = sum;
    }
    #pragma unroll
    for (int item = 0; item < VPL; ++item) partial_accumulators[warp * dim + lane * VPL + item] = accumulator[item];
    __syncthreads();
    if (warp == 0) {
        float global_max = TL_NEG_INF;
        for (int group = 0; group < WARPS; ++group) global_max = fmaxf(global_max, partial_maxima[group]);
        if (lane < WARPS)
            partial_factors[lane] = partial_sums[lane] == 0.0f ? 0.0f : expf(partial_maxima[lane] - global_max);
        if (lane == 0) partial_maxima[0] = global_max;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float global_sum = 0.0f;
        for (int group = 0; group < WARPS; ++group) global_sum += partial_sums[group] * partial_factors[group];
        partial_sums[0] = global_sum;
    }
    __syncthreads();
    float value_sum = 0.0f;
    if (threadIdx.x < dim) {
        for (int group = 0; group < WARPS; ++group)
            value_sum += partial_accumulators[group * dim + threadIdx.x] * partial_factors[group];
    }
"""

_ONLINE_SRC = _ONLINE_BODY + r"""
    if (threadIdx.x < dim) {
        const float denominator = partial_sums[0];
        out[(long)query_index * dim + threadIdx.x] = static_cast<T>(denominator == 0.0f ? 0.0f : value_sum / denominator);
    }
"""

_SPLIT_SRC = _ONLINE_BODY + r"""
    const int partitions = gridDim.y;
    const long state = ((long)query_index * partitions + partition) * (dim + 2);
    if (threadIdx.x < dim) partials[state + threadIdx.x] = value_sum;
    if (threadIdx.x == 0) {
        partials[state + dim] = partial_maxima[0];
        partials[state + dim + 1] = partial_sums[0];
    }
"""

_MERGE_SRC = r"""
    constexpr int dim = 128;
    __shared__ float scratch[130];
    const int partitions = states_shape[1];
    const int query = blockIdx.x;
    const int tid = threadIdx.x;
    const float* st = states + (long)query * partitions * (dim + 2);
    if (tid == 0) {
        float maximum = TL_NEG_INF;
        for (int part = 0; part < partitions; ++part) maximum = fmaxf(maximum, st[part * (dim + 2) + dim]);
        scratch[partitions] = maximum;
    }
    __syncthreads();
    for (int part = tid; part < partitions; part += 128) {
        const float s = st[part * (dim + 2) + dim + 1];
        scratch[part] = s == 0.0f ? 0.0f : expf(st[part * (dim + 2) + dim] - scratch[partitions]);
    }
    __syncthreads();
    if (tid == 0) {
        float denominator = 0.0f;
        for (int part = 0; part < partitions; ++part) denominator += st[part * (dim + 2) + dim + 1] * scratch[part];
        scratch[partitions + 1] = denominator;
    }
    __syncthreads();
    float numerator = 0.0f;
    for (int part = 0; part < partitions; ++part) numerator += st[part * (dim + 2) + tid] * scratch[part];
    const float denominator = scratch[partitions + 1];
    out[(long)query * dim + tid] = static_cast<T>(denominator == 0.0f ? 0.0f : numerator / denominator);
"""

# BQ = 16 query rows (one warp each) x BK = 32 keys per tile, D = 128, log2-domain online softmax.
# Probabilities are rounded to BF16 before P.V, as in the Metal simdgroup_matrix kernel.
_PREFILL_SRC = r"""
    constexpr int D = 128;
    constexpr int BQ = 16;
    constexpr int BK = 32;
    constexpr int LDK = D + 1;
    constexpr int OPL = D / 32;
    constexpr int THREADS = BQ * 32;
    constexpr float LOG2_E = 1.44269504089f;
    __shared__ float q_tile[BQ * D];
    __shared__ float kv_tile[BK * LDK];
    __shared__ float probabilities[BQ * BK];
    const int q_rows = q_shape[0];
    const int length = q_shape[1];
    const int context = k_shape[1];
    const float scale_log2 = params[0] * LOG2_E;
    const int query_block = blockIdx.x;
    const int query_row_group = blockIdx.y;
    const int row_in_block = threadIdx.y;
    const int lane = threadIdx.x;
    const int thread_idx = row_in_block * 32 + lane;
    if (query_row_group >= q_rows) return;
    const int batch = query_row_group / NUM_HEADS;
    const int query_head = query_row_group - batch * NUM_HEADS;
    const int kv_head = query_head / (NUM_HEADS / NUM_KV_HEADS);
    const long kv_row = (long)batch * NUM_KV_HEADS + kv_head;
    const int query_position = query_block * BQ + row_in_block;
    const bool query_valid = query_position < length;
    const T* qbase = q + ((long)query_row_group * length) * D;
    for (int idx = thread_idx; idx < BQ * D; idx += THREADS) {
        const int row = idx / D;
        const int global_row = query_block * BQ + row;
        q_tile[idx] = global_row < length ? static_cast<float>(qbase[(long)global_row * D + (idx - row * D)]) : 0.0f;
    }
    __syncthreads();
    float output[OPL] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = TL_NEG_INF;
    float running_sum = 0.0f;
    const int total_tiles = (context + BK - 1) / BK;
    int tile_limit = total_tiles;
    if (IS_CAUSAL) {
        const int last_query = min((query_block + 1) * BQ, length) - 1;
        const int last_key = last_query + (context - length);
        tile_limit = min(max((last_key + 1 + BK - 1) / BK, 0), total_tiles);
    }
    const T* kbase = k + kv_row * context * D;
    const T* vbase = v + kv_row * context * D;
    for (int tile = 0; tile < tile_limit; ++tile) {
        const int tile_start = tile * BK;
        for (int idx = thread_idx; idx < BK * D; idx += THREADS) {
            const int key = idx / D;
            const int d = idx - key * D;
            kv_tile[key * LDK + d] = tile_start + key < context ? static_cast<float>(kbase[(long)(tile_start + key) * D + d]) : 0.0f;
        }
        __syncthreads();
        const int key_position = tile_start + lane;
        bool valid = query_valid && key_position < context;
        if (IS_CAUSAL) valid = valid && key_position <= query_position + (context - length);
        float score = TL_NEG_INF;
        if (valid) {
            float acc = 0.0f;
            #pragma unroll 16
            for (int d = 0; d < D; ++d) acc += q_tile[row_in_block * D + d] * kv_tile[lane * LDK + d];
            score = acc * scale_log2;
            if (HAS_MASK) score += mask[((long)query_row_group * length + query_position) * context + key_position] * LOG2_E;
        }
        const float tile_max = tl_warp_max(score);
        const float new_max = fmaxf(running_max, tile_max);
        const bool finite_row = query_valid && new_max != TL_NEG_INF;
        const float previous_scale = running_max == TL_NEG_INF || !finite_row ? 0.0f : exp2f(running_max - new_max);
        const float p = score == TL_NEG_INF || !finite_row ? 0.0f : exp2f(score - new_max);
        running_max = new_max;
        running_sum = previous_scale * running_sum + tl_warp_sum(p);
        probabilities[row_in_block * BK + lane] = static_cast<float>(static_cast<T>(p));
        __syncthreads();
        for (int idx = thread_idx; idx < BK * D; idx += THREADS) {
            const int key = idx / D;
            const int d = idx - key * D;
            kv_tile[key * LDK + d] = tile_start + key < context ? static_cast<float>(vbase[(long)(tile_start + key) * D + d]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int o = 0; o < OPL; ++o) {
            const int d = lane + o * 32;
            float partial = 0.0f;
            #pragma unroll 8
            for (int key = 0; key < BK; ++key) partial += probabilities[row_in_block * BK + key] * kv_tile[key * LDK + d];
            output[o] = previous_scale * output[o] + partial;
        }
        __syncthreads();
    }
    if (query_valid) {
        #pragma unroll
        for (int o = 0; o < OPL; ++o) {
            const int d = lane + o * 32;
            out[((long)query_row_group * length + query_position) * D + d] =
                running_sum == 0.0f ? static_cast<T>(0.0f) : static_cast<T>(output[o] / running_sum);
        }
    }
"""

_IN = ["q", "k", "v", "mask", "params"]


def _template(dtype, num_heads, num_kv_heads, is_causal, has_mask):
    return [("T", dtype), ("NUM_HEADS", int(num_heads)), ("NUM_KV_HEADS", int(num_kv_heads)),
            ("IS_CAUSAL", int(bool(is_causal))), ("HAS_MASK", int(bool(has_mask)))]


def _attn_key(query, key, value, mask, is_causal, has_mask, num_heads, num_kv_heads):
    return (query.shape, query.dtype, key.shape, key.dtype, value.shape, value.dtype, mask.shape, mask.dtype,
            bool(is_causal), bool(has_mask), num_heads, num_kv_heads)


_DECODE_PLANS = {}


def _decode_plan(query, key, value, mask, is_causal, has_mask, num_heads, num_kv_heads):
    require_float(query, "decode_attention")
    if query.dtype != key.dtype or query.dtype != value.dtype or mask.dtype != mx.float32:
        raise RuntimeError("decode_attention: q, k, and v dtypes must match; mask must be float32")
    if (query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or query.shape[2] > 256
            or query.shape[2] != key.shape[2] or query.shape[2] != value.shape[2] or key.shape != value.shape
            or num_kv_heads <= 0 or num_heads % num_kv_heads != 0):
        raise RuntimeError("decode_attention: incompatible attention shapes")
    if has_mask and (mask.ndim != 3 or mask.shape[0] != query.shape[0] or mask.shape[1] != query.shape[1]
                     or mask.shape[2] != key.shape[1]):
        raise RuntimeError("decode_attention: mask must have shape [B*Hq,L,S]")
    rows = query.shape[0] * query.shape[1]
    if rows == 0:
        return None
    return (kernel("tl_decode_attention", _IN, ["out"], _DECODE_SRC, header=_HEADER),
            launch_kw((rows,), (1024,), [query.shape], [query.dtype],
                      template=_template(query.dtype, num_heads, num_kv_heads, is_causal, has_mask)))


def decode_attention(query, key, value, mask, scale, is_causal, has_mask, num_heads, num_kv_heads, stream=None):
    pkey = _attn_key(query, key, value, mask, is_causal, has_mask, num_heads, num_kv_heads)
    plan = _DECODE_PLANS.get(pkey, False)
    if plan is False:
        plan = _DECODE_PLANS[pkey] = _decode_plan(query, key, value, mask, is_causal, has_mask, num_heads,
                                                  num_kv_heads)
    if plan is None:
        return mx.zeros(query.shape, dtype=query.dtype)
    return call(plan[0], [query, key, value, mask, f32(scale)], plan[1], stream)[0]


_ONLINE_PLANS = {}


def _online_plan(query, key, value, mask, is_causal, has_mask, num_heads, num_kv_heads):
    if (num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0
            or query.ndim != 3 or key.ndim != 3 or value.ndim != 3
            or query.dtype != mx.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype
            or mask.dtype != mx.float32 or query.shape[1] != 1 or query.shape[2] != 128
            or key.shape[2] != 128 or key.shape != value.shape or key.shape[1] <= 0 or query.shape[0] <= 0
            or query.shape[0] % num_heads != 0 or key.shape[0] * num_heads != query.shape[0] * num_kv_heads):
        raise RuntimeError("decode_attention_online: expected nonempty BF16 single-token D128 GQA")
    if has_mask and (mask.ndim != 3 or mask.shape[0] != query.shape[0] or mask.shape[1] != 1
                     or mask.shape[2] != key.shape[1]):
        raise RuntimeError("decode_attention_online: invalid mask shape")
    rows = query.shape[0] * query.shape[1]
    context = key.shape[1]
    template = _template(query.dtype, num_heads, num_kv_heads, is_causal, has_mask)
    if 256 < context <= 32768:
        partitions = (context + 255) // 256
        split = (kernel("tl_decode_split_bf16_d128", _IN, ["partials"], _SPLIT_SRC, header=_HEADER,
                        ensure_row_contiguous=False),
                 launch_kw((rows, partitions), (256,), [(rows, partitions, 130)], [mx.float32],
                           template=template + [("PARTITION_SIZE", 256)]))
        merge = (kernel("tl_decode_split_merge_bf16_d128", ["states"], ["out"], _MERGE_SRC, header=_HEADER),
                 launch_kw((rows,), (128,), [query.shape], [query.dtype], template=[("T", query.dtype)]))
        return (split, merge)
    return ((kernel("tl_decode_online_bf16_d128", _IN, ["out"], _ONLINE_SRC, header=_HEADER,
                    ensure_row_contiguous=False),
             launch_kw((rows, 1), (256,), [query.shape], [query.dtype],
                       template=template + [("PARTITION_SIZE", 0)])), None)


def _decode_attention_online(query, key, value, mask, scale, is_causal, has_mask, num_heads, num_kv_heads,
                             stream=None):
    pkey = _attn_key(query, key, value, mask, is_causal, has_mask, num_heads, num_kv_heads)
    plan = _ONLINE_PLANS.get(pkey)
    if plan is None:
        plan = _ONLINE_PLANS[pkey] = _online_plan(query, key, value, mask, is_causal, has_mask, num_heads,
                                                  num_kv_heads)
    if not math.isfinite(scale):
        raise RuntimeError("decode_attention_online: expected finite scale and contiguous query/mask")
    query = mx.contiguous(query)
    if has_mask:
        mask = mx.contiguous(mask)
    first, merge = plan
    out = call(first[0], [query, key, value, mask, f32(scale)], first[1], stream)[0]
    if merge is None:
        return out
    return call(merge[0], [out], merge[1], stream)[0]


def _dense_attention_prefill_mma(query, key, value, mask, scale, is_causal, has_mask, num_heads, num_kv_heads,
                                 stream=None):
    if (query.dtype != mx.bfloat16 or key.dtype != mx.bfloat16 or value.dtype != mx.bfloat16
            or mask.dtype != mx.float32):
        raise RuntimeError("dense_attention_prefill_mma: expected BF16 q/k/v and float32 mask")
    if (query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or query.shape[2] != 128 or key.shape[2] != 128
            or value.shape[2] != 128 or key.shape != value.shape or query.shape[1] <= 8
            or num_kv_heads <= 0 or num_heads % num_kv_heads != 0
            or query.shape[0] % num_heads != 0 or key.shape[0] * num_heads != query.shape[0] * num_kv_heads):
        raise RuntimeError("dense_attention_prefill_mma: incompatible BF16/D128 attention shapes")
    if has_mask and (mask.ndim != 3 or mask.shape[0] != query.shape[0] or mask.shape[1] != query.shape[1]
                     or mask.shape[2] != key.shape[1]):
        raise RuntimeError("dense_attention_prefill_mma: mask must have shape [B*Hq,L,S]")
    q_rows, length, _ = query.shape
    try:
        from .flash import dense_prefill_wmma
    except ImportError:
        dense_prefill_wmma = None
    if dense_prefill_wmma is not None:
        return dense_prefill_wmma(query, key, value, mask, scale, is_causal, has_mask, num_heads, num_kv_heads,
                                  stream=stream)
    return launch(
        kernel("tl_dense_prefill_bf16_d128", _IN, ["out"], _PREFILL_SRC, header=_HEADER),
        [query, key, value, mask, f32(scale)], ((length + 15) // 16, q_rows), (32, 16), [query.shape],
        [query.dtype], template=_template(query.dtype, num_heads, num_kv_heads, is_causal, has_mask),
        stream=stream,
    )[0]


__all__ = ["decode_attention", "_decode_attention_online", "_dense_attention_prefill_mma"]
