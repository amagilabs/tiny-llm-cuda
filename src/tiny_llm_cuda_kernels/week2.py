# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Week 2 Day 4 fused primitives, ported from week2_kernels.metal to CUDA.

rms_norm   <- week2_rms_norm / week2_rms_norm_register_cached
rope       <- week2_rope
swiglu     <- week2_swiglu
_qk_norm_rope4 <- week2_qk_norm_rope4_bf16 (fused RMSNorm + RoPE, B=L=1, D=128)
_submit_decode_prefix_if_ordinary -> mx.async_eval (no kernel)

Metal -> CUDA: simd_sum -> warp shuffle sum, threadgroup -> __shared__,
simd_shuffle_xor -> __shfl_xor_sync, fast::cos/sin -> __cosf/__sinf.
"""

import collections
import os

import mlx.core as mx

from .common import call, f32, i32, kernel, launch, launch_kw, require_float

# ---------------------------------------------------------------- RMSNorm

_RMS_SRC = r"""
    const int rows = x_shape[0];
    const int dim = x_shape[1];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float eps = params[0];
    __shared__ float partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const size_t base = (size_t)row * dim;
#if REGISTER_CACHED
    float vals[4];
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int col = tid * 4 + s;
        const float v = col < dim ? static_cast<float>(x[base + col]) : 0.0f;
        vals[s] = v;
        sum += v * v;
    }
#else
    float sum = 0.0f;
    for (int col = tid; col < dim; col += blockDim.x) {
        const float v = static_cast<float>(x[base + col]);
        sum += v * v;
    }
#endif
    sum = tl_warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float t = lane < nwarps ? partial[lane] : 0.0f;
        t = tl_warp_sum(t);
        if (lane == 0) partial[0] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(partial[0] / static_cast<float>(dim) + eps);
#if REGISTER_CACHED
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int col = tid * 4 + s;
        if (col < dim)
            out[base + col] = static_cast<T>(vals[s] * inv * static_cast<float>(weight[col]));
    }
#else
    for (int col = tid; col < dim; col += blockDim.x) {
        out[base + col] = static_cast<T>(
            static_cast<float>(x[base + col]) * inv * static_cast<float>(weight[col]));
    }
#endif
"""


_RMS_SRCS = {
    rc: ("#define REGISTER_CACHED %d\n" % int(rc)) + _RMS_SRC + "\n#undef REGISTER_CACHED\n" for rc in (False, True)
}


def _rms_kernel(register_cached):
    name = "tl_week2_rms_norm_rc" if register_cached else "tl_week2_rms_norm"
    return kernel(name, ["x", "weight", "params"], ["out"], _RMS_SRCS[register_cached])


# Remember recent small RMSNorm outputs so _qk_norm_rope4 can fuse them (the C++ version
# inspects the lazy graph's primitive; Python cannot, so we keep a bounded side table).
_RMS_RECENT = collections.OrderedDict()
_RMS_RECENT_MAX = 256

# rope() fuses with a just-made RMSNorm output (week 3 q/k norm -> RoPE, docs/cuda/DESIGN.md candidate A).
# Module flag so checks can A/B it; TL_FUSE_NORM_ROPE=0 in the environment gives the pre-fusion code path.
FUSE_NORM_ROPE = os.environ.get("TL_FUSE_NORM_ROPE", "1") != "0"


# Week 3's FastRoPE hands rope() `mx.contiguous(norm_out)`, a new array object, so the side-table lookup by id
# would never hit. While a fusable RMSNorm output is pending, mx.contiguous is swapped for a wrapper that returns
# such an output (a fresh row-contiguous buffer from our kernel) as is, which is value-identical; anything else
# goes to MLX's contiguous. rope() swaps the original back once the side table is empty, so the ~800 other
# mx.contiguous calls per decode step do not pay for the wrapper. See docs/cuda/DESIGN.md section 3.
_CONTIG_ORIG = getattr(mx.contiguous, "_tl_orig", mx.contiguous)


def _contiguous_passthrough(a, *args, **kwargs):
    src = _RMS_RECENT.get(id(a))
    if src is not None and src[0] is a:
        return a
    if args or kwargs:
        return _CONTIG_ORIG(a, *args, **kwargs)
    return _CONTIG_ORIG(a)


_contiguous_passthrough._tl_orig = _CONTIG_ORIG
_contiguous_passthrough.__doc__ = _CONTIG_ORIG.__doc__


def _release_norm(x_id):
    """Drop a consumed side-table entry; restore MLX's contiguous when nothing is pending."""
    _RMS_RECENT.pop(x_id, None)
    if not _RMS_RECENT and mx.contiguous is _contiguous_passthrough:
        mx.contiguous = _CONTIG_ORIG


# (x.shape, x.dtype, weight.shape, weight.dtype, eps) -> (kernel, params, launch kwargs, 2-D shape, fusable)
_RMS_PLANS = {}


def _rms_plan(x, weight, eps):
    require_float(x, "rms_norm")
    if (x.ndim == 0 or x.shape[-1] <= 0 or x.dtype != weight.dtype or weight.ndim != 1
            or weight.shape[0] != x.shape[-1]):
        raise RuntimeError("rms_norm: weight must match the input dtype and final dimension")
    dim = x.shape[-1]
    rows = x.size // dim
    register_cached = dim <= 4096
    if register_cached:
        needed = (dim + 3) // 4
        threads = ((needed + 31) // 32) * 32
    else:
        threads = 256
    kw = launch_kw((max(rows, 1),), (threads,), [(rows, dim)], [x.dtype], template=[("T", x.dtype)])
    fusable = (x.dtype == mx.bfloat16 and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1
               and x.shape[2] in (8, 32) and x.shape[3] == 128)
    return (_rms_kernel(register_cached), f32(eps), kw, (rows, dim), fusable)


def rms_norm(x, weight, eps, stream=None):
    shape = x.shape
    key = (shape, x.dtype, weight.shape, weight.dtype, eps)
    plan = _RMS_PLANS.get(key)
    if plan is None:
        plan = _RMS_PLANS[key] = _rms_plan(x, weight, eps)
    k, params, kw, shape2d, fusable = plan
    out = call(k, [x.reshape(shape2d) if len(shape) != 2 else x, weight, params], kw, stream)[0]
    if len(shape) != 2:
        out = out.reshape(shape)
    if fusable:
        _RMS_RECENT[id(out)] = (out, x, weight, float(eps))
        if len(_RMS_RECENT) > _RMS_RECENT_MAX:
            _RMS_RECENT.popitem(last=False)
        if FUSE_NORM_ROPE and mx.contiguous is not _contiguous_passthrough:
            mx.contiguous = _contiguous_passthrough
    return out


# ---------------------------------------------------------------- RoPE

_ROPE_SRC = r"""
    const int batch = x_shape[0];
    const int length = x_shape[1];
    const int heads = x_shape[2];
    const int head_dim = x_shape[3];
    const int dims = DIMS;
    const float base = params[0];
    constexpr int heads_per_thread = 4;
    const int half_dim = dims / 2;
    const int tail_dims = head_dim - dims;
    const int items = half_dim + tail_dims;
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    const long total = (long)batch * length * head_blocks * items;
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int item = index % items;
    const int head_block = (index / items) % head_blocks;
    const int l = (index / ((long)items * head_blocks)) % length;
    const int b = index / ((long)items * head_blocks * length);
    const int first_head = head_block * heads_per_thread;
    const int last_head = min(first_head + heads_per_thread, heads);
    const long row_base = ((long)b * length + l) * heads * head_dim;
    if (item >= half_dim) {
        const int d = dims + item - half_dim;
        for (int h = first_head; h < last_head; ++h) {
            const long e = row_base + (long)h * head_dim + d;
            out[e] = x[e];
        }
        return;
    }
    const int pair = item;
    const float fp = -static_cast<float>(pair) / half_dim;
    const float pos = static_cast<float>(offsets[b] + l);
    float c, s;
    if (sizeof(T) == sizeof(float)) {
        const float angle = pos * powf(base, fp);
        c = cosf(angle); s = sinf(angle);
    } else {
        const float angle = pos * exp2f(fp * log2f(base));
        c = __cosf(angle); s = __sinf(angle);
    }
    for (int h = first_head; h < last_head; ++h) {
        const long hb = row_base + (long)h * head_dim;
        const long ri = TRADITIONAL ? hb + pair * 2 : hb + pair;
        const long ii = TRADITIONAL ? ri + 1 : ri + half_dim;
        const float re = static_cast<float>(x[ri]);
        const float im = static_cast<float>(x[ii]);
        out[ri] = static_cast<T>(re * c - im * s);
        out[ii] = static_cast<T>(im * c + re * s);
    }
"""


def _rope_kernel():
    return kernel("tl_week2_rope", ["x", "offsets", "params"], ["out"], _ROPE_SRC)


_ROPE_PLANS = {}


def _rope_plan(x, offsets, dims, base, traditional):
    require_float(x, "rope")
    if x.ndim != 4 or offsets.dtype != mx.int32 or offsets.ndim != 1 or offsets.shape[0] != x.shape[0]:
        raise RuntimeError("rope: expected x=[B,L,H,D] and one int32 offset per batch row")
    if dims <= 0 or dims > x.shape[3] or dims % 2 != 0:
        raise RuntimeError("rope: dims must be positive, even, and no larger than the head dimension")
    b, l, h, d = x.shape
    head_blocks = (h + 3) // 4
    work = b * l * head_blocks * (dims // 2 + d - dims)
    threads = 256
    blocks = max((work + threads - 1) // threads, 1)
    kw = launch_kw((blocks,), (threads,), [x.shape], [x.dtype],
                   template=[("T", x.dtype), ("DIMS", int(dims)), ("TRADITIONAL", bool(traditional))])
    return (_rope_kernel(), f32(base), kw)


def rope(x, offsets, dims, base, traditional=False, stream=None):
    # Fusion (docs/cuda/DESIGN.md, candidate A): if x is a fusable RMSNorm output we just made, run the fused
    # RMSNorm+RoPE kernel on the norm's raw input instead. The norm node then has no consumer and MLX never
    # evaluates it.
    if FUSE_NORM_ROPE and stream is None and _RMS_RECENT:
        src = _RMS_RECENT.get(id(x))
        if src is not None and src[0] is x:
            fkey = (x.shape, offsets.shape, offsets.dtype, dims, base, traditional, src[3])
            plan = _ROPE_FUSED_PLANS.get(fkey)
            if plan is None:
                plan = _ROPE_FUSED_PLANS[fkey] = _rope_fused_plan(x, offsets, dims, base, traditional, src)
            if plan is not False:
                _release_norm(id(x))
                k, params, kw = plan
                return call(k, [src[1], src[2], offsets, params], kw)[0]
    key = (x.shape, x.dtype, offsets.shape, offsets.dtype, dims, base, traditional)
    plan = _ROPE_PLANS.get(key)
    if plan is None:
        plan = _ROPE_PLANS[key] = _rope_plan(x, offsets, dims, base, traditional)
    k, params, kw = plan
    return call(k, [x, offsets, params], kw, stream)[0]


# ---------------------------------------------------------------- fused QK RMSNorm + RoPE (decode)

_QKNR4_SRC = r"""
    const float eps = params[0];
    const float base = params[1];
    const int dim = 128;
    __shared__ float cs[64];
    __shared__ float sn[64];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (tid < 64) {
        const float fp = -static_cast<float>(tid) / (dim / 2);
        const float angle = static_cast<float>(offsets[0]) * exp2f(fp * log2f(base));
        cs[tid] = __cosf(angle);
        sn[tid] = __sinf(angle);
    }
    __syncthreads();
    const int head = blockIdx.x * 4 + warp;
    float v[4];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float t = static_cast<float>(x[head * 128 + lane * 4 + k]);
        v[k] = t;
        sum += t * t;
    }
    sum = tl_warp_sum(sum);
    const float inv = rsqrtf(sum / static_cast<float>(dim) + eps);
    float r[4], p[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        r[k] = static_cast<float>(static_cast<T>(v[k] * inv * static_cast<float>(weight[lane * 4 + k])));
        p[k] = __shfl_xor_sync(0xffffffffu, r[k], 16);
    }
    if (lane < 16) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair = lane * 4 + k;
            const float c = cs[pair];
            const float s = sn[pair];
            out[head * 128 + pair] = static_cast<T>(r[k] * c - p[k] * s);
            out[head * 128 + pair + 64] = static_cast<T>(p[k] * c + r[k] * s);
        }
    }
"""


def _qknr4_kernel():
    return kernel("tl_week2_qk_norm_rope4", ["x", "weight", "offsets", "params"], ["out"], _QKNR4_SRC)


_QKNR4_PLANS = {}


def _qknr4_plan(shape, dtype, eps, base):
    key = (shape, dtype, eps, base)
    plan = _QKNR4_PLANS.get(key)
    if plan is None:
        kw = launch_kw((shape[2] // 4,), (128,), [shape], [dtype], template=[("T", dtype)])
        plan = _QKNR4_PLANS[key] = (_qknr4_kernel(), f32(eps, base), kw)
    return plan


# (x.shape, offsets.shape, offsets.dtype, dims, base, traditional, eps) -> qknr4 plan, or False if not fusable
_ROPE_FUSED_PLANS = {}


def _rope_fused_plan(x, offsets, dims, base, traditional, src):
    _, raw, weight, eps = src
    ok = (
        x.dtype == mx.bfloat16 and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1
        and x.shape[2] in (8, 32) and x.shape[3] == 128 and raw.shape == x.shape and raw.dtype == x.dtype
        and dims == 128 and not traditional and base > 0 and base != float("inf")
        and weight.dtype == mx.bfloat16 and weight.shape == (128,) and eps > 0
        and offsets.dtype == mx.int32 and offsets.ndim == 1 and offsets.shape[0] == 1
    )
    return _qknr4_plan(x.shape, x.dtype, eps, base) if ok else False


def _qk_norm_rope4(normalized, offsets, dims, base, traditional=False, stream=None, *, weight=None, eps=None):
    """Fused q/k RMSNorm + RoPE (decode, bf16, B = L = 1, H in {8, 32}, D = dims = 128).

    Upstream week 2 passes the RMSNorm *output* and the C++ side fuses through the lazy graph; here that is
    the side-table lookup below. With weight= and eps= given (the week-3 call-site patch, week3_patch.py),
    `normalized` is the raw projection instead and no rms_norm call is made at all: one kernel call per q/k
    rather than two."""
    if weight is not None:
        return _qk_norm_rope4_raw(normalized, weight, eps, offsets, dims, base, traditional, stream)
    require_float(normalized, "rope")
    if (normalized.ndim != 4 or offsets.dtype != mx.int32 or offsets.ndim != 1
            or offsets.shape[0] != normalized.shape[0]):
        raise RuntimeError("rope: expected x=[B,L,H,D] and one int32 offset per batch row")
    if dims <= 0 or dims > normalized.shape[3] or dims % 2 != 0:
        raise RuntimeError("rope: dims must be positive, even, and no larger than the head dimension")
    src = _RMS_RECENT.get(id(normalized))
    fusable = (
        src is not None and src[0] is normalized
        and normalized.dtype == mx.bfloat16 and normalized.shape[0] == 1 and normalized.shape[1] == 1
        and normalized.shape[2] in (8, 32) and normalized.shape[3] == 128 and dims == 128
        and not traditional and base > 0 and base != float("inf")
        and src[2].dtype == mx.bfloat16 and src[2].shape == (128,) and src[3] > 0
        and mx.default_device() == mx.gpu
    )
    if not fusable:
        return rope(mx.contiguous(normalized), offsets, dims, base, traditional, stream=stream)
    _, x, weight, eps = src
    _release_norm(id(normalized))
    k, params, kw = _qknr4_plan(normalized.shape, normalized.dtype, eps, base)
    return call(k, [x, weight, offsets, params], kw, stream)[0]


# (x.shape, x.dtype, weight.shape, weight.dtype, eps, offsets.shape, offsets.dtype, dims, base, traditional)
#   -> qknr4 plan, or False (not fusable: run the two-call rms_norm -> rope path)
_RAW_PLANS = {}


def _qk_norm_rope4_raw(x, weight, eps, offsets, dims, base, traditional, stream):
    key = (x.shape, x.dtype, weight.shape, weight.dtype, eps, offsets.shape, offsets.dtype, dims, base,
           traditional)
    plan = _RAW_PLANS.get(key)
    if plan is None:
        ok = (
            x.dtype == mx.bfloat16 and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1
            and x.shape[2] in (8, 32) and x.shape[3] == 128 and dims == 128 and not traditional
            and base > 0 and base != float("inf") and weight.dtype == mx.bfloat16 and weight.shape == (128,)
            and eps is not None and eps > 0 and offsets.dtype == mx.int32 and offsets.shape == (1,)
            and mx.default_device() == mx.gpu
        )
        plan = _RAW_PLANS[key] = _qknr4_plan(x.shape, x.dtype, float(eps), base) if ok else False
    if plan is False:
        return rope(rms_norm(x, weight, eps, stream=stream), offsets, dims, base, traditional, stream=stream)
    k, params, kw = plan
    return call(k, [x, weight, offsets, params], kw, stream)[0]


# q and k in one launch (week-3 call-site patch, docs/cuda/DESIGN.md section 7). Same per-head math as
# _QKNR4_SRC; blocks [0, Hq/4) take q heads, the rest take k heads (4 heads per block, one per warp, so the
# q/k branch is uniform per warp). Each block builds the same cos/sin table from the shared offset.
_QKNR4_PAIR_SRC = r"""
    const float base = params[2];
    const int dim = 128;
    const int hq = xq_shape[2];
    __shared__ float cs[64];
    __shared__ float sn[64];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (tid < 64) {
        const float fp = -static_cast<float>(tid) / (dim / 2);
        const float angle = static_cast<float>(offsets[0]) * exp2f(fp * log2f(base));
        cs[tid] = __cosf(angle);
        sn[tid] = __sinf(angle);
    }
    __syncthreads();
    const int head_all = blockIdx.x * 4 + warp;
    const bool is_q = head_all < hq;
    const int head = is_q ? head_all : head_all - hq;
    const float eps = is_q ? params[0] : params[1];
    float v[4];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float t = is_q ? static_cast<float>(xq[head * 128 + lane * 4 + k])
                             : static_cast<float>(xk[head * 128 + lane * 4 + k]);
        v[k] = t;
        sum += t * t;
    }
    sum = tl_warp_sum(sum);
    const float inv = rsqrtf(sum / static_cast<float>(dim) + eps);
    float r[4], p[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float w = is_q ? static_cast<float>(wq[lane * 4 + k]) : static_cast<float>(wk[lane * 4 + k]);
        r[k] = static_cast<float>(static_cast<T>(v[k] * inv * w));
        p[k] = __shfl_xor_sync(0xffffffffu, r[k], 16);
    }
    if (lane < 16) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair = lane * 4 + k;
            const float c = cs[pair];
            const float s = sn[pair];
            const T lo = static_cast<T>(r[k] * c - p[k] * s);
            const T hi = static_cast<T>(p[k] * c + r[k] * s);
            if (is_q) {
                out_q[head * 128 + pair] = lo;
                out_q[head * 128 + pair + 64] = hi;
            } else {
                out_k[head * 128 + pair] = lo;
                out_k[head * 128 + pair + 64] = hi;
            }
        }
    }
"""

# (q.shape, k.shape, dtypes, weight shapes/dtypes, eps_q, eps_k, offsets.shape/dtype, dims, base, traditional)
#   -> (kernel, params, launch kwargs), or False (not fusable: two single-output calls)
_PAIR_PLANS = {}


def _qk_norm_rope4_pair(q, k, wq, wk, eps_q, eps_k, offsets, dims, base, traditional=False, stream=None):
    """RMSNorm + RoPE of raw q [1,1,Hq,128] and raw k [1,1,Hk,128] in one kernel launch.

    Returns (q_out, k_out), each bit-identical to _qk_norm_rope4(raw, offsets, ..., weight=, eps=) on its own.
    Falls back to those two calls outside bf16 / B = L = 1 / H in {8, 32} / D = dims = 128 / one int32 offset."""
    key = (q.shape, q.dtype, k.shape, k.dtype, wq.shape, wq.dtype, wk.shape, wk.dtype, eps_q, eps_k,
           offsets.shape, offsets.dtype, dims, base, traditional)
    plan = _PAIR_PLANS.get(key)
    if plan is None:
        ok = (
            q.dtype == mx.bfloat16 and k.dtype == mx.bfloat16 and q.ndim == 4 and k.ndim == 4
            and q.shape[:2] == (1, 1) and k.shape[:2] == (1, 1) and q.shape[2] in (8, 32) and k.shape[2] in (8, 32)
            and q.shape[3] == 128 and k.shape[3] == 128 and dims == 128 and not traditional
            and base > 0 and base != float("inf")
            and wq.dtype == mx.bfloat16 and wq.shape == (128,) and wk.dtype == mx.bfloat16 and wk.shape == (128,)
            and eps_q is not None and eps_q > 0 and eps_k is not None and eps_k > 0
            and offsets.dtype == mx.int32 and offsets.shape == (1,) and mx.default_device() == mx.gpu
        )
        if ok:
            kw = launch_kw(((q.shape[2] + k.shape[2]) // 4,), (128,), [q.shape, k.shape], [q.dtype, k.dtype],
                           template=[("T", q.dtype)])
            kern = kernel("tl_week2_qk_norm_rope4_pair", ["xq", "xk", "wq", "wk", "offsets", "params"],
                          ["out_q", "out_k"], _QKNR4_PAIR_SRC)
            plan = (kern, f32(eps_q, eps_k, base), kw)
        else:
            plan = False
        _PAIR_PLANS[key] = plan
    if plan is False:
        return (_qk_norm_rope4_raw(q, wq, eps_q, offsets, dims, base, traditional, stream),
                _qk_norm_rope4_raw(k, wk, eps_k, offsets, dims, base, traditional, stream))
    kern, params, kw = plan
    out = call(kern, [q, k, wq, wk, offsets, params], kw, stream)
    return out[0], out[1]


# ---------------------------------------------------------------- residual add + RMSNorm (week-3 block)

# h = x + r (rounded to T, exactly as MLX's add), out = rms_norm(h): the register-cached RMSNorm above with the
# add folded into its load, so h's rounding, the sum order and the reduction match `rms_norm(x + r)` bitwise.
# dim comes from weight_shape[0]; one block per row, any leading shape (inputs are row-contiguous).
_ADD_RMS_SRC = r"""
    const int dim = weight_shape[0];
    const int row = blockIdx.x;
    const float eps = params[0];
    __shared__ float partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const size_t base = (size_t)row * dim;
    float vals[4];
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int col = tid * 4 + s;
        float v = 0.0f;
        if (col < dim) {
            const T hv = static_cast<T>(static_cast<float>(x[base + col]) + static_cast<float>(r[base + col]));
            h[base + col] = hv;
            v = static_cast<float>(hv);
        }
        vals[s] = v;
        sum += v * v;
    }
    sum = tl_warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float t = lane < nwarps ? partial[lane] : 0.0f;
        t = tl_warp_sum(t);
        if (lane == 0) partial[0] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(partial[0] / static_cast<float>(dim) + eps);
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int col = tid * 4 + s;
        if (col < dim)
            out[base + col] = static_cast<T>(vals[s] * inv * static_cast<float>(weight[col]));
    }
"""

# (x.shape, x.dtype, r.shape, r.dtype, weight.shape, weight.dtype, eps) -> (kernel, params, kwargs) or False
_ADD_RMS_PLANS = {}


def add_rms_norm(x, r, weight, eps, stream=None):
    """(x + r, rms_norm(x + r)) in one kernel launch; both bit-identical to the two-op version.

    Fast path: bf16, x.shape == r.shape, last dim <= 4096 (the register-cached RMSNorm), GPU. Otherwise
    the add is an MLX op and the norm is rms_norm(), exactly what upstream's block does."""
    key = (x.shape, x.dtype, r.shape, r.dtype, weight.shape, weight.dtype, eps)
    plan = _ADD_RMS_PLANS.get(key)
    if plan is None:
        dim = x.shape[-1] if x.ndim else 0
        ok = (
            x.dtype == mx.bfloat16 and r.dtype == x.dtype and r.shape == x.shape and x.ndim >= 1
            and 0 < dim <= 4096 and weight.dtype == x.dtype and weight.shape == (dim,) and eps > 0
            and x.size > 0 and mx.default_device() == mx.gpu
        )
        if ok:
            rows = x.size // dim
            threads = (((dim + 3) // 4 + 31) // 32) * 32
            kw = launch_kw((rows,), (threads,), [x.shape, x.shape], [x.dtype, x.dtype], template=[("T", x.dtype)])
            kern = kernel("tl_week2_add_rms_norm", ["x", "r", "weight", "params"], ["h", "out"], _ADD_RMS_SRC)
            plan = (kern, f32(eps), kw)
        else:
            plan = False
        _ADD_RMS_PLANS[key] = plan
    if plan is False:
        h = x + r
        return h, rms_norm(mx.contiguous(h), weight, eps, stream=stream)
    kern, params, kw = plan
    out = call(kern, [x, r, weight, params], kw, stream)
    return out[0], out[1]


# ---------------------------------------------------------------- SwiGLU

_SWIGLU_SRC = r"""
    const long n = gate_shape[0];
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = static_cast<float>(gate[i]);
    out[i] = static_cast<T>((g / (1.0f + expf(-g))) * static_cast<float>(up[i]));
"""


_SWIGLU_PLANS = {}


def _swiglu_plan(gate, up):
    require_float(gate, "swiglu")
    if gate.dtype != up.dtype or gate.shape != up.shape:
        raise RuntimeError("swiglu: gate and up must have the same shape and dtype")
    n = gate.size
    threads = 256
    blocks = max((n + threads - 1) // threads, 1)
    kw = launch_kw((blocks,), (threads,), [(n,)], [gate.dtype], template=[("T", gate.dtype)])
    return (kernel("tl_week2_swiglu", ["gate", "up"], ["out"], _SWIGLU_SRC), kw, (n,))


def swiglu(gate, up, stream=None):
    shape = gate.shape
    key = (shape, gate.dtype, up.shape, up.dtype)
    plan = _SWIGLU_PLANS.get(key)
    if plan is None:
        plan = _SWIGLU_PLANS[key] = _swiglu_plan(gate, up)
    k, kw, flat = plan
    if len(shape) == 1:
        return call(k, [gate, up], kw, stream)[0]
    return call(k, [gate.reshape(flat), up.reshape(flat)], kw, stream)[0].reshape(shape)


# ---------------------------------------------------------------- decode prefix submit

def _submit_decode_prefix_if_ordinary(root):
    """Metal build: async_eval the first half of the decode graph so the GPU starts early.

    The C++ guard (unscheduled GPU primitive, not a tracer) is not visible from Python;
    mx.async_eval is a no-op on already-evaluated arrays, so calling it is always safe."""
    if mx.default_device() != mx.gpu:
        return False
    mx.async_eval(root)
    return True


__all__ = ["rms_norm", "rope", "_qk_norm_rope4", "_qk_norm_rope4_pair", "add_rms_norm", "swiglu", "_submit_decode_prefix_if_ordinary"]
