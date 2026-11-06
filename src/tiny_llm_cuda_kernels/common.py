# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file: a CUDA port of the
# Metal kernels in upstream src/extensions_ref/src (Apache-2.0); see NOTICE.
"""Shared helpers for the tiny-llm CUDA kernels (mx.fast.cuda_kernel).

Kernels are built lazily, once per process. Launch geometry is given in CUDA terms
(blocks, threads per block); `launch` converts it to whatever `grid` means for this
MLX build (total threads, Metal-style, or number of blocks), detected once at runtime.
"""

import mlx.core as mx

HEADER = r"""
__device__ __forceinline__ float tl_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float tl_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
"""

_KERNELS = {}
_GRID_IS_THREADS = None
_SCALARS = {}


def kernel(name, inputs, outputs, source, header="", **kw):
    k = _KERNELS.get(name)
    if k is None:
        k = mx.fast.cuda_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=HEADER + header,
            **kw,
        )
        _KERNELS[name] = k
    return k


def _detect_grid_semantics():
    global _GRID_IS_THREADS
    if _GRID_IS_THREADS is not None:
        return _GRID_IS_THREADS
    probe = mx.fast.cuda_kernel(
        name="tl_grid_probe",
        input_names=["x"],
        output_names=["out"],
        source="""
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2048) out[i] = 1;
        """,
    )
    out = probe(
        inputs=[mx.zeros((1,), dtype=mx.int32)],
        grid=(64, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(2048,)],
        output_dtypes=[mx.int32],
        init_value=0,
    )[0]
    n = int(out.sum().item())
    if n == 64:
        _GRID_IS_THREADS = True
    elif n == 2048:
        _GRID_IS_THREADS = False
    else:
        raise RuntimeError(f"cuda_kernel grid probe: unexpected thread count {n}")
    return _GRID_IS_THREADS


def launch_kw(blocks, threads, output_shapes, output_dtypes, template=None, init_value=None):
    """Precompute the keyword arguments of a cuda_kernel call (everything except inputs/stream).

    Wrappers cache the result per (shape, dtype, ...) key so the decode hot path makes one
    dict lookup and one kernel call instead of rebuilding tuples on every token."""
    blocks = tuple(blocks) + (1,) * (3 - len(blocks))
    threads = tuple(threads) + (1,) * (3 - len(threads))
    if _detect_grid_semantics():
        grid = tuple(b * t for b, t in zip(blocks, threads))
    else:
        grid = blocks
    kw = dict(
        grid=grid,
        threadgroup=threads,
        output_shapes=[tuple(s) for s in output_shapes],
        output_dtypes=list(output_dtypes),
    )
    if template:
        kw["template"] = list(template)
    if init_value is not None:
        kw["init_value"] = init_value
    return kw


def call(k, inputs, kw, stream=None):
    if stream is None:
        return k(inputs=inputs, **kw)
    return k(inputs=inputs, stream=stream, **kw)


def launch(k, inputs, blocks, threads, output_shapes, output_dtypes, template=None,
           init_value=None, stream=None):
    return call(k, inputs, launch_kw(blocks, threads, output_shapes, output_dtypes, template, init_value), stream)


def f32(*values):
    """Cached float32 parameter vector (arrays are immutable, so sharing is safe)."""
    a = _SCALARS.get(values)
    if a is None:
        key = tuple(float(v) for v in values)
        a = _SCALARS.get(key)
        if a is None:
            a = mx.array(list(key), dtype=mx.float32)
            _SCALARS[key] = a
        _SCALARS[values] = a
    return a


def i32(*values):
    key = ("i",) + values
    a = _SCALARS.get(key)
    if a is None:
        norm = ("i",) + tuple(int(v) for v in values)
        a = _SCALARS.get(norm)
        if a is None:
            a = mx.array(list(norm[1:]), dtype=mx.int32)
            _SCALARS[norm] = a
        _SCALARS[key] = a
    return a


FLOAT_DTYPES = (mx.float32, mx.float16, mx.bfloat16)


def require_float(x, name):
    if x.dtype not in FLOAT_DTYPES:
        raise RuntimeError(f"{name}: expected float32, float16, or bfloat16")
