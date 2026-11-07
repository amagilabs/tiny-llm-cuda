# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""Gate for the opt-in week-3 `mlp` patch (gate/up + swiglu in one kernel, docs/cuda/DESIGN.md section 9).

Run from the repo root with TINY_LLM_BACKEND=cuda TL_PATCH_WEEK3=all PYTHONPATH=src:. python scripts/cuda/check_mlp.py.
1. kernel: quantized_gate_up_swiglu == swiglu(quantized_matvec(gate), quantized_matvec(up)) bitwise, and its
   error vs swiglu(mx.quantized_matmul(...)) (what upstream week 3 computes) at the level of bf16 rounding
2. model: the patch fires on every layer, custom-kernel calls/step unchanged, QuantizedMatmul nodes -72/step,
   logits close to the bitwise default; greedy-token agreement is reported (not bitwise by construction)
"""

import collections
import re
import sys
import tempfile

import mlx.core as mx

from tiny_llm_cuda_kernels import week2, quantized, week3_patch

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"MLP {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def qweights(K, N, seed):
    mx.random.seed(seed)
    w = (0.05 * mx.random.normal((K, N))).astype(mx.bfloat16)
    wq, s, z = mx.quantize(w, group_size=128, bits=4)
    return wq, s.astype(mx.bfloat16), z.astype(mx.bfloat16)


def mlx_ref(a, g, u):
    mm = [mx.quantized_matmul(a, wq, scales=s, biases=z, transpose=True, group_size=128, bits=4) for wq, s, z in (g, u)]
    return week2.swiglu(mm[0], mm[1])


for (M, N, K) in ((1, 2560, 9728), (2, 2560, 9728), (8, 2560, 9728), (1, 256, 40), (3, 512, 1001)):
    g, u = qweights(K, N, 1), qweights(K, N, 2)
    a = mx.random.normal((M, N)).astype(mx.bfloat16)
    out = quantized.quantized_gate_up_swiglu(a, g[0], g[1], g[2], u[0], u[1], u[2])
    own = week2.swiglu(quantized.quantized_matvec(g[1], g[2], a, g[0]), quantized.quantized_matvec(u[1], u[2], a, u[0]))
    check(f"kernel M={M} N={N} K={K} == own matvec x2 + swiglu", out is not None and bool(mx.array_equal(out, own).item()),
          "(bitwise)")
    for shp in ((1, M, N), (M, 1, N)):  # the ND plan (no reshapes) == the 2-D kernel, bitwise
        a3 = a.reshape(shp)
        pl = quantized.gate_up_swiglu_plan(a3, g[0], g[1], g[2], u[0], u[1], u[2])
        o3 = None if pl is None else quantized.call(pl[0], [a3, g[0], g[1], g[2], u[0], u[1], u[2]], pl[1])[0]
        check(f"nd plan x{shp} K={K} == 2-D kernel", o3 is not None and o3.shape == shp[:-1] + (K,)
              and bool(mx.array_equal(o3.reshape(M, K), out).item()), "(bitwise, no reshape nodes)")
    ref = mlx_ref(a, g, u)
    d = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
    rel = (d.max() / mx.maximum(mx.abs(ref.astype(mx.float32)).max(), 1e-6)).item()
    neq = (d > 0).astype(mx.float32).mean().item()
    check(f"kernel M={M} N={N} K={K} close to MLX quantized_matmul path", rel < 2e-2,
          f"(max |diff| / max |ref| = {rel:.2e}; elements not bit-equal {100 * neq:.1f}%)")
a32 = mx.random.normal((1, 2560))
g = qweights(64, 2560, 3)
check("kernel fallback (f32 activations) -> None",
      quantized.quantized_gate_up_swiglu(a32, g[0], g[1].astype(mx.float32), g[2].astype(mx.float32),
                                         g[0], g[1].astype(mx.float32), g[2].astype(mx.float32)) is None)

# model level
from mlx_lm import load  # noqa: E402
from tiny_llm_ref import models  # noqa: E402
from tiny_llm_ref import qwen3_week3  # noqa: E402,F401

st = week3_patch.patch_status()
check("mlp patch installed", st.get("Qwen3MLP.__call__") == "patched" and week3_patch.ENABLED_MLP,
      f"{st} hashes {week3_patch.source_hashes()}")

from tiny_llm_cuda_kernels import attention as _ka, common as _kc, flash as _kf, matrix as _km, paged as _kp  # noqa: E402
NCALLS = [0]
for _m in (week2, _kp, _ka, _kf, _km, quantized, _kc):
    if not hasattr(_m, "call"):
        continue

    def _counted(k, inputs, kw, stream=None, _o=_m.call):
        NCALLS[0] += 1
        return _o(k, inputs, kw, stream)
    _m.call = _counted

NAME = "Qwen/Qwen3-4B-MLX-4bit"
m, tok = load(NAME)
ids = mx.array(tok.encode("Write a short story about a robot learning to cook."), dtype=mx.int32)


def mlp_prev(mlp, x):
    """The mlp wrapper as of commit 8262675 (reshape to [rows, N], 2-D kernel, reshape back), for the bitwise
    check of the cached-plan wrapper."""
    wg, wu = mlp.w_gate, mlp.w_up
    rows = x.size // x.shape[-1]
    act = quantized.quantized_gate_up_swiglu(x.reshape(rows, x.shape[-1]), wg.weight, wg.scales, wg.biases,
                                             wu.weight, wu.scales, wu.biases)
    return qwen3_week3.quantized_linear(act.reshape(*x.shape[:-1], act.shape[-1]), mlp.w_down)


PLAN_BUILDS = [0]
_plan_orig = week3_patch._mlp_plan


def _plan_counted(self, x):
    PLAN_BUILDS[0] += 1
    return _plan_orig(self, x)


week3_patch._mlp_plan = _plan_counted


def decode(model, n, mlp):
    week3_patch.ENABLED_MLP = mlp
    cache = model.create_kv_cache()
    lg = model(ids[None], 0, cache)
    logits = [lg]
    y = lg[:, -1, :].argmax(-1).astype(mx.int32)
    mx.eval(y)
    off, toks = ids.size, []
    NCALLS[0] = 0
    for i in range(n):
        lg = model(y[None], off, cache, logits_to_keep=1)
        if i < 3:
            logits.append(lg)
        y = lg[:, -1, :].argmax(-1).astype(mx.int32)
        mx.eval(y)
        toks.append(int(y.item()))
        off += 1
    calls = NCALLS[0] / n
    mx.eval(logits)
    for layer in cache:
        layer.release()
    week3_patch.ENABLED_MLP = True
    return toks, logits, calls


def step_graph(model, mlp):
    week3_patch.ENABLED_MLP = mlp
    cache = model.create_kv_cache()
    y = model(ids[None], 0, cache, logits_to_keep=1)[:, -1, :].argmax(-1).astype(mx.int32)
    mx.eval(y)
    out = model(y[None], ids.size, cache, logits_to_keep=1)
    with tempfile.NamedTemporaryFile("w+", suffix=".dot", delete=False) as f:
        path = f.name
    mx.export_to_dot(path, out)
    txt = open(path).read()
    mx.eval(out)
    for layer in cache:
        layer.release()
    week3_patch.ENABLED_MLP = True
    return collections.Counter(mm.group(1) for mm in
                               re.finditer(r'label\s*=\s*"([^"]+)"[^\]]*shape\s*=\s*"?(?:rectangle|box)', txt))


for label, kw in (("paged", dict(week=3, enable_paged_attention=True)),
                  ("dense", dict(week=3, enable_paged_attention=False))):
    model = models.dispatch_model(NAME, m, **kw)
    if label == "paged":
        mx.random.seed(7)
        bad = 0
        for li, layer in enumerate(model.layers_inner):
            for shp in ((1, 1, 2560), (1, 2, 2560), (8, 1, 2560)):
                xx = (0.5 * mx.random.normal(shp)).astype(mx.bfloat16)
                bad += not bool(mx.array_equal(layer.mlp(xx), mlp_prev(layer.mlp, xx)).item())
        check("cached-plan mlp == previous wrapper (8262675), every layer", bad == 0,
              f"(bitwise; {len(model.layers_inner)} layers x 3 decode shapes, mismatches {bad})")
    PLAN_BUILDS[0] = 0
    t0, l0, c0 = decode(model, 64, False)
    f0 = week3_patch.COUNTS["mlp_fused"]
    t1, l1, c1 = decode(model, 64, True)
    nf = week3_patch.COUNTS["mlp_fused"] - f0
    check(f"model {label} plan built once per layer and shape", PLAN_BUILDS[0] <= 2 * 36,
          f"(plan builds over 2 x 64 steps + prefills: {PLAN_BUILDS[0]})")
    check(f"model {label} mlp fires every layer", nf >= 36 * 64, f"(fused MLPs {nf} over 64 steps + prefill)")
    check(f"model {label} calls/step", c1 == c0, f"({c0:.1f} -> {c1:.1f}: swiglu call replaced by the fused call)")
    diffs = []
    for a, b in zip(l0, l1):
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        diffs.append((d.max().item(), mx.abs(b.astype(mx.float32)).max().item()))
    worst = max(dd / mm for dd, mm in diffs)
    check(f"model {label} logits close to bitwise default", worst < 5e-2,
          f"(max |diff| / max |logit| per call: {', '.join(f'{dd:.3f}/{mm:.1f}' for dd, mm in diffs)}; "
          f"prefill argmax equal: {bool(mx.array_equal(l0[0].argmax(-1), l1[0].argmax(-1)).item())})")
    same = next((i for i, (x, y) in enumerate(zip(t0, t1)) if x != y), len(t0))
    print(f"MLP INFO model {label} greedy tokens equal for {same}/64 steps (first 8 off {t0[:8]} on {t1[:8]})",
          flush=True)
    g0, g1 = step_graph(model, False), step_graph(model, True)
    delta = {k: g0.get(k, 0) - g1.get(k, 0) for k in set(g0) | set(g1)}
    delta = {k: v for k, v in sorted(delta.items()) if v}
    check(f"model {label} graph nodes/step", g0.get("QuantizedMatmul", 0) - g1.get("QuantizedMatmul", 0) == 72,
          f"(primitives {sum(g0.values())} -> {sum(g1.values())}; removed per kind {delta})")

print("MLP_ALL_PASS" if ok else "MLP_SOME_FAIL", flush=True)
sys.exit(0 if ok else 1)
