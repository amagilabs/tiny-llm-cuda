# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""Correctness check for the week-3 q/k RMSNorm+RoPE fusion (docs/cuda/DESIGN.md). Run from the repo root on CUDA:
``TINY_LLM_BACKEND=cuda PYTHONPATH=src:. python scripts/cuda/check_fusion.py``.

1. kernel level: rope(mx.contiguous(rms_norm(x))) fused vs unfused, bit-exact, plus close to mx.fast reference
2. passthrough: mx.contiguous returns a registered norm output itself and anything else via MLX
3. model level: week3 paged + dense, greedy decode, fused vs unfused tokens identical, and the fusion fires on
   every q/k pair (side table empty after a decode step), with the call-site patch off
4. call-site patch (src/tiny_llm_cuda_kernels/week3_patch.py, docs/cuda/DESIGN.md section 6): raw-input _qk_norm_rope4 and
   paged_cache_update_kv bit-exact vs the calls they replace; tokens identical patched vs unpatched; custom
   kernel calls per decode step drop by 72 (qk) and 36 more (kv); upstream sources untouched
5. section 7: q+k pair kernel (one launch per layer, -36 calls/step) and the block's residual add +
   post_attention_layernorm kernel (add_rms_norm): bit-exact, tokens identical, call deltas
6. section 8 (fold): each block's out = h + r folded into the next input_layernorm / final norm. Sliced
   add+norm bit-exact, logits bitwise vs fold off and vs unpatched, tokens identical, calls unchanged,
   graph Add -36
"""

import sys

import mlx.core as mx

from tiny_llm_cuda_kernels import week2, paged, week3_patch

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"FUSE {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def pair(x, w, eps, off, base, fuse):
    week2.FUSE_NORM_ROPE = fuse
    n = week2.rms_norm(x, w, eps)
    return week2.rope(mx.contiguous(n), off, 128, base, False)


mx.random.seed(0)
for H in (8, 32):
    for pos in (0, 13, 300, 4095):
        x = (3 * mx.random.normal((1, 1, H, 128))).astype(mx.bfloat16)
        w = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
        off = mx.array([pos], dtype=mx.int32)
        a = pair(x, w, 1e-6, off, 1000000, True)
        b = pair(x, w, 1e-6, off, 1000000, False)
        # mx.fast.rope rotates along axis -2, so put L there: [B, L, H, D] -> [B, H, L, D] and back.
        ref = mx.fast.rope(mx.fast.rms_norm(x, w, 1e-6).transpose(0, 2, 1, 3), 128, traditional=False,
                           base=1000000.0, scale=1.0, offset=pos).transpose(0, 2, 1, 3)
        err = float(mx.max(mx.abs(a.astype(mx.float32) - ref.astype(mx.float32))).item())
        same = bool(mx.array_equal(a, b).item())
        check(f"kernel H={H} pos={pos}", same and err < 0.25,
              f"fused==unfused bitwise: {same}, max|fused-mx.fast| {err:.4f} (bf16, |x|max ~{float(mx.max(mx.abs(b)).item()):.1f})")
week2.FUSE_NORM_ROPE = True

x = mx.random.normal((1, 1, 32, 128)).astype(mx.bfloat16)
w = mx.ones((128,), dtype=mx.bfloat16)
n = week2.rms_norm(x, w, 1e-6)
check("passthrough registered", mx.contiguous(n) is n and getattr(mx.contiguous, "_tl_orig", None) is not None)
check("passthrough other", mx.contiguous(x) is not x and bool(mx.array_equal(mx.contiguous(x), x).item()))
t = mx.random.normal((4, 6)).T
check("passthrough non-contiguous input", bool(mx.array_equal(mx.contiguous(t), t).item()))
# different offsets per call still use their own offsets (no stash across calls)
o1, o2 = mx.array([5], dtype=mx.int32), mx.array([77], dtype=mx.int32)
week2._RMS_RECENT.clear()
q = week2.rms_norm(x, w, 1e-6)
k = week2.rms_norm(x, w, 1e-6)
check("wrapper installed while pending", mx.contiguous is week2._contiguous_passthrough)
rq = week2.rope(mx.contiguous(q), o1, 128, 1e6)
rk = week2.rope(mx.contiguous(k), o2, 128, 1e6)
check("wrapper removed once side table drained", mx.contiguous is week2._CONTIG_ORIG,
      f"(pending {len(week2._RMS_RECENT)})")
week2.FUSE_NORM_ROPE = False
uq = week2.rope(mx.contiguous(week2.rms_norm(x, w, 1e-6)), o1, 128, 1e6)
uk = week2.rope(mx.contiguous(week2.rms_norm(x, w, 1e-6)), o2, 128, 1e6)
week2.FUSE_NORM_ROPE = True
check("own offsets per call", bool(mx.array_equal(rq, uq).item()) and bool(mx.array_equal(rk, uk).item()))
week2._RMS_RECENT.clear()

# call-site patch, kernel level
for H in (8, 32):
    for pos in (0, 300, 4095):
        x = (3 * mx.random.normal((1, 1, H, 128))).astype(mx.bfloat16)
        w = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
        off = mx.array([pos], dtype=mx.int32)
        raw = week2._qk_norm_rope4(x, off, 128, 1000000, False, weight=w, eps=1e-6)
        ref = pair(x, w, 1e-6, off, 1000000, False)
        week2._RMS_RECENT.clear()
        check(f"qk raw H={H} pos={pos}", bool(mx.array_equal(raw, ref).item()), "(raw-input qknr4 == rms_norm->rope, bitwise)")
week2.FUSE_NORM_ROPE = True
# q and k in one launch (fusion_plan section 7): each output == its own raw-input qknr4 call, bitwise
for (HQ, HK) in ((32, 8), (8, 8), (8, 32), (32, 32)):
    for pos in (0, 300, 4095):
        xq = (3 * mx.random.normal((1, 1, HQ, 128))).astype(mx.bfloat16)
        xk = (3 * mx.random.normal((1, 1, HK, 128))).astype(mx.bfloat16)
        wq = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
        wk = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
        off = mx.array([pos], dtype=mx.int32)
        pq, pk = week2._qk_norm_rope4_pair(xq, xk, wq, wk, 1e-6, 1e-5, off, 128, 1000000)
        rq = week2._qk_norm_rope4(xq, off, 128, 1000000, False, weight=wq, eps=1e-6)
        rk = week2._qk_norm_rope4(xk, off, 128, 1000000, False, weight=wk, eps=1e-5)
        same = bool(mx.array_equal(pq, rq).item()) and bool(mx.array_equal(pk, rk).item())
        check(f"qk pair Hq={HQ} Hk={HK} pos={pos}", same and pq.shape == xq.shape and pk.shape == xk.shape,
              "(one launch == two raw qknr4 calls, bitwise, distinct eps)")
# outside the fast path the pair falls back to the two single calls (f32 here)
xq32 = mx.random.normal((1, 1, 32, 128))
xk32 = mx.random.normal((1, 1, 8, 128))
w32 = mx.ones((128,))
fq, fk = week2._qk_norm_rope4_pair(xq32, xk32, w32, w32, 1e-6, 1e-6, mx.array([3], dtype=mx.int32), 128, 1e6)
rq = week2.rope(week2.rms_norm(xq32, w32, 1e-6), mx.array([3], dtype=mx.int32), 128, 1e6)
week2._RMS_RECENT.clear()
check("qk pair fallback f32", bool(mx.array_equal(fq, rq).item()) and fk.shape == xk32.shape)
# residual add + RMSNorm in one kernel: h == MLX x + r and out == rms_norm(contiguous(x + r)), bitwise
for shape in ((1, 1, 2560), (1, 7, 2560), (1, 1, 4096), (1, 1, 128), (3, 2, 1000)):
    x = (2 * mx.random.normal(shape)).astype(mx.bfloat16)
    r = (2 * mx.random.normal(shape)).astype(mx.bfloat16)
    w = (1 + 0.2 * mx.random.normal((shape[-1],))).astype(mx.bfloat16)
    h, n = week2.add_rms_norm(x, r, w, 1e-6)
    rh = x + r
    rn = week2.rms_norm(mx.contiguous(rh), w, 1e-6)
    week2._RMS_RECENT.clear()
    same = bool(mx.array_equal(h, rh).item()) and bool(mx.array_equal(n, rn).item())
    check(f"add+norm {shape}", same and h.shape == shape and n.shape == shape,
          "(h == x + r and norm(h) == rms_norm(x + r), bitwise)")
# large-magnitude mix (rounding of the add matters), and a non-contiguous residual input
x = (300 * mx.random.normal((1, 1, 2560))).astype(mx.bfloat16)
r = (0.01 * mx.random.normal((1, 1, 2560))).astype(mx.bfloat16)
w = mx.ones((2560,), dtype=mx.bfloat16)
h, n = week2.add_rms_norm(x, r, w, 1e-6)
check("add+norm rounding", bool(mx.array_equal(h, x + r).item())
      and bool(mx.array_equal(n, week2.rms_norm(mx.contiguous(x + r), w, 1e-6)).item()))
xt = mx.random.normal((2560, 1)).astype(mx.bfloat16).T.reshape(1, 1, 2560)
rt = mx.random.normal((1, 2560)).astype(mx.bfloat16)[:, ::-1].reshape(1, 1, 2560)
h, n = week2.add_rms_norm(xt, rt, w, 1e-6)
check("add+norm non-contiguous input", bool(mx.array_equal(h, xt + rt).item())
      and bool(mx.array_equal(n, week2.rms_norm(mx.contiguous(xt + rt), w, 1e-6)).item()))
# fallback (f32, dim > 4096): MLX add + rms_norm
x32, r32 = mx.random.normal((1, 1, 5000)), mx.random.normal((1, 1, 5000))
h, n = week2.add_rms_norm(x32, r32, mx.ones((5000,)), 1e-6)
check("add+norm fallback", bool(mx.array_equal(h, x32 + r32).item())
      and bool(mx.array_equal(n, week2.rms_norm(x32 + r32, mx.ones((5000,)), 1e-6)).item()))
# fold: the final norm sees (h + r)[:, -k:] -> add_rms_norm(h[:, -k:], r[:, -k:]) must match bitwise
x = mx.random.normal((1, 7, 2560)).astype(mx.bfloat16)
r = mx.random.normal((1, 7, 2560)).astype(mx.bfloat16)
w = (1 + 0.1 * mx.random.normal((2560,))).astype(mx.bfloat16)
_, n = week2.add_rms_norm(x[:, -1:, :], r[:, -1:, :], w, 1e-6)
check("add+norm on sliced rows (fold final norm)",
      bool(mx.array_equal(n, week2.rms_norm(mx.contiguous((x + r)[:, -1:, :]), w, 1e-6)).item()))
# chained: out = h + r of one block, then the next block's input norm, two add+norm kernels deep
h1, _ = week2.add_rms_norm(x, r, w, 1e-6)
h2, n2 = week2.add_rms_norm(h1, x, w, 1e-6)
check("add+norm chained (fold across layers)", bool(mx.array_equal(h2, (x + r) + x).item())
      and bool(mx.array_equal(n2, week2.rms_norm(mx.contiguous((x + r) + x), w, 1e-6)).item()))
week2._RMS_RECENT.clear()
for dt in (mx.bfloat16, mx.float32):
    for (page_id, start, length) in ((0, 0, 1), (2, 127, 1), (3, 40, 5), (1, 0, 128)):
        kp = mx.random.normal((4, 8, 128, 128)).astype(dt)
        vp = mx.random.normal((4, 8, 128, 128)).astype(dt)
        kk = mx.random.normal((1, 8, length, 128)).astype(dt)
        vv = mx.random.normal((1, 8, length, 128)).astype(dt)
        fk, fv = paged.paged_cache_update_kv(kp, vp, kk, vv, page_id, start)
        rk = paged.paged_cache_update(kp, kk, page_id, start)
        rv = paged.paged_cache_update(vp, vv, page_id, start)
        same = bool(mx.array_equal(fk, rk).item()) and bool(mx.array_equal(fv, rv).item())
        check(f"kv update {dt} page={page_id} start={start} len={length}", same, "(fused K+V == two updates, bitwise)")
# non-contiguous K/V input (the model passes transposed slices)
kp = mx.zeros((4, 8, 128, 128), dtype=mx.bfloat16)
kt = mx.random.normal((1, 3, 8, 128)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
fk, fv = paged.paged_cache_update_kv(kp, kp, kt, kt, 1, 7)
rk = paged.paged_cache_update(kp, mx.contiguous(kt), 1, 7)
check("kv update non-contiguous input", bool(mx.array_equal(fk, rk).item()) and bool(mx.array_equal(fv, rk).item()))

# model level
from mlx_lm import load  # noqa: E402
from tiny_llm_ref import models  # noqa: E402
from tiny_llm_ref import paged_kv_cache, qwen3_week3  # noqa: E402

st = week3_patch.patch_status()
check("patch installed", st.get("Qwen3MultiHeadAttention.__call__") == "patched"
      and st.get("Qwen3TransformerBlock.__call__") == "patched"
      and st.get("TinyKvPagedPool.write_page_slice") == "patched"
      and st.get("Qwen3ModelWeek3.__call__") == "patched", f"{st} hashes {week3_patch.source_hashes()}")

# count custom-kernel launches: every wrapper calls its module-global `call`

from tiny_llm_cuda_kernels import attention as _ka, common as _kc, flash as _kf, matrix as _km, quantized as _kq  # noqa: E402
NCALLS = [0]
for _m in (week2, paged, _ka, _kf, _km, _kq, _kc):  # _kc last-or-first is fine: modules hold the original
    if not hasattr(_m, "call"):
        continue
    _orig_call = _m.call

    def _counted(k, inputs, kw, stream=None, _o=_orig_call):
        NCALLS[0] += 1
        return _o(k, inputs, kw, stream)
    _m.call = _counted

NAME = "Qwen/Qwen3-4B-MLX-4bit"
m, tok = load(NAME)
ids = mx.array(tok.encode("Write a short story about a robot learning to cook."), dtype=mx.int32)


def _set(qk, kv, qk2, blk, fold=False):
    week3_patch.ENABLED_QK, week3_patch.ENABLED_KV = qk, kv
    week3_patch.ENABLED_QK2, week3_patch.ENABLED_BLOCK = qk2, blk
    week3_patch.ENABLED_FOLD = fold


ALL = (True, True, True, True, True)


def decode(model, n, fuse, qk=False, kv=False, qk2=False, blk=False, fold=False):
    week2.FUSE_NORM_ROPE = fuse
    _set(qk, kv, qk2, blk, fold)
    cache = model.create_kv_cache()
    y = model(ids[None], 0, cache, logits_to_keep=1)[:, -1, :].argmax(-1).astype(mx.int32)
    mx.eval(y)
    week2._RMS_RECENT.clear()
    off = ids.size
    toks, left = [], []
    NCALLS[0] = 0
    for _ in range(n):
        y = model(y[None], off, cache, logits_to_keep=1)[:, -1, :].argmax(-1).astype(mx.int32)
        left.append(len(week2._RMS_RECENT))
        mx.eval(y)
        toks.append(int(y.item()))
        off += 1
    calls = NCALLS[0] / n
    for layer in cache:
        layer.release()
    week2.FUSE_NORM_ROPE = True
    _set(*ALL)
    return toks, left, calls


def logits(model, flags, n=3):
    """Full prefill logits (logits_to_keep=None) then n decode-step logits (logits_to_keep=1), evaluated."""
    _set(*flags)
    cache = model.create_kv_cache()
    out = [model(ids[None], 0, cache)]
    y = out[0][:, -1, :].argmax(-1).astype(mx.int32)
    off = ids.size
    for _ in range(n):
        lg = model(y[None], off, cache, logits_to_keep=1)
        out.append(lg)
        y = lg[:, -1, :].argmax(-1).astype(mx.int32)
        off += 1
        mx.eval(y)
    mx.eval(out)
    for layer in cache:
        layer.release()
    _set(*ALL)
    return out


def same(a, b):
    return len(a) == len(b) and all(bool(mx.array_equal(u, v).item()) for u, v in zip(a, b))


def step_graph(model, qk2, blk, fold=False):
    """Primitive-node counts of one unevaluated decode-step graph (mx.export_to_dot), all other parts on."""
    import collections, re, tempfile
    _set(True, True, qk2, blk, fold)
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
    _set(*ALL)
    # primitive nodes are drawn as boxes; array nodes as ovals
    prims = [m.group(1) for m in re.finditer(r'label\s*=\s*"([^"]+)"[^\]]*shape\s*=\s*"?(?:rectangle|box)', txt)]
    return collections.Counter(prims), txt.count("->")


for label, kw in (("paged", dict(week=3, enable_paged_attention=True)),
                  ("dense", dict(week=3, enable_paged_attention=False))):
    model = models.dispatch_model(NAME, m, **kw)
    tf, lf, cf = decode(model, 64, True)
    tu, lu, cu = decode(model, 64, False)
    check(f"model {label} tokens fused==unfused (patch off)", tf == tu, f"(64 greedy steps; first 8 {tf[:8]})")
    check(f"model {label} fusion fires every q/k pair (patch off)", max(lf) == 0 and min(lu) >= 72,
          f"(side-table entries left per step: fused {max(lf)}, unfused {min(lu)})")
    week2._RMS_RECENT.clear()
    tq, lq, cq = decode(model, 64, True, qk=True)
    tb, lb, cb = decode(model, 64, True, qk=True, kv=True)
    n0 = week3_patch.COUNTS["attn_pair"]
    t2, l2, c2 = decode(model, 64, True, qk=True, kv=True, qk2=True)
    n2 = week3_patch.COUNTS["attn_pair"] - n0
    b0 = week3_patch.COUNTS["blk_fused"]
    ta, la, ca = decode(model, 64, True, qk=True, kv=True, qk2=True, blk=True)
    nb = week3_patch.COUNTS["blk_fused"] - b0
    c0 = dict(week3_patch.COUNTS)
    check(f"model {label} tokens patched(all: qk+kv+qk2+blk)==unpatched", ta == tf)
    check(f"model {label} block add+norm fires every layer", nb >= 36 * 64 and ca == c2,
          f"(fused blocks {nb} over 64 steps + prefill; calls/step {ca:.1f}, unchanged from {c2:.1f} as the "
          f"add+norm kernel replaces the rms_norm call)")
    check(f"model {label} tokens patched(qk)==unpatched", tq == tf)
    check(f"model {label} tokens patched(qk+kv)==unpatched", tb == tf)
    check(f"model {label} tokens patched(qk+kv+qk2)==unpatched", t2 == tf)
    check(f"model {label} no norm left pending (patched)", max(lq) == 0 and max(lb) == 0 and max(l2) == 0
          and max(la) == 0)
    want_kv = 36  # dense week 3 also appends through TinyKvPagedPool.write_page_slice (then gathers)
    check(f"model {label} calls/step", cf - cq == 72 and cq - cb == want_kv and cb - c2 == 36,
          f"(unpatched {cf:.1f}, qk {cq:.1f}, qk+kv {cb:.1f}, +qk2 {c2:.1f}; expect -72, -{want_kv}, -36) counts {c0}")
    check(f"model {label} q/k pair fires every layer", n2 >= 36 * 64, f"(pair calls {n2} over 64 steps + prefill)")
    f0, ff0 = week3_patch.COUNTS["fold_fused"], week3_patch.COUNTS["fold_final"]
    tz, lz, cz = decode(model, 64, True, *ALL[:4], fold=True)
    nf, nff = week3_patch.COUNTS["fold_fused"] - f0, week3_patch.COUNTS["fold_final"] - ff0
    check(f"model {label} tokens patched(all + fold)==unpatched", tz == tf, f"(first 8 {tz[:8]})")
    check(f"model {label} fold fires every layer boundary + final norm", nf >= 35 * 64 and nff >= 64
          and week3_patch.COUNTS["fold_layer_fallback"] == 0 and max(lz) == 0,
          f"(input folds {nf}, final folds {nff} over 64 steps + prefill; layer fallbacks "
          f"{week3_patch.COUNTS['fold_layer_fallback']})")
    check(f"model {label} calls/step (fold)", cz == ca, f"({ca:.1f} -> {cz:.1f}, expect unchanged)")
    lg_fold = logits(model, ALL)
    lg_blk = logits(model, ALL[:4] + (False,))
    lg_off = logits(model, (False,) * 5)
    check(f"model {label} logits bitwise fold == fold off", same(lg_fold, lg_blk),
          f"(prefill [1,{ids.size},vocab] + 3 decode steps)")
    check(f"model {label} logits bitwise fold == unpatched", same(lg_fold, lg_off))
    try:
        g_off, e_off = step_graph(model, True, False)
        g_on, e_on = step_graph(model, True, True)
        g_fd, e_fd = step_graph(model, True, True, True)
        n_off, n_on, n_fd = sum(g_off.values()), sum(g_on.values()), sum(g_fd.values())
        add_off, add_on = g_off.get("Add", 0), g_on.get("Add", 0)
        if n_off == 0:
            print(f"FUSE INFO model {label} graph: no primitive nodes parsed from export_to_dot", flush=True)
        else:
            check(f"model {label} graph nodes/step (blk)", add_off - add_on == 36 and n_off - n_on >= 36,
                  f"(primitives {n_off} -> {n_on}, Add {add_off} -> {add_on}, edges {e_off} -> {e_on}; "
                  f"top off {g_off.most_common(8)}; top on {g_on.most_common(8)})")
            delta = {k: g_on.get(k, 0) - g_fd.get(k, 0) for k in set(g_on) | set(g_fd)}
            delta = {k: v for k, v in sorted(delta.items()) if v}
            check(f"model {label} graph nodes/step (fold)", g_on.get("Add", 0) - g_fd.get("Add", 0) == 36
                  and n_on - n_fd >= 36,
                  f"(primitives {n_on} -> {n_fd}, edges {e_on} -> {e_fd}; removed per kind {delta})")
    except Exception as e:  # export_to_dot missing in this build: report, do not gate
        print(f"FUSE INFO model {label} graph count unavailable: {e!r}", flush=True)
    week2._RMS_RECENT.clear()

print("FUSE_ALL_PASS" if ok else "FUSE_SOME_FAIL", flush=True)
sys.exit(0 if ok else 1)
