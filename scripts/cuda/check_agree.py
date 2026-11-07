# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""Long greedy-agreement check for promoting the opt-in week-3 `mlp` patch (docs/cuda/DESIGN.md section 4).

Run from the repo root with TINY_LLM_BACKEND=cuda TL_PATCH_WEEK3=all PYTHONPATH=src:. python scripts/cuda/check_agree.py. week3_patch.ENABLED_MLP is
toggled per call, so mlp off is exactly the bitwise default `1` and mlp on is `all`.

For each prompt and each of paged / dense: two caches decode in lockstep for STEPS steps. Reference (`1`)
runs greedy; the candidate (`all`) is teacher-forced with the reference's tokens, so up to the first
argmax mismatch both runs are the same free-running greedy decode, and the step of that first mismatch is
the first-divergence step. Teacher forcing then continues, which gives next-token match over every step,
the max |logit diff| before divergence and over all steps, and the NLL of the reference continuation under
both. A second pass feeds a fixed English passage token by token through the decode path (the mlp patch
only fires for <= 8 rows, so a prefill perplexity would not exercise it) and compares perplexity.

Prints AGREE lines and AGREE_PROMOTE (every prompt agrees >= MIN_AGREE steps, paged and dense) or
AGREE_KEEP_OPTIN.
"""

import inspect
import math
import os
import sys
import time

import mlx.core as mx

from tiny_llm_cuda_kernels import week3_patch
from mlx_lm import load
from tiny_llm_ref import models

STEPS = int(os.environ.get("AGREE_STEPS", "512"))
MIN_AGREE = int(os.environ.get("AGREE_MIN", "256"))
NAME = "Qwen/Qwen3-4B-MLX-4bit"
PROMPTS = [
    "Write a short story about a robot learning to cook.",
    "Explain in three sentences what a KV cache does in LLM inference.",
    "List the planets of the solar system and one fact about each.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]
PASSAGE = (
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined "
    "movable metal type, oil-based ink and a wooden screw press adapted from those used to make wine. Before "
    "his workshop in Mainz produced its first books, copies were made by hand, usually by scribes working in "
    "monasteries, and a single Bible could take years to finish. The new method cut that time to weeks. "
    "Within fifty years presses were running in more than two hundred cities across Europe, and millions of "
    "books had been printed. Prices fell, literacy rose, and ideas moved faster than any ruler or church "
    "could control. Scholars could compare identical copies of a text, which made errors easier to find and "
    "correct. Pamphlets spread news of discoveries, wars and religious disputes. Many historians argue that "
    "the Reformation, the Scientific Revolution and the rise of national languages all depended on cheap "
    "printed material. The basic design of the press changed little for three hundred years, until steam "
    "power and rotary cylinders in the nineteenth century made it possible to print thousands of sheets an "
    "hour, which in turn made daily newspapers affordable for ordinary readers."
)

if not (week3_patch.patch_status().get("Qwen3MLP.__call__") == "patched"):
    print(f"AGREE FAIL mlp patch not installed {week3_patch.patch_status()}", flush=True)
    sys.exit(1)

m, tok = load(NAME)


def fwd(model, cache, tokens, off, mlp):
    week3_patch.ENABLED_MLP = mlp
    lg = model(tokens[None], off, cache, logits_to_keep=1)[:, -1, :].astype(mx.float32)
    week3_patch.ENABLED_MLP = True
    return lg


def nll(lg, t):
    return (mx.logsumexp(lg, axis=-1) - lg[:, t]).squeeze()


def new_cache(model, n):
    if "capacity" in inspect.signature(model.create_kv_cache).parameters:
        cap = n + 1 if getattr(model, "use_bounded_kv_capacity", False) else None
        return model.create_kv_cache(capacity=cap)
    return model.create_kv_cache()


def release(cache):
    for layer in cache:
        layer.release()


def lockstep(model, ids, steps):
    """Reference greedy (mlp off) + candidate teacher-forced (mlp on), one shared token stream."""
    ca, cb = new_cache(model, ids.size + steps), new_cache(model, ids.size + steps)
    la, lb = fwd(model, ca, ids, 0, False), fwd(model, cb, ids, 0, True)
    first_div, match = None, 0
    max_pre, max_all, max_ref = 0.0, 0.0, 0.0
    at_all, max_top = -1, 0.0  # step of max_all; max |d log p| over tokens with reference p > 1e-4
    nll_a = nll_b = 0.0
    off = ids.size
    for i in range(steps):
        ya, yb = la.argmax(-1), lb.argmax(-1)
        d = mx.abs(la - lb).max()
        lpa, lpb = la - mx.logsumexp(la, axis=-1, keepdims=True), lb - mx.logsumexp(lb, axis=-1, keepdims=True)
        dt = mx.where(lpa > -9.21, mx.abs(lpa - lpb), 0.0).max()
        mx.eval(ya, yb, d, dt)
        ta, tb, dv = int(ya.item()), int(yb.item()), d.item()
        max_top = max(max_top, dt.item())
        if dv > max_all:
            max_all, at_all = dv, i
        max_ref = max(max_ref, mx.abs(la).max().item())
        if ta == tb:
            match += 1
        elif first_div is None:
            first_div = i
        if first_div is None:
            max_pre = max(max_pre, dv)
        if i > 0:
            pa, pb = nll(la, ta), nll(lb, ta)
            mx.eval(pa, pb)
            nll_a += pa.item()
            nll_b += pb.item()
        nxt = mx.array([ta], dtype=mx.int32)
        la, lb = fwd(model, ca, nxt, off, False), fwd(model, cb, nxt, off, True)
        off += 1
    mx.eval(la, lb)
    release(ca)
    release(cb)
    return dict(first_div=steps if first_div is None else first_div, match=match, max_pre=max_pre,
                max_all=max_all, at_all=at_all, max_top=max_top, max_ref=max_ref, nll_a=nll_a / max(steps - 1, 1), nll_b=nll_b / max(steps - 1, 1))


def passage_ppl(model, ids):
    """Teacher-forced decode over a fixed passage: token 0 as a 1-token prefill, then one token per step."""
    n = ids.size
    ca, cb = new_cache(model, n), new_cache(model, n)
    tot_a = tot_b = 0.0
    match = 0
    max_d = 0.0
    for i in range(n - 1):
        x = ids[i:i + 1]
        la, lb = fwd(model, ca, x, i, False), fwd(model, cb, x, i, True)
        t = int(ids[i + 1].item())
        pa, pb, d = nll(la, t), nll(lb, t), mx.abs(la - lb).max()
        same = mx.equal(la.argmax(-1), lb.argmax(-1)).squeeze()
        mx.eval(pa, pb, d, same)
        tot_a += pa.item()
        tot_b += pb.item()
        max_d = max(max_d, d.item())
        match += bool(same.item())
    release(ca)
    release(cb)
    k = n - 1
    return math.exp(tot_a / k), math.exp(tot_b / k), match, k, max_d


ok_all = True
for label, kw in (("paged", dict(week=3, enable_paged_attention=True)),
                  ("dense", dict(week=3, enable_paged_attention=False))):
    model = models.dispatch_model(NAME, m, **kw)
    f0 = week3_patch.COUNTS["mlp_fused"]
    for pi, prompt in enumerate(PROMPTS):
        ids = mx.array(tok.encode(prompt), dtype=mx.int32)
        t = time.time()
        r = lockstep(model, ids, STEPS)
        ok = r["first_div"] >= MIN_AGREE
        ok_all &= ok
        print(f"AGREE {label} prompt{pi} first_divergence_step {r['first_div']}/{STEPS} "
              f"{'>=' if ok else '<'} {MIN_AGREE}; teacher-forced next-token match {r['match']}/{STEPS}; "
              f"max|dlogit| before div {r['max_pre']:.3f}, all steps {r['max_all']:.3f} at step {r['at_all']} (max|logit| {r['max_ref']:.1f}); "
              f"max|d log p| over tokens with p > 1e-4 {r['max_top']:.3f}; "
              f"mean NLL of ref continuation `1` {r['nll_a']:.4f} vs `all` {r['nll_b']:.4f} ({time.time() - t:.0f}s)",
              flush=True)
    ids = mx.array(tok.encode(PASSAGE), dtype=mx.int32)
    pa, pb, match, k, max_d = passage_ppl(model, ids)
    print(f"AGREE {label} passage ({k} decode steps) perplexity `1` {pa:.4f} vs `all` {pb:.4f} "
          f"(rel {100 * (pb - pa) / pa:+.2f}%); argmax match {match}/{k}; max|dlogit| {max_d:.3f}", flush=True)
    print(f"AGREE {label} mlp fused calls {week3_patch.COUNTS['mlp_fused'] - f0}", flush=True)

print("AGREE_PROMOTE" if ok_all else "AGREE_KEEP_OPTIN", flush=True)
