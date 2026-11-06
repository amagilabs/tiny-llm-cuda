# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
"""Decode-speed bench: the course's reference model on the CUDA kernels vs stock mlx-lm, same GPU, same run.

usage (repo root, CUDA venv active):
    TINY_LLM_BACKEND=cuda PYTHONPATH=src:. python scripts/cuda/bench_decode.py [hf-model-id]
default model Qwen/Qwen3-4B-MLX-4bit. Greedy, 256 decode steps after the prompt prefill.
"sync loop" = mx.eval every step; "pipelined" = mx.async_eval of step n+1 before waiting on step n, as
mlx-lm's stream_generate does. Pipelined runs also check that their tokens equal the sync loop's.
"""
import sys
import inspect, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from tiny_llm_ref import models
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-MLX-4bit"
PROMPT = "Write a short story about a robot learning to cook."
m, tok = load(NAME)
ids = mx.array(tok.encode(PROMPT), dtype=mx.int32)

def tiny(label, pipelined=False, **kw):
    # pipelined=False: mx.eval every step (the original loop; all before/after numbers use this).
    # pipelined=True: mlx-lm stream_generate style. Build step n+1 and async_eval it, then wait for step n,
    # so Python graph build overlaps GPU work. Same model, kernels and cache; greedy tokens checked
    # against the sync loop.
    model = models.dispatch_model(NAME, m, **kw)
    def run(n, pipe):
        if "capacity" in inspect.signature(model.create_kv_cache).parameters:
            cap = ids.size + n + 1 if getattr(model, "use_bounded_kv_capacity", False) else None
            cache = model.create_kv_cache(capacity=cap)
        else:
            cache = model.create_kv_cache()
        y = model(ids[None], 0, cache, logits_to_keep=1)[:, -1, :].argmax(-1).astype(mx.int32)
        mx.eval(y)
        toks = []
        t = time.time()
        off = ids.size
        for _ in range(n):
            nxt = model(y[None], off, cache, logits_to_keep=1)[:, -1, :].argmax(-1).astype(mx.int32)
            if pipe:
                mx.async_eval(nxt)
                mx.eval(y)
            else:
                mx.eval(nxt)
            toks.append(nxt)
            y = nxt
            off += 1
        mx.eval(y)
        dt = time.time() - t
        for layer in cache:
            layer.release()
        return dt, [int(v.item()) for v in toks]
    run(8, pipelined)
    n = 256
    dt, toks = run(n, pipelined)
    extra = ""
    if pipelined:
        _, ref = run(n, False)
        extra = f"; greedy tokens match sync loop: {toks == ref}"
    print(f"BENCH {label}: {n/dt:.1f} tok/s decode ({n} tok, {dt:.2f}s){extra}", flush=True)

tiny("tiny-llm ref week3 paged (CUDA kernels)", week=3, enable_paged_attention=True)
tiny("tiny-llm ref week3 dense (CUDA kernels)", week=3, enable_paged_attention=False)
tiny("tiny-llm ref week2 (CUDA kernels)", week=2)
tiny("tiny-llm ref week3 paged (CUDA kernels) PIPELINED async_eval loop", pipelined=True, week=3,
     enable_paged_attention=True)
tiny("tiny-llm ref week3 dense (CUDA kernels) PIPELINED async_eval loop", pipelined=True, week=3,
     enable_paged_attention=False)

def stock_sync(n):
    # Stock mlx-lm model in the same sync loop as tiny() (mx.eval every step), for a like-for-like reference.
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(m)
    y = m(ids[None], cache=cache)[:, -1, :].argmax(-1).astype(mx.int32)
    mx.eval(y)
    t = time.time()
    for _ in range(n):
        y = m(y[None], cache=cache)[:, -1, :].argmax(-1).astype(mx.int32)
        mx.eval(y)
    return time.time() - t
stock_sync(8)
dt = stock_sync(256)
print(f"BENCH stock mlx-lm model, same sync loop: {256/dt:.1f} tok/s decode (256 tok, {dt:.2f}s)", flush=True)

def run_mlx(n):
    last = None
    t = time.time()
    for r in stream_generate(m, tok, PROMPT, max_tokens=n):
        last = r
    return time.time() - t, last
run_mlx(8)
dt, last = run_mlx(256)
print(f"BENCH stock mlx-lm: {256/dt:.1f} tok/s end-to-end (256 tok, {dt:.2f}s); "
      f"mlx-lm generation_tps {last.generation_tps:.1f}", flush=True)
