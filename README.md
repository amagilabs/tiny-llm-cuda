<!-- Copyright 2026 AmagiLabs. SPDX-License-Identifier: Apache-2.0. Part of tiny-llm-cuda; see NOTICE. -->
# tiny-llm-cuda

A fork of [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) that runs the course's reference solution on an
NVIDIA GPU. The course is written for Apple silicon: MLX plus a set of custom Metal kernels. This fork adds an
optional CUDA backend for those kernels, so weeks 1 to 3 also run on Linux with an NVIDIA card.

All credit for the course, its code and its book goes to Alex Chi Z. ([@skyzh](https://github.com/skyzh)) and
the tiny-llm contributors. Read the book at **https://skyzh.github.io/tiny-llm/**. This repo does not copy or
change it. The upstream README is kept as [README.upstream.md](README.upstream.md).

## What changed

- `src/tiny_llm_cuda_kernels/`: CUDA versions of the 12 Metal kernels the reference solution calls (RMSNorm,
  RoPE, SwiGLU, 4-bit quantized matmul and embedding, decode and prefill attention, the paged KV cache and
  paged attention). They are CUDA source strings compiled at runtime through MLX's `mx.fast.cuda_kernel`, so
  there is no C++ build step.
- `src/extensions_ref/tiny_llm_ext_ref/__init__.py`: one opt-in switch. With `TINY_LLM_BACKEND=cuda` the
  reference extension loads the CUDA kernels. Without it, the file behaves exactly as upstream, so Mac use is
  unchanged.
- `scripts/cuda/`: setup, test runner, decode benchmark and the correctness checks for the week-3 fusions.
- Everything else, including the rest of `src/tiny_llm_ref` and the learner's starter code, is upstream code at
  commit `fa22518`. [NOTICE](NOTICE) lists every added and modified file.

How the kernels map from Metal and what the week-3 call-site patch does: [docs/cuda/DESIGN.md](docs/cuda/DESIGN.md).

## Setup (Linux, NVIDIA GPU)

Tested on one RTX 3090 (24 GB), driver 595.99.02 (CUDA 13.2), Python 3.12.

```bash
git clone https://github.com/amagilabs/tiny-llm-cuda.git
cd tiny-llm-cuda
bash scripts/cuda/setup.sh            # venv in .venv-cuda + Qwen3 0.6B / 1.7B / 4B MLX 4-bit weights
. .venv-cuda/bin/activate
```

`setup.sh` installs `requirements-cuda.txt`: `mlx[cuda13]==0.32.4`, `mlx-lm==0.31.3`, numpy, pytest and
huggingface_hub. If your driver only supports CUDA 12, change `mlx[cuda13]` to `mlx[cuda12]` there first.

## Use it

Turn the backend on with one environment variable:

```bash
export TINY_LLM_BACKEND=cuda
export PYTHONPATH=src:.

python main.py --model qwen3-4b --solution ref --loader week3 --prompt "What does a KV cache do?"
bash scripts/cuda/run_tests.sh                       # weeks 1-3 + week 4 day 1 reference tests
python scripts/cuda/bench_decode.py                  # decode tok/s vs stock mlx-lm
```

Unset `TINY_LLM_BACKEND` (or set it to anything else) and the upstream Metal path is used again.

Other switches:

| variable | values | effect |
|---|---|---|
| `TINY_LLM_BACKEND` | `cuda` / unset | CUDA kernels / upstream Metal extension |
| `TL_PATCH_WEEK3` | `1` (default), `all`, `0`, or a list like `qk,kv` | week-3 call-site patch: fewer kernel calls per decode step. `1` is bitwise identical to unpatched; `all` adds a faster MLP path that is not (about 1% on logits) |
| `TL_FUSE_NORM_ROPE` | `1` (default) / `0` | small RMSNorm-into-RoPE fusion |

## Test results (RTX 3090, 2026-11-06)

`bash scripts/cuda/run_tests.sh`, reference-solution tests from `tests_refsol/`:

| part | tests | result |
|---|---|---|
| Week 1 (days 1-7) | 250 | all pass |
| Week 2 (days 1-5 + reference runtime) | 54 | all pass |
| Week 3 (days 1-7) | 152 | all pass |
| Week 4 day 1 | 16 | all pass |
| **Total** | **472** | **472 pass, 0 fail** |

Weeks 1-3 cover every custom kernel. Week 4 days 2-8 and the other helper test files pass as well.
Six tests still fail on Linux: 1 in `test_week_4_day_9.py`, 4 in `test_week_4_starter_sync.py` and 1 in
`test_extension_interface_sync.py`. All six import the learner's own compiled extension
(`src/extensions/tiny_llm_ext`), which is the course exercise and is not part of this port.
`scripts/cuda/check_fusion.py` passes every check: tokens and logits are identical with the week-3 patch on and off.

Decode speed, Qwen3-4B MLX 4-bit, greedy, 256 tokens after the prompt (`scripts/cuda/bench_decode.py`, one run):

| path | tok/s |
|---|---|
| reference week 3, paged attention, CUDA kernels, `mx.eval` every step | 101.2 |
| same, pipelined (`mx.async_eval` one step ahead, as mlx-lm does) | 126.6 |
| reference week 2, CUDA kernels | 40.4 |
| stock mlx-lm `stream_generate`, same GPU, same run | 151.8 |

Pipelined, the course model runs at about 83% of stock mlx-lm. The remaining gap is mostly CPU-side graph
building, not the kernels (see [DESIGN.md](docs/cuda/DESIGN.md), section 5).

## Scope

This is a learning port, not a serving engine. It runs on one GPU and has no tensor parallelism, no
OpenAI-compatible server and no multi-GPU support. For production serving on NVIDIA, use vLLM or llama.cpp.

## License

Code: Apache License 2.0, same as upstream ([LICENSE](LICENSE), [NOTICE](NOTICE)). The book in `book/` is
upstream's, under CC BY-NC-SA 4.0, and is unchanged.
