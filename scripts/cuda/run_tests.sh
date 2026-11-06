#!/usr/bin/env bash
# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file; see NOTICE.
#
# Run the course's reference-solution tests (tests_refsol/) on the CUDA backend, one file at a time,
# and print a per-file result plus a total. Activate the venv from scripts/cuda/setup.sh first.
# usage: bash scripts/cuda/run_tests.sh [pytest file globs...]
#   default: weeks 1-3 and week 4 day 1, the set the CUDA port is checked against
set -uo pipefail
cd "$(dirname "$0")/../.."
export TINY_LLM_BACKEND=cuda
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import mlx.core as mx
from extensions_ref import tiny_llm_ext_ref
import tiny_llm_cuda_kernels
print("mlx", mx.__version__, "| gpu:", mx.is_available(mx.gpu), "| backend:", tiny_llm_ext_ref.rms_norm.__module__)
PY
if [ $# -eq 0 ]; then
  set -- tests_refsol/test_week_1_day_*.py tests_refsol/test_week_2_day_*.py \
    tests_refsol/test_week_2_reference_runtime.py tests_refsol/test_week_3_day_*.py tests_refsol/test_week_4_day_1.py
fi
pass=0; fail=0; bad_files=0
for f in "$@"; do
  out=$(timeout 1800 python -m pytest -q -p no:cacheprovider "$f" 2>&1); rc=$?
  last=$(printf '%s\n' "$out" | tail -1)
  p=$(printf '%s' "$last" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
  n=$(printf '%s' "$last" | grep -oE '[0-9]+ (failed|error)' | grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null || echo 0)
  pass=$((pass + ${p:-0})); fail=$((fail + ${n:-0}))
  [ $rc -eq 0 ] || bad_files=$((bad_files + 1))
  echo "RESULT $(basename "$f"): rc=$rc $last"
done
echo "TOTAL passed=$pass failed=$fail files_with_failures=$bad_files"
[ $bad_files -eq 0 ]
