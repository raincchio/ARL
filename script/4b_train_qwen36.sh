#!/usr/bin/env bash
# ARL SWE10 Qwen3.6 preset; accepts ARM-style key=value.
# Evaluation trajectories per task: eval_samples=8 (default 8), or --eval-samples 8.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/script/train_swe10_aligned.sh" "$@"
