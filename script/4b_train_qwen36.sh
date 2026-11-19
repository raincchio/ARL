#!/usr/bin/env bash
# ARL SWE10 Qwen3.6 preset; accepts ARM-style key=value.
# Trajectories per task: train_group_size=8 and eval_group_size=8 by default.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/script/train_swe10_aligned.sh" "$@"
