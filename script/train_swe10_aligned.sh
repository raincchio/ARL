#!/usr/bin/env bash
# SWE-compatible subset of the user's ARM Qwen3.6 GRPO recipe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KERNEL_RL_ZERO3_REDUCE_BUCKET="${KERNEL_RL_ZERO3_REDUCE_BUCKET:-100000000}"
export KERNEL_RL_FREEZE_ROUTER="${KERNEL_RL_FREEZE_ROUTER:-0}"
export KERNEL_RL_FREEZE_SHARED_EXPERT="${KERNEL_RL_FREEZE_SHARED_EXPERT:-1}"
exec "$ROOT/script/train_swe10.sh" \
  --model /models/Qwen/Qwen3.6-35B-A3B \
  --harness mini_swe_agent --eval-async \
  --colocate --sampling-util 0.7 --sampler-dp 4 \
  --sampling-replay --router-replay \
  --batch-size 8 --train-group-size 8 --epochs 10 \
  --max-turns 250 \
  --max-tokens 40960 \
  --prefill-tokens 1056 --prefill-batch-tokens 2048 \
  --enable-prefix-caching --mamba-cache-mode align \
  --pack-tokens 32768 --head-chunks 8 \
  --adv-estimator grpo --think-budget 32768 \
  --warmup-steps 30 --warmup-lr-scale 0.5 \
  --think-cost-lambda 0.001 --think-cost-free-tokens 8192 \
  --format-reward-weight "${FORMAT_REWARD_WEIGHT:-0.05}" \
  --mask-positive-incorrect --mask-negative-correct \
  --no-save-optimizer --no-save-trajectories --no-save-best-solutions \
  "$@"
