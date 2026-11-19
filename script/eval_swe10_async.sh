#!/usr/bin/env bash
# ARM-style evaluator queue, run on the evaluation machine.
# Default: 8 GPUs/job, TP=4 (from spec or shared defaults), utilization=0.85.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARGS=()
case "${1:-}" in
  add|rm|redo|status|log|start|stop|restart|_serve|-h|--help) ;;
  '') set -- --help;;
  *) ARGS+=(add);;
esac
for arg in "$@"; do
  case "$arg" in
    GPUS=*) ARGS+=(--gpus "${arg#*=}");;
    NGPUS=*) ARGS+=(--ngpus "${arg#*=}");;
    SAMPLER_DP=*) ARGS+=(--sampler-dp "${arg#*=}");;
    SAMPLER_TP=*) ARGS+=(--sampler-tp "${arg#*=}");;
    SAMPLING_UTIL=*) ARGS+=(--util "${arg#*=}");;
    poll=*) ARGS+=(--poll "${arg#*=}");;
    retries=*) ARGS+=(--retries "${arg#*=}");;
    *=*) echo "Unknown evaluator option: $arg" >&2; exit 2;;
    *) ARGS+=("$arg");;
  esac
done
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m arl.experiments.eval_service "${ARGS[@]}"
