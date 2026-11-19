#!/usr/bin/env bash
# Single-answer GRPO on IFEval. Usage: ./script/train_ifeval_grpo.sh [key=value ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${CONF:-$ROOT/script/train_ifeval_grpo.conf}"
source "$CONF"
PASS=()
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "Expected key=value: $kv"; exit 1; }
  k="${kv%%=*}"
  [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "Invalid key: $k"; exit 1; }
  declare "$kv"
  [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] || PASS+=("$kv")
done
IFS=',' read -ra TG <<<"$TRAIN_GPUS"
IFS=',' read -ra SG <<<"$SAMPLER_GPUS"
NPROC=${#TG[@]}
source "$ROOT/script/training_runtime.sh"
arl_training_gpu_layout "$TRAIN_GPUS" "$SAMPLER_GPUS" "${COLOCATE:-0}" || exit 1
(( batch_size >= NPROC )) || { echo "batch_size < training ranks"; exit 1; }
(( SAMPLER_DP > 0 && ${#SG[@]} % SAMPLER_DP == 0 )) || { echo "Invalid SAMPLER_DP"; exit 1; }
log_path="$(date +"$log_path")"
if [[ "$log_path" != /* ]]; then log_path="$ROOT/$log_path"; fi
if [[ -e "$log_path" ]]; then
  base="${log_path%_v*}"; n=1
  while [[ -e "${base}_v$n" ]]; do n=$((n + 1)); done
  log_path="${base}_v$n"
fi
ARGS=("env=instruction_following.ifeval" "model_name=$POLICY" "log_path=$log_path")
for k in harness eval_harness train_set test_set test_fraction test_limit task_limit env_max_turns eval_max_concurrency \
    ifeval_reward_mode ifeval_split_seed ifeval_python ifeval_upstream ifeval_timeout batch_size group_size extra_samples num_epochs shuffle_seed \
    temperature top_p top_k max_tokens prompt_budget_tokens sampling_max_model_len \
    truncated_turn learning_rate warmup_steps warmup_lr_scale grad_clip num_substeps clip_low clip_high \
    adv_estimator group_adv_std group_adv_std_floor whiten_advantages whiten_std length_weighted_baseline \
    mask_positive_incorrect mask_negative_correct think_cost_lambda think_cost_free_tokens think_budget think_force_text sampling_replay router_replay save_optimizer \
    eval_every eval_at_start eval_num_samples save_every load_checkpoint_path resume_optimizer start_batch \
    reasoning_effort enable_thinking wandb_project; do
  [[ -z "${!k:-}" ]] || ARGS+=("$k=${!k}")
done
ARGS+=("eval_temperature=$temperature" "eval_top_p=$top_p" "eval_top_k=$top_k")
ARGS+=("wandb_name=$(basename "$log_path")" "${PASS[@]}")
ENV=("PYTHONPATH=$ROOT" "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"
  "CUDA_VISIBLE_DEVICES=$ARL_TRAIN_VISIBLE_GPUS" "KERNEL_RL_SAMPLER_GPUS=$SAMPLER_GPUS"
  "KERNEL_RL_SAMPLER_DP=$SAMPLER_DP" "KERNEL_RL_SAMPLING_GPU_UTIL=$SAMPLING_UTIL"
  "KERNEL_RL_ZERO_OFFLOAD=$ZERO_OFFLOAD" "KERNEL_RL_MICRO_BSZ=1"
  "KERNEL_RL_PACK_TOKENS=${PACK_TOKENS:-0}" "KERNEL_RL_HEAD_CHUNKS=${HEAD_CHUNKS:-0}"
  "KERNEL_RL_REBALANCE=${REBALANCE:-1}"
  "KERNEL_RL_COLOCATE=${COLOCATE:-0}" "KERNEL_RL_COLOCATE_HEADROOM_GB=${COLOCATE_HEADROOM_GB:-8}")
for k in NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF DEEPSPEED_TIMEOUT KERNEL_RL_FIXED_RMSNORM KERNEL_RL_WANDB_PROXY; do
  [[ -z "${!k:-}" ]] || ENV+=("$k=${!k}")
done
[[ ! -v KERNEL_RL_WANDB_PROXY ]] || ENV+=("KERNEL_RL_WANDB_PROXY=$KERNEL_RL_WANDB_PROXY")

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/train_ifeval_grpo_$(date +%Y%m%d_%H%M%S)_$$.log"
LAUNCH=(python3 -m arl.train)
if (( NPROC > 1 )); then LAUNCH=(torchrun "--nproc_per_node=$NPROC" "--master_port=$PORT" -m arl.train); fi
echo ">>> IFEval GRPO: ranks=$NPROC reward=$ifeval_reward_mode log=$LOG output=$log_path"
cd "$ROOT"
env "${ENV[@]}" "${LAUNCH[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG"
