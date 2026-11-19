#!/usr/bin/env bash
# ARL agentic GRPO × CudaEnv 启动器。
#
#   ./script/train_cuda_grpo.sh                               # 全用 train_cuda_grpo.conf
#   ./script/train_cuda_grpo.sh POLICY=/path key=value ...    # 覆盖 (大写 = 启动器变量, 小写 = GRPOConfig 键)
#   ./script/train_cuda_grpo.sh task_limit=16 batch_size=8 group_size=4 env_max_turns=4 \
#       max_tokens=8192 eval_every=0 save_every=0             # 冒烟
#
# 进程结构 (同 ARM 4_train_grpo.sh): torchrun 起 |TRAIN_GPUS| 个训练 rank, rank0 在
# SAMPLER_GPUS 上另起 vLLM (TP=卡数/SAMPLER_DP); 评测全在远端 CudaEval, 本机卡一张不占。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${CONF:-$ROOT/script/train_cuda_grpo.conf}"
# shellcheck disable=SC1090
source "$CONF"

PASS=()
for kv in "$@"; do
  k="${kv%%=*}"
  if [[ "$kv" == *=* && "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
    declare "$kv"          # 启动器变量 (POLICY / TRAIN_GPUS / ...)
  elif [[ "$kv" == *=* ]]; then
    declare "$kv"; PASS+=("$kv")
  else
    PASS+=("$kv")
  fi
done

log_path="$(date +"$log_path")"
if [ -e "$ROOT/$log_path" ] || [ -e "$log_path" ]; then   # 自动递增 _vN
  base="${log_path%_v*}"; n=1
  while [ -e "$ROOT/${base}_v$n" ]; do n=$((n + 1)); done
  log_path="${base}_v$n"
fi

IFS=',' read -ra _tg <<<"$TRAIN_GPUS"; NPROC=${#_tg[@]}
IFS=',' read -ra _sg <<<"$SAMPLER_GPUS"
source "$ROOT/script/training_runtime.sh"
arl_training_gpu_layout "$TRAIN_GPUS" "$SAMPLER_GPUS" "${COLOCATE:-0}" || exit 1
(( batch_size >= NPROC )) || { echo "batch_size=$batch_size < $NPROC ranks"; exit 1; }

# 评测服务探活 (repo 模式 + workspace 节点)
CUDAENV_ROOT="${CUDAENV_ROOT:-$(cd "$ROOT/envs/coding/CudaEnv" && pwd)}"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY PYTHONPATH="$CUDAENV_ROOT" python3 -c "
from cudaenv.client import check_health; print('>>> eval service:', check_health('$EVAL_ENDPOINT'))" \
  || { echo "评测服务 $EVAL_ENDPOINT 不可用 (需要 /repo/evaluate + ws_version>=1 的活节点)"; exit 1; }

ARGS=(
  "model_name=$POLICY" "log_path=$log_path"
  "env=coding.cuda" "train_set=$train_set" "test_fraction=$test_fraction" "test_limit=$test_limit"
  "env_max_turns=$env_max_turns" "env_enable_analyze=$env_enable_analyze"
  "env_reward_mode=$env_reward_mode" "eval_endpoint=$EVAL_ENDPOINT"
  "batch_size=$batch_size" "group_size=$group_size" "extra_samples=$extra_samples"
  "num_epochs=$num_epochs" "temperature=$temperature" "top_p=$top_p" "top_k=$top_k"
  "eval_temperature=$temperature" "eval_top_p=$top_p" "eval_top_k=$top_k"
  "max_tokens=$max_tokens"
  "prompt_budget_tokens=$prompt_budget_tokens" "truncated_turn=$truncated_turn"
  "learning_rate=$learning_rate" "grad_clip=$grad_clip" "num_substeps=$num_substeps"
  "clip_low=$clip_low" "clip_high=$clip_high"
  "eval_every=$eval_every" "eval_at_start=$eval_at_start" "eval_num_samples=$eval_num_samples"
  "save_every=$save_every"
)
[ -n "${test_set:-}" ] && ARGS+=("test_set=$test_set")
[ -n "${reasoning_effort:-}" ] && ARGS+=("reasoning_effort=$reasoning_effort")
[ -n "${wandb_project:-}" ] && ARGS+=("wandb_project=$wandb_project" "wandb_name=$(basename "$log_path")")
# Optional algorithm and resume settings also work in machine-specific conf files.
for _v in harness eval_harness warmup_steps warmup_lr_scale adv_estimator group_adv_std group_adv_std_floor \
    whiten_advantages whiten_std length_weighted_baseline mask_positive_incorrect \
    mask_negative_correct think_cost_lambda think_cost_free_tokens think_budget think_force_text sampling_replay router_replay save_optimizer resume_optimizer load_checkpoint_path start_batch; do
  [ -n "${!_v:-}" ] && ARGS+=("$_v=${!_v}")
done
# 命令行里的小写键放最后 (chz 后者覆盖前者)
ARGS+=("${PASS[@]}")

if [ "$NPROC" -gt 1 ]; then
  LAUNCH=(torchrun "--nproc_per_node=$NPROC" "--master_port=$PORT" -m arl.train)
else
  LAUNCH=(python3 -m arl.train)
fi
ENV=(
  "PYTHONPATH=$ROOT:$CUDAENV_ROOT" "CUDAENV_ROOT=$CUDAENV_ROOT"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"
  "CUDA_VISIBLE_DEVICES=$ARL_TRAIN_VISIBLE_GPUS"
  "KERNEL_RL_SAMPLING_GPU_UTIL=$SAMPLING_UTIL"
  "KERNEL_RL_SAMPLER_GPUS=$SAMPLER_GPUS"
  "KERNEL_RL_SAMPLER_DP=$SAMPLER_DP"
  "KERNEL_RL_ZERO_OFFLOAD=$ZERO_OFFLOAD"
  "KERNEL_RL_MICRO_BSZ=1" "KERNEL_RL_PACK_TOKENS=${PACK_TOKENS:-0}" "KERNEL_RL_HEAD_CHUNKS=${HEAD_CHUNKS:-0}"
  "KERNEL_RL_REBALANCE=${REBALANCE:-1}"
  "KERNEL_RL_COLOCATE=${COLOCATE:-0}" "KERNEL_RL_COLOCATE_HEADROOM_GB=${COLOCATE_HEADROOM_GB:-8}"
  "CUDAENV_EVAL_ENDPOINT=$EVAL_ENDPOINT" "CUDAENV_CLIENT_MARGIN=$EVAL_CLIENT_MARGIN"
  "NO_PROXY=127.0.0.1,*" "no_proxy=127.0.0.1,*"
)
for _v in NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF DEEPSPEED_TIMEOUT KERNEL_RL_FIXED_RMSNORM KERNEL_RL_WANDB_PROXY; do
  [ -n "${!_v:-}" ] && ENV+=("$_v=${!_v}")
done

[[ ! -v KERNEL_RL_WANDB_PROXY ]] || ENV+=("KERNEL_RL_WANDB_PROXY=$KERNEL_RL_WANDB_PROXY")

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/train_cuda_grpo_$(date +%Y%m%d_%H%M%S)_$$.log"
_ip=$(hostname -I 2>/dev/null | awk "{print \$1}")
echo "host $(hostname) ip ${_ip:-unknown}" > "$LOG"
echo ">>> log=$LOG  log_path=$log_path  ranks=$NPROC sampler=$SAMPLER_GPUS"
( cd "$ROOT" && env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY "${ENV[@]}" "${LAUNCH[@]}" "${ARGS[@]}" ) 2>&1 | tee -a "$LOG"
