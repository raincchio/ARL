#!/usr/bin/env bash
# ARL agentic GRPO × CudaEnv, Qwen3.5-9B。启动流程和 train_cuda_grpo.sh 一样, 只把 conf 换成
# train_cuda_grpo_qwen35_9b.conf (卡位 / 显存安排见那个文件开头的注释)。
#
#   ./script/train_cuda_grpo_qwen35_9b.sh                       # 16 卡: 训练 8 (不 offload) + 采样 4×TP2
#   ./script/train_cuda_grpo_qwen35_9b.sh TRAIN_GPUS=0,1,2,3 SAMPLER_GPUS=4,5,6,7 SAMPLER_DP=2 \
#       ZERO_OFFLOAD=optimizer batch_size=8                       # 8 卡机
#   ./script/train_cuda_grpo_qwen35_9b.sh TRAIN_GPUS=0,1 SAMPLER_GPUS=2,3 SAMPLER_DP=2 ZERO_OFFLOAD=optimizer \
#       train_set=toy test_fraction=0 batch_size=2 group_size=2 env_max_turns=3 max_tokens=4096 \
#       prompt_budget_tokens=12288 eval_every=0 eval_at_start=False save_every=0   # 冒烟
#
# 在交给 train_cuda_grpo.sh 之前先查两件事, 免得等权重加载完 (几分钟) 才炸:
#   采样 TP 要能整除注意力头数和 KV 头数; 训练 rank 太少又不 offload 的话会 OOM。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONF="${CONF:-$ROOT/script/train_cuda_grpo_qwen35_9b.conf}"
# shellcheck disable=SC1090
source "$CONF"
for kv in "$@"; do
  k="${kv%%=*}"
  [[ "$kv" == *=* && "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] && declare "$kv"
done

IFS=',' read -ra _tg <<<"$TRAIN_GPUS"; IFS=',' read -ra _sg <<<"$SAMPLER_GPUS"
(( ${#_sg[@]} % SAMPLER_DP == 0 )) || { echo "SAMPLER_DP=$SAMPLER_DP 不整除采样卡数 ${#_sg[@]}"; exit 1; }
TP=$(( ${#_sg[@]} / SAMPLER_DP ))
read -r NH NKV < <(python3 -c "
import json; c = json.load(open('$POLICY/config.json')); t = c.get('text_config') or c
print(t['num_attention_heads'], t['num_key_value_heads'])") \
  || { echo "读不到 $POLICY/config.json"; exit 1; }
(( NH % TP == 0 && NKV % TP == 0 )) \
  || { echo "采样 TP=$TP 不合法: 要同时整除 num_attention_heads=$NH 和 num_key_value_heads=$NKV"; exit 1; }
if (( ${#_tg[@]} < 8 )) && [ "$ZERO_OFFLOAD" = none ]; then
  echo "训练 rank 只有 ${#_tg[@]} 个, 9B 不 offload 每卡要 >100G (见 conf 开头的估算), 请加 ZERO_OFFLOAD=optimizer"
  exit 1
fi
echo ">>> Qwen3.5-9B: train ranks=${#_tg[@]} offload=$ZERO_OFFLOAD | sampler ${SAMPLER_DP}x TP=$TP"
exec "$ROOT/script/train_cuda_grpo.sh" "$@"
