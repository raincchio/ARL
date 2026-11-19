#!/usr/bin/env bash
# Shared training/sampler GPU layout. No changes to environment configuration.
arl_training_gpu_layout() {
  local train_gpus="$1" sampler_gpus="$2" colocate_mode="$3"
  local -a train_ids sampler_ids
  local gpu other found
  [[ "$colocate_mode" == 0 || "$colocate_mode" == 1 ]] || { echo 'COLOCATE must be 0 or 1'; return 1; }
  IFS=',' read -ra train_ids <<<"$train_gpus"
  IFS=',' read -ra sampler_ids <<<"$sampler_gpus"
  [[ -n "$train_gpus" && -n "$sampler_gpus" ]] || { echo 'Training and sampler GPUs must be nonempty'; return 1; }
  for gpu in "${sampler_ids[@]}"; do
    found=0
    for other in "${train_ids[@]}"; do
      [[ "$gpu" != "$other" ]] || found=1
    done
    if [[ "$colocate_mode" == 1 && "$found" == 0 ]]; then
      echo 'COLOCATE=1 requires SAMPLER_GPUS to be a subset of TRAIN_GPUS'; return 1
    elif [[ "$colocate_mode" == 0 && "$found" == 1 ]]; then
      echo 'Training and sampler GPUs overlap; set COLOCATE=1 for shared GPUs'; return 1
    fi
  done
  ARL_TRAIN_VISIBLE_GPUS="$train_gpus"
  [[ "$colocate_mode" == 1 ]] || ARL_TRAIN_VISIBLE_GPUS+=",$sampler_gpus"
}
