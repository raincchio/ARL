# ARL — Agentic RL (GRPO)

从 ARM（`../ARM`，8821e43）里 Qwen3.8-27B 的 GRPO 训练路径（`script/4c_train_qwen38.sh`）整理出来的精简训练器。
**对话流程从单轮或 Kevin 式 refine 改成了多轮工具调用的 agentic RL**；环境通过消息层协议接入，
第一个接进来的是 CudaEnv（`envs/CudaEnv`，CUDA-Agent 工作区环境）。
环境本体放在 `envs/<Name>/`（各自独立的包，ARL 不 import 时不依赖），ARL 侧的适配放在 `arl/envs/<name>.py`；之后的 SweEnv 同样处理。

```
torchrun ×N 训练 rank (DeepSpeed ZeRO-3)          rank0 起的 vLLM (SAMPLER_GPUS, TP)
   │  arl.grpo: rollout → advantage → train_step → NCCL 权重同步 ─────┘
   │
   └─ arl.agent.rollout (wave 异步): 渲染 prompt → submit_and_poll → env.step(text)
                                                   │
                                   CudaEnv (工具: write/edit/read_file, compile/verify/profile[/analyze])
                                                   │ POST /repo/evaluate
                                   CudaEval 集群 (CudaTasks 题库判分 / 防作弊 / 基线)
```

## 和 ARM 的对应

| | ARM 4c (Qwen3.8-27B) | ARL |
|---|---|---|
| 对话 | Kevin refine：4 轮，每轮重写完整 kernel，每轮都评分 | **agentic**：最多 16 轮工具调用，一轮里可以有多个工具调用，回复里没有工具调用就算交卷 |
| 奖励 | 每轮评测分（0.3·正确 + log 加速比），gamma=0.8，按 late 方式折扣 | **轨迹终局里程碑** {-1,1,2,3}（best-of-trajectory），一条轨迹的每一轮共享这个值 |
| advantage | Dr-GRPO，adv_unit=trajectory，组 = 题 | 相同（`finalize_ppo_data` 原样） |
| 历史 / 序列 | 每轮整段重写 prompt，不回放思考，一轮一个 datum | **harness 式**：思考原样回放，token 只追加（`AgentRenderer.continue_prompt`），一条轨迹一个 datum；超预算触发压缩时前缀断开，另起一个 datum |
| 截断 | drop_truncated：丢掉该轮并结束轨迹 | `truncated_turn=mask`：该轮留在序列里但不进 loss，轨迹继续（环境会提醒模型把推理写短） |
| 轮数 | require_full_turns：不满 4 轮的轨迹整条丢 | 去掉（agent 会提前交卷） |
| rollout | overlap_turns wave 循环 + extra_samples | 同一套 wave 循环，换成消息层环境 |
| 训练栈 | training_client / sampling / NCCL 同步 / loss / train_step | **原样复制**（包名 r3lm→arl），另外补了 CudaAgent 的 `_row_logits` 防 OOM 修复 |
| 删掉的 | RM、π_ref KL、value head / tower critic、长度奖励、think_budget、sampling_replay、tailrl、轮间同步 rollout、curriculum | — |

## 布局

```
arl/grpo.py              GRPOConfig (chz) + run_grpo: 批循环、评估、存档、熔断
arl/train.py             入口 (torchrun -m arl.train key=value ...)
arl/agent/protocol.py    环境协议 (AgentEnv / TaskSource, 鸭子类型)
arl/agent/render.py      消息 -> token: 追加 (harness 常态) / 重渲染 + 逐级压缩 (超预算兜底)
arl/agent/rollout.py     agentic wave rollout + 留档 (trajectories/, best_solutions/)
arl/envs/cuda.py         CudaEnv 适配 (CudaTasks 题库 -> TaskSource)
envs/CudaEnv/            CudaEnv 本体 (cudaenv 包, 由 ARL 仓库统一管理)
arl/algorithms/          loss / train_step / ppo_data (Dr-GRPO) / data_processing   <- ARM
arl/models/              training_client / sampling / dist_sampling / renderers ...  <- ARM
arl/utils/               distributed / ml_log / checkpoint_utils                      <- ARM
script/train_cuda_grpo.sh/.conf   启动器 (GPU 划分、评测探活、torchrun)
tests/test_agent_rollout.py       单进程端到端 (假采样 + 真 tokenizer + 真 CudaEnv/假评测)
```

## 用法

```bash
./script/train_cuda_grpo.sh                                   # 27B, 16 卡: 训练 8 + 采样 8 (conf 里的 A/B 组)
./script/train_cuda_grpo.sh POLICY=../CudaAgent/runs/rft_v0/policy_full     # 从 RFT checkpoint 起
./script/train_cuda_grpo.sh POLICY=/models/Qwen/Qwen3-8B TRAIN_GPUS=12,13 SAMPLER_GPUS=14,15 \
    train_set=toy test_fraction=0 batch_size=2 group_size=2 env_max_turns=3 max_tokens=4096 \
    max_tokens_late=0 prompt_budget_tokens=12288 eval_every=0 eval_at_start=False save_every=0   # 冒烟
python3 tests/test_agent_rollout.py                           # 不用 GPU
```

参数分两种写法：大写的（`POLICY` / `TRAIN_GPUS` / `SAMPLER_GPUS` / `EVAL_ENDPOINT` ...）是启动器变量，
小写的是 `GRPOConfig` 的键，命令行写的值优先于 conf。

产出在 `runs/<name>/`：
- `metrics.jsonl`（同时上报 wandb）
- `checkpoints/N/`：HF 格式，复合 config 已经补回，vLLM 可以直接加载，不需要再跑合并脚本
- `trajectories/rank*.jsonl.gz`：每条轨迹每一轮的回复原文和工具结果
- `best_solutions/rank*.jsonl`：每道题最好成绩对应的工作区文件

## 要盯的曲线

- `rollout/reward_mean`：里程碑均值，应该从 ≈-1 往上爬。
- `rollout/correct_rate` / `faster_eager_rate` / `faster_compile_rate`。
- `rollout/truncated_rate`：超过 8% 时会自动把生成上限放宽到 `max_tokens_late`。
- `rollout/ctx_overflow_trajs`：历史压缩到最狠一档还是塞不进 prompt 预算的轨迹数。
- `rollout/append_rate`：走 token 追加路径的轮次占比（理想接近 1）；`rollout/datums_per_traj`（理想是 1）、`rollout/datum_len_max`（≤ prompt_budget + max_tokens）。
- `rollout/turns_per_traj`、`rollout/n_invalid_calls_per_turn`。
- `ppo/grad_norm`、`ppo/clip_frac`、`ppo/adv_zero_frac`（整组同分、没有信号的样本占比）。
- `eval/pass_rate` 和 `eval/geomean_speedup_*`（held-out，每 `eval_every` 批评一次）。

## 已知差异和待办

- **不支持精确续跑**：checkpoint 只存权重（同 ARM）。`load_checkpoint_path=` 配合 `start_batch=` 可以从
  中断处接着跑数据，但优化器状态会重新开始。
- 序列打包（`PACK_TOKENS`）和分块 lm_head（`HEAD_CHUNKS`）的代码都在，但没有开（ARM 4c 也没开）。
  一条轨迹合成一个 datum，最长 = `prompt_budget_tokens + max_tokens` = 32768 + 16384 = 49152，
  对应 ARM 4c 在 96G 卡上实测的 ~50k 上限，所以 conf 里默认不放宽生成上限。要更长先开分块 lm_head。
- `top_k=20` 沿用 CudaAgent 在 agent 模式下的实测结论（温度 1.0 不截断 top_k 时 Qwen3.8 会退化），
  和 ARM 4c 的 -1 不一样。
