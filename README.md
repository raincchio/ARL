# ARL — Agentic RL (GRPO)

从 ARM（`../ARM`，8821e43）里 Qwen3.8-27B 的 GRPO 训练路径（`script/4c_train_qwen38.sh`）整理出来的精简训练器；算法与训练核心已继续对齐到 **2026-11-16 ARM 工作区**（HEAD `021af58`，含当前未提交的训练改动）。
**对话流程从单轮或 Kevin 式 refine 改成了多轮工具调用的 agentic RL**；环境通过消息层协议接入，
第一个接进来的是 CudaEnv（`envs/coding/CudaEnv`，CUDA-Agent 工作区环境）。
环境分为 **coding** 和 **instruction_following** 两类。CudaEnv（算子优化）与 SweEnv（SWE-bench 仓库修复）均属于 coding；instruction_following 已接入 IFEval；coding 也支持 ProgramBench 从零重建程序。
环境本体放在 `envs/<category>/<Name>/`，训练适配放在 `arl/envs/<category>/<name>.py`；注册和分类说明见 [环境目录](envs/README.md)。

Qwen3.6 SWE 当前入口为 `script/4b_train_qwen36.sh`：mini_swe_agent、250 轮、异步评估。
评估统一默认 TP=4、每题8条、并发64，全部题目一起排队，关闭路由回放；结果上传到
独立的 `<训练名>-eval` W&B run，记录 resolved_rate 和 pass_any_rate。
默认值及覆盖参数见 [SWE 说明](envs/coding/SweEnv/README.md)。

启动时用 `train_group_size` 和 `eval_group_size` 分别设置训练、评估的每题轨迹数。

```bash
./script/4b_train_qwen36.sh train_group_size=16 eval_group_size=8
```

两者默认均为 8。训练值指定每题目标保留的轨迹数，评估值用于基座、周期 checkpoint 和最终评估。
旧参数 `group_size`（训练）及 `eval_samples`、`eval_num_samples`、`EVAL_SAMPLES`（评估）继续可用。
参数随运行冻结到 `async_eval/spec.json`，修改启动参数不影响已经创建的运行。

训练和评估共用的解码参数默认是 `temperature=1.0 top_p=0.95 top_k=20`。
评估可逐项覆盖，例如 `./script/4b_train_qwen36.sh temperature=0.8 top_p=0.9 top_k=40 eval_temperature=0.6`。
基座、周期 checkpoint 和最终评估都会使用覆盖值。`eval_top_p` 和 `eval_top_k` 同样可单独设置。

Qwen3.6 SWE 训练默认用 `0.05` 乘以每条轨迹的工具调用格式正确率。启动时可用
`./script/4b_train_qwen36.sh format_reward_weight=0.1` 调整，设为 `0` 可关闭。
也可通过环境变量 `FORMAT_REWARD_WEIGHT` 设置，显式命令行参数优先。

`train_concurrency` 控制每个训练 rank 的 SWE 工作区并发数，默认 2。16 个 rank 下，
`./script/4b_train_qwen36.sh train_concurrency=8` 将训练端上限设为 128 个工作区。
也可设置环境变量 `TRAIN_CONCURRENCY`。加 `--check` 可查看总上限 `train_workspace_limit`。
`train_concurrency=-1` 和 `eval_concurrency=-1` 分别取消训练端和评估端的工作区名额限制。
评估默认 64，也可用环境变量 `EVAL_CONCURRENCY` 设置。实际并发取决于轨迹数量和 SWE 服务容量。

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
| advantage | Dr-GRPO，adv_unit=trajectory，组 = 题 | 相同（最新版 `finalize_ppo_data`，可选 TailRL 与正确性优势屏蔽） |
| 历史 / 序列 | 每轮整段重写 prompt，不回放思考，一轮一个 datum | **harness 式**：思考原样回放，token 只追加（`AgentRenderer.continue_prompt`），一条轨迹一个 datum；超预算触发压缩时前缀断开，另起一个 datum |
| 截断 | drop_truncated：丢掉该轮并结束轨迹 | `truncated_turn=mask`：该轮留在序列里但不进 loss，轨迹继续（环境会提醒模型把推理写短） |
| 轮数 | require_full_turns：不满 4 轮的轨迹整条丢 | 去掉（agent 会提前交卷） |
| rollout | overlap_turns wave 循环 + extra_samples | 同一套 wave 循环，换成消息层环境 |
| 训练栈 | training_client / sampling / NCCL 同步 / loss / train_step | 沿用 ARM，并同步 warmup、优化器存档/恢复、打包后的跨 rank 重排；保留 CudaAgent 的 `_row_logits` 防 OOM 修复 |
| 删掉的 | RM、π_ref KL、value head / tower critic、长度奖励、轮间同步 rollout、curriculum | — |

训练 rollout 按轨迹分配到各 rank：`batch_size=8, group_size=8` 在 16 个 rank 上每个管理 4 条轨迹。
同题的 8 条 reward 和有效 token 数经 CPU/Gloo 汇总，用完整组计算 GRPO/TailRL 基线和全局 advantage 白化，token 数据留在本 rank。
`extra_samples` 按全组完成顺序保留前 `group_size` 条，其余跨 rank 取消；训练前继续按 token 数重平衡。
`script/run_status.py` 显示各 rank 实际分到的轨迹数。评估继续按题分配。

## 布局

```
arl/grpo.py              GRPOConfig (chz) + run_grpo: 批循环、评估、存档、熔断
arl/train.py             入口 (torchrun -m arl.train key=value ...)
arl/agent/protocol.py    环境协议 (AgentEnv / TaskSource, 鸭子类型)
arl/agent/render.py      消息 -> token: 追加 (harness 常态) / 重渲染 + 逐级压缩 (超预算兜底)
arl/agent/rollout.py     agentic wave rollout + 留档 (trajectories/, best_solutions/)
arl/envs/coding/cuda.py  算子优化适配 (env=coding.cuda，兼容 cuda)
arl/envs/coding/swe.py   SWE-bench 适配 (env=coding.swe，兼容 swe)
arl/envs/instruction_following/  IFEval 指令遵循训练适配
envs/coding/CudaEnv/     算子工作区独立环境包
envs/coding/SweEnv/      SWE-bench 独立环境包
envs/instruction_following/     指令遵循类环境目录
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
    prompt_budget_tokens=12288 eval_every=0 eval_at_start=False save_every=0   # 冒烟
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
- `rollout/truncated_rate`：达到每轮生成上限而截断的比例；生成上限固定为 `max_tokens`。
- `rollout/ctx_overflow_trajs`：历史压缩到最狠一档还是塞不进 prompt 预算的轨迹数。
- `rollout/append_rate`：走 token 追加路径的轮次占比（理想接近 1）；`rollout/datums_per_traj`（理想是 1）、`rollout/datum_len_max`（≤ prompt_budget + max_tokens）。
- `rollout/turns_per_traj`、`rollout/n_invalid_calls_per_turn`。
- `ppo/grad_norm`、`ppo/clip_frac`、`ppo/adv_zero_frac`（整组同分、没有信号的样本占比）。
- `eval/pass_rate` 和 `eval/geomean_speedup_*`（held-out，每 `eval_every` 批评一次）。

## 已知差异和待办

- 默认 checkpoint 只存权重。`save_optimizer=True` 同时存 DeepSpeed 状态与 warmup 进度；只保留最新优化器状态，旧 policy 权重保留。恢复使用 `load_checkpoint_path=runs/.../checkpoints/000005 resume_optimizer=True start_batch=5`。`start_batch` 仍需指定；不恢复随机数、环境与生成上限闩状态，因此不保证逐位复现。
- 序列打包（`PACK_TOKENS`）和分块 lm_head（`HEAD_CHUNKS`）的代码都在，但没有开（当前 ARL 默认仍关闭）。
  一条轨迹合成一个 datum，最长 = `prompt_budget_tokens + max_tokens` = 32768 + 16384 = 49152，
  对应 ARM 4c 在 96G 卡上实测的 ~50k 上限，所以 conf 里默认不放宽生成上限。要更长先开分块 lm_head。
- `top_k=20` 沿用 CudaAgent 在 agent 模式下的实测结论（温度 1.0 不截断 top_k 时 Qwen3.8 会退化），
  和 ARM 4c 的 -1 不一样。

## 本次算法对齐

- `warmup_steps=5 warmup_lr_scale=0.5`：前 5 次实际 policy 更新用半学习率，非有限跳步不计数，续训恢复进度；`ppo/learning_rate` 记录实际学习率。
- `mask_positive_incorrect` / `mask_negative_correct`：默认 False。保留原组基线与白化统计，在白化之后分别屏蔽错误轨迹的正优势、正确轨迹的负优势。标签来自终局环境指标 `correct`，未知标签不屏蔽；同轨迹压缩出的所有 datum 共享标签。
- `adv_estimator=grpo|tailrl`，以及 `group_adv_std`、`group_adv_std_floor`、`whiten_advantages`、`whiten_std`、`length_weighted_baseline` 都可从 conf 或命令行传入；Dr-GRPO 默认值保持不变。
- `think_cost_lambda=0` 默认关闭。开启后按原始采样 token 计思考代价，每次回复只对超过 `think_cost_free_tokens=0` 的部分按每 8192 token 扣 lambda；各轮代价相加，从轨迹终局训练回报扣一次，再共享给该轨迹所有 datum。截断但保留的轮仍计代价，强制 token 不计；任务奖励、正确率和评估指标保持环境原口径。可独立开启 `think_budget` 两段式生成；强制插入的收尾 token 不参与 loss。两项均要求 `enable_thinking=True`，不要求 Kevin 的 multi_turn 环境。
- Qwen3.6-35B-A3B 的 16 卡 ZeRO-3 训练可用 `bash script/4b_train_qwen36.sh --sp16` 开启序列并行。默认关闭，显式关闭用 `--no-sp16`。加 `--check` 可以只检查配置。开启后，16 个训练 rank 共同处理每个 token 包，采样和评估沿用原有并行布局。
- `PACK_TOKENS=<预算>` 开启打包后，跨 rank 按 token 数重排负载，并携带源 rank 的 response 权重保持 loss 归一口径；`REBALANCE=0` 关闭重排。`HEAD_CHUNKS` 同样可由启动器传入。ZeRO-3 的 `overlap_comm` 对齐 ARM 默认关闭，可通过 `KERNEL_RL_ZERO3_OVERLAP_COMM=1` 显式开启。

保留 ARL 的 agent 工具协议、轨迹终局里程碑奖励、截断轮 mask 和历史压缩策略。本次不移植 ARM 的 Kevin harness 混合、RM/value/PPO 支路、逐轮奖励/折扣和环境适配。训练侧已同步 sampling replay、MoE 路由回放、固定 RMSNorm 和共卡运行时；现有环境与 Claude Code/MCP 的执行语义保留。

- `sampling_replay=True` 同时切换采样侧 processed logprob 和训练侧 temperature/top-k/top-p 分布；默认 False。开关不可按混合环境单独切换，各环境自己的采样参数仍逐批传给 loss。
- `router_replay=True` 开启 MoE 专家路由回放，路由数据沿 sample → transition → datum → trainer 传递。默认 False；与 ARM 一样要求支持的 Qwen MoE 及修补后的 vLLM routed-experts buffer，否则启动时报错。
- 5120 维 Qwen3.5/3.8 使用 ARM 的固定 RMSNorm 采样内核，`KERNEL_RL_FIXED_RMSNORM=0` 可关闭；不改训练 backward。
- `COLOCATE=1` 可让采样 GPU 成为训练 GPU 的子集，通过 KV 释放/唤醒和同卡 IPC 同步权重；默认 `COLOCATE=0` 保留本仓库现有分卡布局。共卡时设置 `TRAIN_GPUS`、`SAMPLER_GPUS` 和显存比例，`COLOCATE_HEADROOM_GB=8` 为默认训练余量。
- 训练 logger 同步 ARM 的 W&B 专用默认代理，显式 `KERNEL_RL_WANDB_PROXY=` 可关闭。保留 ARL 的 `_row_logits` 显存修复、单 rank 首次同步初始化和 policy 子目录 metadata 恢复。

CPU 回归：`python3 -m unittest discover -s tests`；真实 tokenizer + 假采样/评测的 agent 冒烟：`python3 tests/test_agent_rollout.py`。

## SWE-bench 环境接口与 mini-swe-agent

已注册 `env=coding.swe`，环境实现位于 `envs/coding/SweEnv/`，适配器为 `arl/envs/coding/swe.py`。
每条轨迹独立 Docker 工作区。mini-swe-agent 使用固定版本官方 SWE 提示词，通过 `bash` 修改仓库并输出选定 patch 交卷，官方 harness 在新容器判分，奖励为 resolved 的 0/1。SWE 示例使用 250 步、每条命令 60 秒；本地运行差异见 [harness 说明](harnesses/mini_swe_agent/README.md)。
默认复用现有远程 Docker 宿主机和 500 条 Verified 数据，固定划分 450 条训练、50 条 held-out。

```bash
./script/train_swe_grpo.sh
```

参数、离线回归和真实 harness 冒烟见 [SweEnv 文档](envs/coding/SweEnv/README.md)，启动配置见 [train_swe_grpo.conf](script/train_swe_grpo.conf)。

## Benchmark 目录与入口

已按清单搭建 26 个独立 benchmark 目录（coding 15 个、instruction_following 11 个），
各版本分别登记在 `envs/<category>/benchmarks/<group>/<benchmark>/`。
默认选择 benchmark 评测入口；显式 `--mode train` 提示“训练接入待完成”。
当前 CUDA、SWE、ProgramBench、IFEval、MultiCodeIF 均通过各自 harness 接入本地 on-policy GRPO；MultiCodeIF 另保留 API/离线自修复，ProgramBench 另保留提交包离线官方判分。其余仅有目录的新 benchmark 不自动成为训练环境。

```bash
python3 -m arl.benchmarks --list
python3 -m arl.benchmarks coding.swe_bench_verified
python3 -m arl.benchmarks instruction_following.ifeval --mode train
```

完整清单、分类与接入状态见 [Benchmark 目录](envs/BENCHMARKS.md)。
已有 `coding.swe` / `coding.cuda` 训练环境及启动脚本保留；新增 benchmark 名称不会自动转入这些训练流程。

MultiCodeIF：先运行 `./script/setup_multicodeif.sh`，再用 `python3 -m arl.benchmarks multicodeif` 查看数据清单；实际评测通过 `--config` 配置。详见 [MultiCodeIF 接入说明](envs/instruction_following/benchmarks/code/multicodeif/README.md)。

## Agent harness

Agent 执行流程单独放在 [harnesses/](harnesses/README.md)。`arl_agent` 已接入 CUDA 算子/KernelBench 的采集、GRPO 和评估；`multicodeif_self_repair` 已接入 MultiCodeIF 本地 token 采集、自修复 GRPO 和评估，并保留 API/离线文本接口。`mini_swe_agent` 已通过真实上游 DefaultAgent 接入 SWE/ProgramBench 的本地采集、GRPO 和评估；`claude_code` 已通过真实 CLI 和本地 Qwen token 桥接入 SWE 的采集、GRPO 和评估，现为 SWE 默认；`codex` 的运行适配仍待完成。

通过 [实验配置](experiments/README.md) 分别指定 collect/train/evaluate 的 env、harness、model 和 split，启动前校验配套关系。默认执行 evaluate；用 `--check` 仅校验，不加载模型。MultiCodeIF 的本地 on-policy GRPO 配置见 `experiments/multicodeif_grpo.yaml`；五环境共享策略训练见 `experiments/all_environments_grpo.yaml`。API/离线文本输出不能代替 token/logprob/mask 训练轨迹。

```bash
python3 -m arl.experiments experiments/swe.yaml --phase collect --check
python3 -m arl.experiments experiments/multicodeif.yaml --phase evaluate --check
python3 -m arl.experiments --matrix
```

## Instruction following 训练

`instruction_following.ifeval`（别名 `ifeval`）已接通原生 GRPO，默认 `arl_agent`。使用固定版本官方规则检查器，支持 strict 全约束 0/1 奖励和 fraction 比例奖励；默认公开题目内部划分 487/54。

```bash
./script/setup_ifeval.sh
python3 -m arl.experiments experiments/ifeval.yaml --phase train --check
./script/train_ifeval_grpo.sh
```

数据使用范围、官方判分、离线验证和训练配置见 [IFEvalEnv](envs/instruction_following/IFEvalEnv/README.md)。

训练可通过 `train.environments` 同时指定多个环境，各自设置 harness、数据和采样权重，共享一个模型与优化器；每个全局批次选择一个环境，评估与指标按环境分别记录。示例：[SWE + IFEval 混合配置](experiments/mixed.yaml)，详细语义见 [实验说明](experiments/README.md)。

## ProgramBench

`coding.programbench` 已支持官方 v1.2.5 判分，默认 `mini_swe_agent` 已接入本地 collect/train/evaluate；先运行 `bash script/setup_mini_swe_agent.sh` 安装固定版本。默认奖励是官方有效测试通过比例，同时记录全部通过的 resolved；基础设施错误丢弃。

```bash
python3 -m arl.experiments experiments/programbench.yaml --phase train --check
python3 -m arl.experiments experiments/mixed_programbench.yaml --phase train --check
```

环境、Docker 资源准备与提交包评测见 [ProgramBenchEnv](envs/coding/ProgramBenchEnv/README.md)。

## 当前 harness 配套边界

`arl_agent` 仅用于 CUDA 算子优化 / KernelBench（coding.cuda）。IFEval 使用独立的 direct 单轮生成 harness；SWE 与 ProgramBench 默认 mini_swe_agent，已通过上游 agent 与本地 token bridge 接入，不允许 arl_agent 兜底。单环境和混合配置预检通过；CPU 联调验证采样、训练数据、共享优化器及参数同步调用，真实 GPU 训练尚未运行。离线提交包评测继续可用。
