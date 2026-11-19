# SweEnv — SWE-bench 的 agentic RL 环境

2026-11-13：已增加宿主机 master/worker 服务后端，并在 `.70`、`.71` 两机部署验证。训练容器通过 `swe_endpoint=http://10.0.134.70:18770` 调用；默认 `script/train_swe_grpo.conf` 和所有含 SWE 的实验示例均使用该 HTTP 服务；`script/train_swe_grpo_local.conf` 为兼容入口。服务源码和统一部署入口已迁到 `.70` 宿主机 `/root/workspace/SweContainerService`。本仓库只保留训练客户端及 [客户端检查说明](../../../script/swe_client/README.md)。原本地/SSH 后端调用宿主机已有 worker 脚本，可通过 `swe_worker_script` 指定路径。

分类：`coding`；训练环境名 `coding.swe`（兼容 `swe`）。

`SweWorkspaceEnv` 实现 ARL 的 `AgentEnv`，`arl/envs/coding/swe.py` 提供 `TaskSource`。训练直接使用 SWE-bench 实例；每题 G 条轨迹共享问题，分别使用独立工作区与最终 0/1 奖励。

一次轨迹：官方 instance 镜像 → `/testbed` checkout 到 `base_commit` → 多轮 `bash` 修改源码、自测 → `submit` 或无工具回复交卷 → 提取相对 base commit 的完整 diff → **全新容器**运行官方 SWE-bench harness → resolved=1，否则=0。

工具为 OpenAI function schema，支持 Qwen XML 与 Hermes JSON 调用：

```text
<tool_call>
<function=bash>
<parameter=command>
cd /testbed && sed -n '1,160p' package/module.py
</parameter>
</function>
</tool_call>
```

`bash` 每次是新 shell，文件修改持久化，默认工作目录 `/testbed`，conda 的 `testbed` Python 已在 PATH。`submit` 无参数，立即结束轨迹。默认最多 64 轮、每轮最多 4 个调用、命令超时 120 秒、输出最多 16 KB。生成被截断时不执行其中的工具调用，反馈给模型重试。上下文超预算时逐级删旧思考、压工具输出、丢旧轮次，仓库文件不受历史压缩影响。

## 训练

仓库默认对接 `/root/share/swe-qwen-collect` 使用的 HTTP master、Verified 数据和官方评分器：

```bash
./script/train_swe_grpo.sh
# 小规模训练；实际需要 GPU 和可访问的 Docker 宿主机
./script/train_swe_grpo.sh TRAIN_GPUS=0,1 SAMPLER_GPUS=2,3 \
  task_limit=4 test_fraction=0 batch_size=2 group_size=2 env_max_turns=8 \
  max_tokens=1024 prompt_budget_tokens=8192 eval_every=0 save_every=0
```

配置见 `script/train_swe_grpo.conf`。默认从 Verified 的 500 个实例中，按固定 seed 切出 50 个 held-out，450 个实例直接训练。`test_fraction=0` 使用全部实例训练；它们的结果应标记为训练集结果。显式 `test_set` 必须与 `train_set` 的 instance ID 不重叠。

`train_set` / `test_set` 支持：

- `remote:/host/path/instances.jsonl`：数据在 Docker/SSH 宿主机，由 worker 读取。默认走这条路径，无须下载数据。
- 本地 JSONL 或 JSON 数组：完整官方实例记录。判分所需的 `test_patch`、`FAIL_TO_PASS`、`PASS_TO_PASS`、`version` 等字段保留在可信记录中；模型只看到 repo、base_commit 与 problem_statement。
- `verified` / `lite` / `full` 或 Hugging Face dataset ID：通过 `datasets.load_dataset(..., split=swe_dataset_split)` 读取，需要本地缓存或允许网络。启动器默认开启 HF 离线模式，以现有数据文件为主。

同一实例的多轮/压缩段共享最终回报与正确性标签，沿用 trajectory GRPO、warmup、优势屏蔽、thinking cost 和优化器 checkpoint。官方隐测只在交卷后运行，测试反馈不作为后续 agent 轮次输入。空补丁和测试失败奖励为 0；Docker、SSH 或 harness 基础设施失败标记为 infra，整条轨迹退出训练。最终 patch 包括 tracked、staged、新增文件以及相对 base commit 的提交差异。

## Docker worker

训练机器仅需 Python、训练依赖；不依赖本地 Docker SDK 或 SWE-bench 包。推荐配置 `swe_endpoint` 调用宿主机常驻 master。兼容的本地/SSH 后端仅将 JSON 请求发给宿主机已有的 `swe_worker_script`，由 `swe_python` 执行，不再从训练仓库发送服务端源码。两种后端都不修改 AgentBench 服务。

宿主机要求：Docker CLI/daemon、Python 环境内的 `docker` 和 `swebench`；提前缓存所选实例的官方 `swebench/sweb.eval.x86_64.*:latest` 镜像。首次缺镜像会报 infra，不会在训练过程中隐式下载数 GB 镜像。本版本针对标准 Python SWE-bench（Verified/Lite/Full），不包含 Pro、Multimodal 等其他 harness。

关键参数：

| 参数 | 含义 |
|---|---|
| `swe_ssh_host` / `swe_ssh_port` | Docker 宿主机；空 host 为本机 |
| `swe_python` | 宿主机安装官方 harness 的 Python 绝对路径 |
| `swe_work_dir` | 宿主机判分日志、实例记录、predictions 的保存目录 |
| `swe_proxy` | 仅可信 grader 用于取得官方环境配置；不传给 agent 容器 |
| `swe_memory` / `swe_cpus` | 每个 agent 工作区的 Docker 资源限制，默认 4g / 2 CPU |
| `eval_max_concurrency` | **每个训练 rank** 的活跃轨迹上限；实际并发同时受 master 的集群容量和按 IP 调度限制 |
| `swe_command_timeout` / `swe_eval_timeout` | agent 命令和官方评测各自的超时 |

Agent 容器禁网、无宿主机挂载、无 Docker socket，删除 capabilities；每条轨迹独立。官方金标 patch、test_patch 与 tests 列表只在可信 grader 侧使用。结束、异常、超额采样取消时回收容器，agent 容器另有 4 小时退出上限。判分使用唯一 run ID，避免官方 harness 复用旧报告。官方评测容器由 harness 按其默认资源设置运行。

指标包含 `rollout/resolved_rate`、`eval/resolved_rate`、`ppo/reward_mean`。轨迹留档包含终局 patch 和判分路径；`best_solutions/rank*.jsonl` 的成功条目同时包含 `instance_id`、`model_name_or_path`、`model_patch`，方便导出标准 predictions。

## 验证

```bash
python3 -m unittest discover -s tests -p test_swe_env.py
# 真实工作区：空补丁应为 0，不启动官方隐测
python3 script/smoke_swe_env.py \
  --endpoint http://10.0.134.70:18770 \
  --token-file /root/ai_data/swe/service/token \
  --dataset remote:/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl
# 可信集成测试：应用参考补丁，验证官方判分应为 1；绝不用于模型训练轨迹
python3 script/smoke_swe_env.py --gold \
  --endpoint http://10.0.134.70:18770 \
  --token-file /root/ai_data/swe/service/token \
  --dataset remote:/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl \
  --proxy http://10.0.134.70:10809
```

已实测 `sympy__sympy-21847` 的参考补丁：官方 harness 4.1.0，resolved=True，FAIL_TO_PASS 1/1，PASS_TO_PASS 9/9。另用不修复问题的补丁实测 resolved=False、奖励 0；两者均无 infra 错误。没有启动 GPU RL 训练。

官方协议与评测说明：[SWE-bench harness](https://www.swebench.com/SWE-bench/reference/harness/)。

当前 harness 边界：本环境默认 [claude_code](../../../harnesses/claude_code/README.md)，通过本地 Qwen token 桥运行真实 CLI、MCP 远程 bash 和最终工作区 diff 判分。mini_swe_agent 仍可显式选择；其真实上游 DefaultAgent 已通过本地 token bridge 接入 collect/train/evaluate。mini_swe_agent 直接加载 2.4.6 官方 swebench.yaml 提示词与反馈模板，使用模型输出的 patch 提交；本文原生 bash/submit 示例用于底层环境测试。SWE 示例预算为 250 步、命令 60 秒；本地采样/计费及隔离差异见 [harness 说明](../../../harnesses/mini_swe_agent/README.md)。选择 mini_swe_agent 时先运行 `bash script/setup_mini_swe_agent.sh`；默认 Claude Code 的依赖检查和运行方式见 [Claude Code 说明](../../../harnesses/claude_code/README.md)。训练配置预检和 CPU 训练链路验证通过，真实 GPU 训练尚未实跑；不允许用 arl_agent 替代。


## 前 10 题训练效果验证

已准备启动脚本 `script/train_swe10.sh`，起始模型默认
`/models/Qwen/Qwen3.6-35B-A3B`，harness 为 Claude Code。未指定题目参数时，按服务端
Verified JSONL 的顺序选前 10 题，在任何随机打乱之前固定为
`tasks.private.jsonl`，并保存公开 ID/数据散列到 `tasks.json`。
当前前 10 题全部是 Astropy 任务。训练和测试都使用这 10 题，评估显式
`split=train`、`test_fraction=0`；不会因自动 held-out 切分丢掉部分题。
这用于小集合上的训练效果检查，不能据此报告 held-out 泛化成绩。

从仓库根目录启动：

```bash
# 仅检查配置，不联网、不创建工作区、不占 GPU
./script/train_swe10.sh --run-dir runs/swe10_qwen36_v1 --check
# 按顺序执行：底座测试 -> GRPO -> 最终模型测试 -> 生成对比
./script/train_swe10.sh --run-dir runs/swe10_qwen36_v1
```

该目录已完成默认 10 题镜像 preflight。调整选题时请使用新 run 目录。
省略 `--run-dir` 会生成新的唯一目录。

默认训练与采样均使用全部可见 GPU，`COLOCATE=1`、`SAMPLER_DP=4`。5 epochs、每批 10 题、每题训练采样 8 条轨迹，
训练前后各每题采样 4 条轨迹；温度 1、top-p .95、top-k 20，预算保持一致。
最多 250 次模型回复、每轮 4096 token；总上下文默认读取模型配置，不另设固定输入预算。
只使用官方 resolved 0/1 奖励，thinking cost 默认关闭。
训练使用 ARM 对齐的 Dr-GRPO、clip .2/.28、2 substeps、lr 1e-6、5 次更新 warmup；
仅保存最终 policy checkpoint，不额外保存大型 optimizer 状态。

需要修改 GPU/规模时使用新的 run 目录，例如：

```bash
./script/train_swe10.sh --run-dir runs/swe10_qwen36_small \
  --train-gpus 0,1,2,3 --sampler-gpus 4,5,6,7 --no-colocate --epochs 2
# 默认共卡；采样卡必须是训练卡的子集
./script/train_swe10.sh --run-dir runs/swe10_qwen36_colocate \
  --train-gpus 0,1,2,3,4,5,6,7 --sampler-gpus 4,5,6,7 --colocate --sampling-util .65
```

`--phase prepare` 只固定任务/检查镜像；`--phase baseline|train|evaluate|compare`
可单独执行各阶段。中途阶段失败后，可在同一个目录执行尚未开始的后续阶段；
已有阶段输出不会被覆盖。改变模型/训练/评估参数需选择新目录。
训练和评估读取各自固定的任务文件；默认共用训练题目。完整 grader 记录权限为 0600，不写入公开 manifest。

输出 `baseline/`、`train/`、`after/`，日志分别为 `baseline.log`、`train.log`、
`evaluate.log`；最终模型在 `train/checkpoints/final/policy/`。
`comparison.json` 汇总训练前后 resolved 次数/比率、每题至少成功一次的题数、
逐题结果、丢弃样本、实际 optimizer step 数、非零梯度/有效 action token 批数。
有基础设施失败、没有 optimizer step 或没有非零训练信号时标记为 inconclusive，
不会自动宣称算法有效。样本量小且采样有随机性，前后差异应结合逐题结果判断。


### 对齐用户 ARM 参数的 SWE10 配方

新增 `script/train_swe10_aligned.sh`，保留可用于 SWE 的训练设置：

```bash
# 只检查最终配置和传入的训练开关
./script/train_swe10_aligned.sh --check
# 启动完整训练前后对比流程；此命令此前尚未执行
./script/train_swe10_aligned.sh
```

配方为 Qwen3.6-35B-A3B、8 卡共卡、SAMPLER_DP=2（训练及独立测试都使用）、
sampling_util=.7、batch/group=8、10 epochs、每题测试 16 次、4 次模型回复、
max_tokens=40960、PACK_TOKENS=32768、HEAD_CHUNKS=8、
sampling/router replay=True、adv_estimator=grpo、think_budget=32768、
warmup_steps=30/scale=.5、thinking cost .005/free 8192、两种正确性优势屏蔽开启。
optimizer、完整 trajectories、best solutions 保存均关闭。训练内部 eval_at_start=False；
仍按小集合验证流程单独测试底座和最终模型。输出目录为
`runs/swe10_qwen36_aligned_v1`，可用 `--run-dir` 覆盖；其余参数可在配方末尾覆盖。

以下参数已向 `train_swe10.sh` 开放：

| 参数 | 默认值 |
| --- | --- |
| `--sampling-replay` / `--no-sampling-replay` | False |
| `--router-replay` / `--no-router-replay` | False |
| `--pack-tokens` | KERNEL_RL_PACK_TOKENS，未设置则 0 |
| `--head-chunks` | KERNEL_RL_HEAD_CHUNKS，未设置则 0 |
| `--max-tokens-late` | 0 |
| `--adv-estimator` | grpo；也支持 tailrl |
| `--warmup-steps` / `--warmup-lr-scale` | 5 / .5 |
| `--think-budget` | 0 |
| `--think-cost-lambda` / `--think-cost-free-tokens` | 0 / 0 |
| `--format-reward-weight` | 通用入口默认 0，Qwen3.6 aligned 配方为 0.05。每条训练轨迹按上报 `format_ok` 的轮次计算格式正确率，再乘以权重加到终局回报。无格式记录时加 0 分。 |
| `--mask-positive-incorrect` / `--no-mask-positive-incorrect` | False |
| `--mask-negative-correct` / `--no-mask-negative-correct` | False |
| `--save-optimizer` / `--no-save-optimizer` | False |
| `--save-trajectories` / `--no-save-trajectories` | True |
| `--save-best-solutions` / `--no-save-best-solutions` | True |
| `--truncated-turn` | mask；也支持 train/drop_traj |

关闭完整轨迹保存时，测试阶段仍生成 `outcomes.jsonl`：仅含 task、batch/tid、
reward、dropped、infra 标量，用于逐题统计和核对数量；不保存 prompt、回复或 patch。
thinking cost 仅作用于训练回报，测试报告仍使用官方 resolved 0/1。
训练若有动态生成预算，训练前后测试均采用峰值预算。

配方保留前缀环境变量：KERNEL_RL_ZERO3_REDUCE_BUCKET=100000000、
KERNEL_RL_FREEZE_ROUTER=0、KERNEL_RL_FREEZE_SHARED_EXPERT=1，可用同名环境变量覆盖。
当前生效的 vLLM 0.19 安装已包含两处 ARM 路由 buffer 修复，脚本直接复用，
不创建/启用隔离覆盖层。若其他机器缺少修复，启动时 `arl.models.vllm_overlay`
可创建项目内隔离覆盖层，复制两处修复文件，其余模块链接当前安装，
通过子进程 PYTHONPATH 使用，不修改共享 site-packages。不匹配的 vLLM 版本/源码直接拒绝修补。--check 不创建覆盖层。
覆盖层来源/散列记录在 run/vllm_overlay.json。该修复通过 CPU slot 索引边界验证，
本次尚未执行 GPU router replay/共卡训练。

未移植 Kevin/KernelBench 的 adv_unit=turn、traj_score/gamma、逐轮 reward improve、
速度/正确性混合奖励、kernel profiler/static_libcall、curriculum/levels、Kevin prompt、
length reward、异步评估和 ARM 截断反馈开关。SWE 保留 trajectory 终局奖励与当前
truncated_turn 策略；不把参数名相似的不同环境行为当作等价。


### 选择 SWE10 harness

`train_swe10.sh` 和 `train_swe10_aligned.sh` 均支持 `--harness`；通用入口默认
`claude_code`，Qwen 预设入口默认 `mini_swe_agent`。选择统一作用于 baseline、训练和最终测试，比较报告也记录所选 harness。
不同 harness 使用不同 run 目录，避免实验身份混用：

```bash
./script/train_swe10_aligned.sh --harness claude_code --run-dir runs/swe10_claude_aligned_v1
./script/train_swe10_aligned.sh --harness mini_swe_agent --run-dir runs/swe10_mini_aligned_v1
```

两条命令分别启动独立完整流程，不要同时占用同一组 GPU。加 `--check` 只做配置预检。
模型仍由 `--model` 指定本地 Qwen；harness 仅改变 agent 的交互/提交流程。


### ARM 风格参数与另一台机器异步评估

所有 SWE10 入口现在支持 `key=value`，也保留原 `--flag` 写法。重复参数后者覆盖前者，
布尔值支持 True/False、1/0；不会使用 shell eval。新增熟悉的入口：

```bash
# 训练机：ARM 对齐配方 + 外部异步评估，不等待测试。
KERNEL_RL_ZERO3_REDUCE_BUCKET=100000000 KERNEL_RL_FREEZE_ROUTER=0 KERNEL_RL_FREEZE_SHARED_EXPERT=1 \
./script/4b_train_qwen36.sh \
  run_dir=/root/share/arl-runs/swe10_qwen36_async_v1 \
  harness=mini_swe_agent COLOCATE=1 SAMPLING_UTIL=0.7 SAMPLER_DP=4 \
  sampling_replay=True router_replay=True batch_size=8 train_group_size=8 \
  num_epochs=10 eval_group_size=8 max_turns=250 eval_tp=4 eval_concurrency=64 \
  max_tokens=40960 PACK_TOKENS=32768 HEAD_CHUNKS=8 \
  think_budget=32768 think_cost_lambda=0.001 think_cost_free_tokens=8192 \
  warmup_steps=30 warmup_lr_scale=0.5 \
  mask_positive_incorrect=True mask_negative_correct=True \
  save_optimizer=False save_trajectories=False save_best_solutions=False \
  eval_at_start=False eval_async=True eval_every=5

# 上述命令加 check=True，仅预检，不联网、不写配置、不启动 GPU。
# 评估机：在训练端目录已经创建后入队，不在训练机执行此命令。
./script/eval_swe10_async.sh add /root/share/arl-runs/swe10_qwen36_async_v1 \
  GPUS=auto NGPUS=8 poll=20 retries=2
./script/eval_swe10_async.sh status
./script/eval_swe10_async.sh log
./script/eval_swe10_async.sh rm /root/share/arl-runs/swe10_qwen36_async_v1
./script/eval_swe10_async.sh redo /root/share/arl-runs/swe10_qwen36_async_v1 final
./script/eval_swe10_async.sh stop
```

`4b_train_qwen36.sh` 是本 ARL 仓库的 SWE10 配方别名，不是 ARM 的 KernelBench 任务入口。
`harness=mini_swe_agent` 可替换 Claude Code。普通/对齐/4b 三个入口省略 run_dir 都会
自动生成唯一目录；跨机评估需把打印的绝对目录传给评估机。示例显式使用共享路径，
两台机器必须按相同绝对路径访问 run 和初始模型；评估机需有相同版本的 ARL/harness。
`save_best_solutions=False` 控制是否保存成功解 patch 和评分信息。
KernelBench/ Kevin 专属参数（如 gamma、adv_unit=turn、eval_profiling）会报错，不会静默忽略。

异步评估服务改编自 ARM 的 `rlhf/ppo/eval_service.py`，复用其队列/通配符、跨进程
心跳锁、GPU UUID 共享租约、GPU 探测、失败重试、残留进程组回收、状态和 redo 存档。
GPU 租约继续使用 ARM 的 `/root/share/arm_async_eval/v1/gpu_leases` 协议，与 ARM/CudaMaster
协作；可用 GPU_LEASE_DIR 指定同一共享锁目录。独立服务状态位于本仓库
`logs/eval_service/<IP>/<instance>`；ARL_EVAL_SERVICE_DIR 可隔离测试/服务状态。
服务已运行时 add 不改变它的 GPU 参数，换配置用 restart。
新发布的 `async_eval/spec.json` 自动保存 `sampler_tp: 4`（可用训练入口的 `eval_tp` 覆盖）。
服务默认每个任务使用 8 卡、TP=4，即两个引擎，显存利用率 0.85。优先级为显式
`SAMPLER_DP` 或 `SAMPLER_TP`、run spec、共享默认 TP=4；两种显式参数不能同时指定。
旧 spec 没有 TP 时同样默认 TP=4，普通重启不会退回 TP=8。

训练机写 `async_eval/spec.json` 固定任务散列、harness、采样/预算和模型身份，
并按 eval_every 保存 policy。评估机只读取 `train/checkpoints.jsonl` 中已经发布的完整记录，
不会读取正在后台写入的 checkpoint；最终 batch 只评 final，避免周期 checkpoint 和 final
重复评估。每个 checkpoint 使用新子进程加载权重，结束后释放 GPU。
训练内部 eval_at_start=False；独立底座测试仍作为该小集合前后比较的基线，在评估机执行，
不阻塞训练。完整训练结束后，训练进程退出，不等待评估队列。

SWE10 独立评估默认将全部选中的题目及其采样轨迹放入同一批队列，最多同时运行
64 条轨迹（`eval_concurrency=64`），结束一条即可让等待的轨迹接入，不再每两题等待
整批结束。每题默认 8 条轨迹（`eval_group_size=8`），评估默认关闭 router replay，
只记录 `resolved_rate` 和 `pass_any_rate`。同步与异步评估均使用 TP=4；同步评估显存
利用率可用 `eval_util` 覆盖。训练仍由 `batch_size` 和 `concurrency` 单独控制。已启动的评估不会热更新；
修改当前 run 的 spec 后，下一次 checkpoint 子进程读取新配置。

`train_concurrency=-1` 和 `eval_concurrency=-1` 分别关闭训练和评估的本地工作区并发上限。
评估设置适用于基座、周期 checkpoint 和最终模型。正整数指定并发上限。

评估输出：`async_eval/metrics.jsonl`、`done/`、`failed/`、`running/`、`progress/`、
`results/<checkpoint>/<attempt>/` 和 `logs/`。失败重试使用新 attempt，不覆盖旧结果；
标量 outcomes 在关闭完整 trajectories 时仍保留。base 和 final 都完成后自动生成根目录
`comparison.json`，其中仍检查 infra、采样数、optimizer step 和非零梯度信号。
评估机 token 路径不同时可用 `SWE_EVAL_TOKEN_FILE=/path/to/token` 覆盖，令牌值不写入 spec。
实际跨机器 GPU 作业尚未启动；已通过 CPU 状态机、锁、租约和进程组回收验证。


### GPU 默认值更新：训练/采样都使用全部可见设备

`train_swe10.sh`、`train_swe10_aligned.sh`、`4b_train_qwen36.sh` 不再写死 8 张卡。
默认 TRAIN_GPUS=all、SAMPLER_GPUS=all、COLOCATE=1、SAMPLER_DP=4。
all 尊重显式 CUDA_VISIBLE_DEVICES；未指定可见性掩码时查询全部设备 ID。
可用 TRAIN_GPUS=0,1,... 和 SAMPLER_GPUS=... 显式覆盖，也支持 all/auto。
默认 packing 为 32768，保持 REBALANCE=1；batch_size 小于训练 rank 数时允许空 rank
参与 rollout 集合通信，然后通过已有训练 datum 重分配参与更新。
显式 PACK_TOKENS=0 且 batch_size 小于 rank 数时仍拒绝配置，避免误用不重排路径。

本机 16 卡的默认预检结果：16 个训练 rank，全部 16 卡采样，4 个 TP=4 副本。
前 10 题和 batch_size=8 保持不变；不为凑满 rank 而重复题目。
4 rank CPU/Gloo 测试验证空 rank 接收训练数据，且重排前后归一化目标一致。
没有启动 GPU 训练。

```bash
./script/4b_train_qwen36.sh SAMPLER_DP=4 check=True
# 仍可显式选卡，采样卡为训练卡子集；卡数需能被 DP 整除。
./script/4b_train_qwen36.sh TRAIN_GPUS=0,1,2,3 SAMPLER_GPUS=0,1,2,3 SAMPLER_DP=4 check=True
```

上述默认值针对训练机和 SWE10 的采样流程；另一台机器的异步评估服务仍通过
GPUS/NGPUS 单独指定设备，默认按照 TP=4 切分；也可显式指定 SAMPLER_TP 或 SAMPLER_DP。

`4b_train_qwen36.sh` / `train_swe10_aligned.sh` 当前预设为 mini_swe_agent、异步评估、
250 轮、每轮最多 40960 token、prefill 1056/2048、prefix caching 与 Mamba align。
训练开启路由回放，评估关闭；思考预算 32768、think_cost_lambda=0.001。
通用 `train_swe10.sh` 仍允许选择其他模型和 harness，Qwen 专属缓存参数由预设入口提供。


## 指定训练和评估题目

上述启动脚本（包括 `4b_train_qwen36.sh`）均支持 `key=value`：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `dataset` | SWE 服务机器上的训练 JSONL 路径 | Verified 数据集 |
| `task_limit`（也可 `train_task_limit`） | 不传训练 IDs 时取前 N 题 | `10` |
| `train_instance_ids` | 逗号分隔的训练 instance ID，保留给定顺序 | 未指定 |
| `train_ids_file` | 本机训练 ID 文件：每行一个 ID，或 JSON 字符串数组 | 未指定 |
| `eval_dataset` | SWE 服务机器上的评估 JSONL 路径 | 同 `dataset` |
| `eval_task_limit` | 不传评估 IDs 时取评估数据集前 N 题 | 同 `task_limit` |
| `eval_instance_ids` | 逗号分隔的评估 instance ID | 未指定 |
| `eval_ids_file` | 本机评估 ID 文件，格式同上 | 未指定 |

未传任何 `eval_*` 选题参数时，评估完整沿用训练选题。
一旦传入任一评估选题参数，就独立选择评估题目。
显式 IDs 决定实际题数，数量参数只用于未指定 IDs 的选择；同一侧 IDs 列表与文件互斥。
ID 必须准确存在于对应数据集中；重复、缺失或带首尾空白的 ID 会报错，不自动修改。

```bash
# 训练前 10 题，评估仍用相同 10 题（默认行为）
./script/4b_train_qwen36.sh task_limit=10 harness=claude_code

# 分别指定题目；其余训练参数可继续追加
./script/4b_train_qwen36.sh \
  train_instance_ids=astropy__astropy-12907,astropy__astropy-13033 \
  eval_instance_ids=astropy__astropy-13236,astropy__astropy-13398 \
  harness=claude_code eval_async=True

# 从本机 ID 文件选择；两个文件可以包含不同题目
./script/4b_train_qwen36.sh train_ids_file=/path/train_ids.txt eval_ids_file=/path/eval_ids.txt

# 指定独立评估数据集，并分别限定题数
./script/4b_train_qwen36.sh task_limit=20 eval_dataset=/host/eval.jsonl eval_task_limit=10
```

选题在 prepare 时固定到 `tasks.private.jsonl` / `eval_tasks.private.jsonl`，
后者仅独立指定评估选题时生成。`tasks.json` 公开训练/评估 ID、重叠情况和散列。
baseline 与最终评估始终使用相同评估题目。异步服务读取共享目录中的固定记录，
无需在评估机再次提供 ID 文件；改变选题必须使用新的 run 目录。
比较报告区分训练集诊断、无重叠评估和部分重叠评估，不能仅靠更换文件名声称泛化。


`eval_at_start=True` 可用于启动参数，表示执行独立底座 baseline。
`eval_async=True` 时由另一台评估机执行，不阻塞训练；同步模式先跑 baseline 再训练。
生成的训练配置内部仍是 `eval_at_start=False`，避免另起内部评估或切分训练题目。
本前后比较配方默认就包含独立 baseline，因此启动参数为 False 或省略时也保留 baseline。


SWE10/4b 启动脚本默认从项目 `.secrets/swe_token` 读取 SWE 服务 token，
该目录已被 Git 忽略，文件权限应为 0600。可用
`token_file=/absolute/path/to/token` 或环境变量 `SWE_TOKEN_FILE` 覆盖。
只保存服务端已有的 token，不生成新值；token 内容不要写进命令或实验配置。
跨机器只同步 Git 代码不会携带此文件，需要单独复制。异步评估机路径不同时
可设置 `SWE_EVAL_TOKEN_FILE=/local/path/to/token`。

## W&B 与实验目录命名

训练默认启用 W&B，统一项目名 `ARL`。SWE/4b 启动脚本的实验名取最终
run 目录名，不取内部 `train/` 子目录名。底层 GRPO 及其他训练预设的 W&B
默认项目也统一为 `ARL`。

不指定 `run_dir` / `log_path` 时，默认路径格式：

```text
runs/<模型简称>_<MMDD>_v<N>
runs/qwen36_1116_v1
```

模型简称来自 `POLICY`：Qwen 按系列/版本命名，例如 Qwen3.6-35B-A3B → `qwen36`、Qwen3.5-9B → `qwen35`。
其他模型使用规范化的模型目录名。模型完整路径和训练选项保留在配置中。
同一模型同一天从 v1 开始；目录已占用时递增到 v2、v3，原子创建避免并发启动撞名。
`check=True` 仅预览可用目录，不占用版本；真正启动时版本可能因其他任务占位而递增。
显式指定路径不会自动递增，路径支持 `%m%d` / `%Y%m%d` 日期占位符，例如：

```bash
./script/4b_train_qwen36.sh log_path=./runs/qwen36_swe_%m%d_v3
```

此时 W&B 实验名为 `qwen36_swe_1116_v3`，project 为 `ARL`。
训练指标继续保留在 `<run_dir>/train/metrics.jsonl`，W&B 本地记录位于该 train
目录下；`wandb=False` 可关闭 SWE 启动脚本的上传。W&B 登录凭据沿用本机 SDK
配置或环境变量，不写入实验配置。初始化/上传失败时保留本地指标并打印告警。
异步评估指标写入 `<run_dir>/async_eval/metrics.jsonl`，并沿用 ARM 的方式上传到同项目的
`<训练名>-eval` W&B run。各 checkpoint 使用固定 run ID 续写，横轴为训练 step
（baseline 为 0，第 N 批 checkpoint 为 N-1）。重评早于 W&B 当前 step 的点不会覆盖
历史曲线，以本地 metrics.jsonl 为准；上传失败不影响评估结果及完成标记。


## 总上下文与每轮生成长度

SWE/4b 入口已移除独立 `prompt_budget` 限制，以及后期生成放宽机制。
现在只配置 `max_model_len`（总输入+输出窗口）与 `max_tokens`（每轮生成总上限，含思考）。
`max_model_len=0` 默认读取原始模型 config.json 的 max_position_embeddings（支持 text_config）。
也支持 `sampling_max_model_len` 作为 max_model_len 的别名。

```bash
./script/4b_train_qwen36.sh max_model_len=262144 max_tokens=40960
```

Qwen3.6-35B-A3B 本地配置上限为 262144，因此上述配置每轮最多允许
221184 输入 token，为 40960 的生成上限留足空间。不再触及 32768 就压缩历史；
只有实际输入超过总窗口减去生成上限时，才按 harness 现有逻辑压缩，无法容纳则结束轨迹。
训练、baseline、最终评估与异步评估统一使用同一个窗口和生成上限。
内部配置中的 `prompt_budget_tokens` 为总窗口推导的可用输入空间，不是额外可调限制。
不能把总窗口设到模型支持上限以上；更大的窗口和更长的实际历史会增加资源需求。

`max_tokens_late`、截断率触发切换的相关参数，以及 SWE 入口的 `prompt_budget` /
`prompt_budget_tokens` 已不再接受。旧命令应删除这些参数；每轮始终使用 `max_tokens`。
`think_budget` 继续限制思考部分，包含在每轮 `max_tokens` 内。

## 固定 prefill 调度预算及 4×TP4

SWE/4b 支持 `prefill_tokens` 和 `prefill_batch_tokens`（也接受大写同名键）。

```bash
./script/4b_train_qwen36.sh \
  COLOCATE=1 TRAIN_GPUS=all \
  SAMPLER_GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 SAMPLER_DP=4 \
  prefill_tokens=1056 prefill_batch_tokens=2048
```

上述16张采样卡分成 `[0–3]`、`[4–7]`、`[8–11]`、`[12–15]` 四个副本，每副本TP4。
各副本使用相同配置，独立连续批处理；新轨迹按负载分配，后续轮次固定返回同一副本。
`check=True` 打印 `sampler_tp` 和四组 `sampler_replicas`，仅预检，不启动模型。
单进程独立baseline/final评估现在也遵循显式SAMPLER_GPUS池与SAMPLER_DP；
多进程旧式推理仍沿用各rank的KERNEL_RL_SAMPLER_TP设备切片。

- `prefill_tokens=1056`：每个引擎每次调度的总token上限（prefill和decode共享）。
- `prefill_batch_tokens=2048`：vLLM编译/预热的max_num_batched_tokens，保持与调度上限独立。
- 两者默认0，保持vLLM默认。只设prefill_tokens时，编译预算自动为max(2048,prefill_tokens)。
- 显式编译预算不能小于运行时预算；启用Mamba align时，预算必须是实际缓存块大小的整数倍。

此预算不限制上下文总长、单次回复长度或轨迹轮数。并发请求共享预算，末尾不足一个块，
所以固定的是每引擎总调度上限，并不保证每条请求每次都得到1056token。
这也不保证所有并发安排、TP和硬件下greedy逐位相同。

通过项目内BudgetAsyncScheduler接入vLLM scheduler_cls，保留async scheduling；
预算写入additional_config，支持spawn并与已有fixed RMSNorm设置合并，不修改共享vLLM。
引擎启动日志打印 `[prefill]`、实际block_size和两个预算。Mamba/GDN与全注意力缓存页需对齐，
block_size随模型、dtype、TP变化；1056来自先前TP2实验，不是通用常数。

训练和独立评估配置都会冻结这两个参数，异步评估从共享spec继承预算。
若评估机也要求四个TP4，给异步服务分配16卡：

```bash
./script/eval_swe10_async.sh add /shared/run_dir GPUS=auto NGPUS=16 SAMPLER_DP=4
```

只加调度预算不会自动启用prefix caching；此前缓存开关实验保持独立，生产默认未改变。

本机Qwen3.6-35B-A3B配置预检确认：TP2的缓存块为1056token，TP4为528token。
因此四个TP4副本设置prefill_tokens=1056，相当于每引擎两块预算；528/1584也分别是一块/三块。
模型加载配置会根据KV页与GDN状态大小自动推导该数值，并非通用常数。
对齐模式在末尾还可能单独切到最后一个完整块边界；即便预算相同，缓存开/关的
实际分块也不能一概假定一致。要比较数值，应固定TP、缓存模式、并发条件并实测。

## 开启跨轮 prefix caching

SWE/4b 正式开放以下两个 `key=value` 参数，训练、baseline/final和异步评估均生效：

```bash
enable_prefix_caching=True mamba_cache_mode=align
```

与四个TP4副本和固定调度预算组合：

```bash
./script/4b_train_qwen36.sh \
  COLOCATE=1 SAMPLER_GPUS=all SAMPLER_DP=4 \
  prefill_tokens=1056 prefill_batch_tokens=2048 \
  enable_prefix_caching=True mamba_cache_mode=align
```

16张可见采样卡对应4×TP4；每个副本均收到同样的缓存配置。
`enable_prefix_caching` 省略时保留vLLM/模型默认（当前Qwen混合模型默认关闭），
不会因开放参数自动开启。`mamba_cache_mode` 接受none/align/all；当前Qwen不支持all，
启动预检会要求改用align。显式align/all要求enable_prefix_caching=True。
none为vLLM原始默认模式；显式开缓存后vLLM可能按模型将其自动转换为align，建议Qwen直接明确写align。

关闭时使用 `enable_prefix_caching=False mamba_cache_mode=none`。
参数与prefill预算一起写入阶段JSON及异步spec，不依赖评估机继承训练机shell环境。
更改配置请使用新run目录；权重同步与共卡切换沿用已有cache reset/清理流程。
此前实验验证了固定权重复用与模拟权重更新后清缓存，尚未验证完整GRPO共卡训练生命周期。

显式 `enable_prefix_caching=True` 时，更新/恢复阶段若 `reset_prefix_cache` 返回失败，
ARL会停止并报错，避免继续使用旧策略KV；不再只打印告警后继续。四个副本中任何一个
清理失败都会向上传播。共卡切换保持pause+clear→KV unmap→映射→resume→reset顺序。
四个TP副本的前缀缓存相互独立；目前已接入轨迹级副本亲和性，新轨迹按负载分配，
后续轮次与思考/回答两段固定在原副本。KV仍可能被驱逐，权重更新仍会清理缓存；
因此单引擎探针的提速不能直接当作四副本训练的整体提速。


## 跨轮副本亲和性

Claude Code和mini-swe-agent通过共享rollout自动启用轨迹级副本绑定，无需额外命令参数。
同一轨迹的每轮请求固定返回同一副本；一轮的思考/回答两段也固定在同一引擎。
新轨迹优先选择在飞请求少的副本，平局时选择已绑定轨迹数少的副本。

绑定键包括采样客户端namespace、来源rank、rollout scope和轨迹ID，避免不同rank的
同名轨迹串绑。scope在正常结束、异常或取消时释放，并取消属于该scope的未完成请求，
不增加分布式collective；后续批次重新分配，不累积旧绑定。
保持的是副本归属，不保证缓存永远命中：显存压力下KV仍可能被驱逐，更新权重仍必须清缓存。
轨迹不会为了临时负载变化迁移到别的副本；这也意味着长轨迹可能造成后期副本负载不齐。
