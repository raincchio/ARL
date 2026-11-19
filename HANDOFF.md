# ARL 交接记录

更新日期：2026-11-10。工作目录：`/root/workspace/ARL`。

## 用户最新约定

- 环境分为 `coding` 和 `instruction_following`，SWE 和 CUDA 算子都属于 coding。
- collect / train / evaluate 分别选择模型、env、harness，启动前校验配套能力。
- 训练可能同时指定多个环境，使用 `train.environments`；共享模型与优化器，每个全局批次按权重选择一个环境，指标与评估按环境分别记录。
- **arl_agent 仅用于 CUDA 算子优化 / KernelBench，不允许用于 SWE、ProgramBench、IFEval。**
- SWE 默认 `claude_code`（真实 CLI 2.1.226 + 本地 Qwen token 桥 + 远程 MCP bash），可显式选 `mini_swe_agent`；ProgramBench 默认 `mini_swe_agent`。不能回退到 arl_agent。
- IFEval 当前使用内部 `direct` 单轮生成适配。用户刚问过业界通常使用什么 harness：回答为常见 lm-evaluation-harness 或 Google 官方评分脚本；**本仓库尚未接入 lm-eval，用户尚未明确要求实施该接入。**
- 用户选定模型：`/models/Qwen/Qwen3.6-35B-A3B`。最初给的是 `/model/...`，实际可用路径为 `/models/...`。

## 当前真实状态

| 环境 | 默认 harness | 状态 |
| --- | --- | --- |
| coding.cuda | arl_agent | 原生采集、GRPO、评估已接通 |
| coding.swe | claude_code | 真实 CLI、本地 token 采集/GRPO/评估已接；CPU 更新和远程官方负例验证通过，当前 harness 的 GPU 训练未实跑 |
| coding.programbench | mini_swe_agent | 上游 agent、本地采集/GRPO/评估已接；CPU 链路和远程 Docker 判分验证通过，GPU 训练未实跑 |
| instruction_following.ifeval | direct | 单轮生成、官方判分、采集和 GRPO 已接通 |
| instruction_following.multicodeif | multicodeif_self_repair | 本地 token 采集、自修复 GRPO、官方约束评估已接；API/离线文本接口保留 |

arl_agent 继续只服务 KernelBench。SWE/ProgramBench 使用独立的 mini-swe-agent 上游 DefaultAgent，复用公共 token 采样与 PPO 数据处理，不调用 arl_agent。文档若仍有历史描述，以注册信息、兼容性检查和本交接为准。

Codex、Claude Code 真实运行适配仍 pending。mini-swe-agent 已固定 2.4.6 并接通。新建的 direct 不是第三方框架，也不是改名后的 arl_agent：入口独立，仅复用公共 token 渲染与采样组件。

## 核心代码与配置

- `arl/envs/registry.py`、`arl/envs/{coding,instruction_following}/__init__.py`：环境注册、默认/支持的 harness。
- `arl/harnesses/catalog.json`、`compatibility.py`：harness 状态、操作、模型后端与 token 轨迹能力检查。
- `harnesses/arl_agent/entry.py`：额外限制 env_name 必须为 coding.cuda。
- `harnesses/direct/entry.py`：IFEval 单轮生成入口。
- `harnesses/mini_swe_agent/entry.py`、`arl/harnesses/mini_swe.py`：上游 DefaultAgent 的 token/workspace bridge；依赖缺失或版本不符在预检报错。
- `arl/experiments/`：YAML/JSON 配置、阶段执行、身份与数据导出。
- `arl/envs/mixture.py`：确定性加权批次计划、成员配置及 SourcePool 清理。
- `arl/grpo.py`：共享模型/优化器、多环境批次切换、各环境独立的生成上限扩容状态及周期评估。
- `experiments/{ifeval,swe,programbench,mixed,mixed_programbench,multicodeif}.yaml`：示例配置。
- 含 SWE/ProgramBench 的混合示例已通过预检。SWE+IFEval 混合驱动、共享优化器、每步采样参数同步调用已通过 CPU 联调。

混合权重表示批次选择概率，不是奖励或 loss 的乘数。数据耗尽后重新打乱循环使用；一个混合 epoch 默认包含 ceil(各环境训练题数总和 / batch_size) 个批次，也可配置 mixture_steps_per_epoch。恢复时按同一计划用 start_batch 跳过前缀。

## 数据划分

- SWE 示例：Verified 500 条公开题目内部切为 450 train / 50 held-out，seed 默认 0。
- IFEval 示例：541 条公开题目切为 487 train / 54 held-out，test_fraction=0.1、ifeval_split_seed=0。
- CUDA 启动配置：留出 2%，最多 64 条。裸 GRPO 通用 test_fraction 默认 0.02，示例可覆盖。
- ProgramBench 全量 200 题，环境配置默认 10%；首轮示例仅选 3 题，test_fraction=0.34，切为 2/1。
- 指定独立 test_set 时不再自动按比例划分。SWE、IFEval、ProgramBench 有重叠检查，CUDA 原适配尚未补同等检查。
- SWE/CUDA 先应用 task_limit 再划分；IFEval/ProgramBench 先划分再应用限制。调整 SWE task_limit 可能改变 held-out 身份。
- collect/train 默认用 train；evaluate 默认用 test。相同数据和配置保持固定划分，不随 epoch 重切。
- 这些是公开 benchmark 内部训练实验划分，不能把训练后的结果称为未见过该 benchmark 的全量评测成绩。

## IFEval

- 环境：`envs/instruction_following/IFEvalEnv/`；适配：`arl/envs/instruction_following/ifeval.py`。
- 官方检查器固定 commit：`e6890f85757dd84e27ca6df2dd30651dafad28e0`，文件哈希在 upstream.json。
- 官方源码、数据、隔离 venv 和 NLTK 资源位于 `.cache/benchmarks/`。
- 默认 strict 全约束通过得 1，否则为 0；可选 fraction，通过比例为奖励，但 correct 仍表示全部 strict 约束通过。
- 已验证官方检查器正负判分、token/logprob/mask 导出及 GRPO 正负优势；未启动正式 GPU 训练。

```bash
PIP_INDEX_URL=https://pypi.org/simple ./script/setup_ifeval.sh
python3 -m arl.benchmarks ifeval
python3 -m arl.experiments experiments/ifeval.yaml --phase train --check
```

标准评估可进一步接 lm-evaluation-harness，训练仍需本地 on-policy token 轨迹；不要把 lm-eval 和优化器训练路径混为一体。

## ProgramBench

- 环境：`envs/coding/ProgramBenchEnv/`；适配：`arl/envs/coding/programbench.py`。
- 默认 mini_swe_agent 已接通；arl_agent 被禁止。`arl.experiments` 的训练入口可执行；`arl.benchmarks` 是独立离线数据/提交包接口，其 training_status 仍 pending，不能用后者代替实验训练入口。
- 固定官方 v1.2.5，commit `27f02157c785f8da3647aa6dbbe6b9137f99f10e`，镜像 `task_cleanroom_v6`。
- 启动时校验官方 Python 源码及 200 题 task.yaml/tests.json 哈希，排除 testorg 测试夹具。
- 根据参考 executable 和文档从零重建代码，归档完整工作区后调用官方 `_evaluate_instance`。不自行重写 ignored tests / branches 的评分规则。
- 断网 Docker，agent 用户，在 /workspace 操作；不挂载宿主机源码、Docker socket 或隐藏测试。
- 默认奖励为官方有效测试通过比例；strict 可选。compile_failed/copy_executable_failed 记 0；测试缺失、分支/system 错误、超时及分数不一致记 infra 并丢弃。
- 当前没有接官方自动作弊审计，行为测试分数不能直接等同于完成 leaderboard 合规审核。
- 提交包离线判分不执行 agent，报告明确记录 harness_execution=not_run_offline_grading，可独立检查数据或评测已有 archive。

复用的远程资源：

```text
SSH: root@10.0.134.90:22
Python: /mnt/ai_data/benchmarks/programbench/.venv/bin/python
Blobs: /mnt/ai_data/benchmarks/programbench/blobs
Work directory: /mnt/ai_data/arl/programbench_runs
Blob revision: de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5
```

本机与远程任务清单均读取到 200 个任务；远程仅部分镜像已准备。预检缺少镜像/blob 会报错，不自动拉取全量。

真实 Docker 冒烟均通过，且已清理本次容器：

- 编译失败提交：0 分，非 infra。`runs/programbench_smoke_20261109_080327/result.json`。
- 能编译的占位程序：429 项行为测试，通过 35 项，score=0.08158508158508158，非 infra。`runs/programbench_smoke_20261109_080842/result.json`。
- 以上都是合成联调，**不是模型成绩**。

tar 读取参考 executable 可能因权限返回非零。实现遵循现有 baseline：只要产生归档就继续官方评测，保留 archive_warning，不因该警告直接记 infra。

```bash
PIP_INDEX_URL=https://pypi.org/simple ./script/setup_programbench.sh
python3 -m arl.benchmarks programbench
python3 -m arl.benchmarks programbench --config envs/coding/benchmarks/project_generation/programbench/remote_inspect.json
python3 script/smoke_programbench_env.py
python3 script/smoke_programbench_env.py --mode stub
```

后两条需要远程 Docker，会执行真实编译/测试，但不启动模型。

## SWE 与 MultiCodeIF

SWE 已验证独立 Docker、bash/submit 工具、最终 diff 和官方 resolved 0/1 判分；默认 mini-swe-agent 已接运行/GRPO token bridge，GPU 训练尚未实跑。50 题耗时尚无完整实测。之前给出的约 20–80 分钟估算依赖 8 ranks × 每 rank 2 个工作区及假定单题 5–20 分钟，不能当作实测结果。

MultiCodeIF 固定上游 commit `faafada5356ce81d228793f3787b27615cb763bf`；实际读到 single_level 1526 条、multi_level 660 条（165 条链）。已接官方约束检查、API/离线预测和自修复，文本轨迹可供后续训练数据构造，现已补齐本地 GRPO token/logprob/mask、优化器与参数同步路径。不能说数据集本身不支持训练。

此前 Qwen3.6 的 MultiCodeIF 22 题试跑已完成并清理服务：20 题有评分，最终通过 8，2 题 generation_error，不作为模型失败；只评约束，不评代码功能。结果在 `runs/arl_multicodeif_qwen36_20261109_052248/`。

## 建议下一步

1. mini_swe_agent 的 ProgramBench/SWE collect、evaluate 和 GRPO 数据路径已接通。下一步可按用户需要通过既有 GPU 队列进行模型实测。
2. 可参考现有 `/root/workspace/AgentBench/agentbench/host/pb_agent.py`、`pb_trial.py`、`pb_grade.py`；不要修改其他项目的共享运行环境或已有任务。
3. 保留实际 harness、模型、环境身份、消息与轨迹；不通过改名字调用 arl_agent。
4. mini-swe-agent 的 token/logprob/mask 由真实本地采样保留；共享优化器与每步参数同步调用已通过 CPU 联调，但不等于真实 GPU 优化器训练已经验证。
5. 若用户确认要统一 IFEval 标准评估，再接 lm-evaluation-harness；当前只是讨论了常见做法。
6. 检查并清理文档中的历史能力描述，防止与最新 harness 边界矛盾。

## 验证和工作区注意事项

- 最新完整回归及 ProgramBench 对齐验证见文末最新记录。
- IFEval direct 训练配置预检通过。SWE/ProgramBench 单环境与混合配置也已通过预检。
- 没有启动正式 GPU 训练。ProgramBench Docker 联调不使用 GPU。
- 改动尚未提交，工作区包含多轮累计修改；**不要整体回滚、清理未跟踪文件或恢复旧 envs/CudaEnv 路径**。环境此前迁移到了 envs/coding/。
- 保留原有未跟踪的 `script/train_cuda_grpo_qwen35_9b.conf` 和 `.sh`，不要误当成本次新增文件清理。
- `.cache/benchmarks/`、runs/ 已忽略，不提交官方代码缓存、依赖或运行产物。
- GPU 任务遵守 `/root/workspace/MACHINES.md`，通过既有 Hybrid1 队列调度，不能手动抢卡，不能占用 CudaEval 评测节点。不擅自启动整个调度器或他人的排队任务。

## 本次 mini-swe-agent 对接（2026-11-09）

- 固定上游 2.4.6，安装到 `.cache/benchmarks/mini-swe-agent`，不改共享 Python；重建环境使用 `bash script/setup_mini_swe_agent.sh`。
- 执行真实 `DefaultAgent.step()`、上游动作校验和提交检查。模型请求由本地 token 采样器提供，工作区及官方判分沿用现有环境。
- 初版使用 ARL 环境提示词和文本观察；SWE、ProgramBench 随后均按用户要求切换各自的官方配置，以下面的最新记录为准。
- 支持 local collect/train/evaluate；尚未提供 mini 的 OpenAI-compatible API 后端。GRPO 的 token/logprob/mask、正负优势、混合训练共享优化器及更新后同步调用通过 CPU 测试。
- ProgramBench 真实远程 Docker + mini 冒烟：`runs/programbench_smoke_20261109_090641/result.json`，compile_failed，0 分，infra_error=false，Submitted，上游 agent 类型与完整消息已记录。
- SWE 真实远程 Docker + mini 冒烟：sympy__sympy-21847 合成负例，官方 resolved=false、reward=0、infra_error=false，原有通过测试保持通过；容器已清理。报告在 `/mnt/ai_data/arl/swe_runs/arl-swe-fcb67228f0e04d36b4452ff5c836be81/logs/run_evaluation/arl-swe-fcb67228f0e04d36b4452ff5c836be81/arl/sympy__sympy-21847/report.json`。
- 以上是合成联调，不是模型成绩；未启动 GPU 模型、正式训练或新队列任务。
- 最终预检：swe、programbench、ifeval 各 collect/train/evaluate；mixed、mixed_programbench 的 train/evaluate；multicodeif 的 collect/evaluate，合计 15 个阶段配置全部通过。模型路径 `/models/Qwen/Qwen3.6-35B-A3B` 已确认存在。

## SWE 对齐官方配置（用户要求：对齐）

- 直接加载 mini-swe-agent 2.4.6 包内 `config/benchmarks/swebench.yaml`，SHA256 固定为 `9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e`。system/instance 提示词与上游 `DefaultAgent.run(problem_statement)` 逐字比对通过，不再添加 ARL 提示词。
- 官方 bash schema、结构化工具消息、长输出及格式错误模板；Qwen 渲染视图把 API 的 JSON 字符串参数转换为字典，上游轨迹仍保留原格式。
- 按官方提示词创建/查看 patch.txt，再执行 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`。仅评分 stdout 中的 patch；空提交或上限结束不再自动收集工作区 diff。原生环境 submit 接口保留用于底层测试，但默认 SWE mini 执行路径不调用它。
- SWE 单环境、混合示例及 shell 训练配置统一为 250 步 / 命令 60 秒；命令 interpreter/env 从官方 YAML 读取，包括 BASH_ENV。实际 env_max_turns / timeout 覆盖记录到轨迹。后续已改为未指定时使用环境默认（SWE 250）。
- 保留本地差异：本地 token 采样与 Qwen chat template；无 API 计费，cost_limit=0（上游为 $3）；生成及上下文上限可配置，保留现有 Docker 隔离。evaluate 示例 temp=1 / top_p=1 / top_k=-1，上游 v2 API temperature 留空。不能声称所有模型服务默认值/完整榜单运行完全一致。
- 全量 142 项测试通过；实际 Qwen3.6 tokenizer 验证工具调用后第二轮渲染成功，1318→1379 tokens，消息只新增一个工具调用。记录在 `runs/swe_official_alignment/tokenizer_check.json`。
- 远程真实 Docker 官方提交/判分冒烟再次通过：sympy__sympy-21847 合成负例，输出 patch 181 bytes，resolved=false，infra_error=false；报告 `/mnt/ai_data/arl/swe_runs/arl-swe-78b8de8fbd7849f0ba666368f6d792e7/logs/run_evaluation/arl-swe-78b8de8fbd7849f0ba666368f6d792e7/arl/sympy__sympy-21847/report.json`。不是模型成绩，未启动 GPU。

## ProgramBench 对齐官方配置（用户要求：对齐）

- 直接加载 mini-swe-agent 2.4.6 的 `config/benchmarks/programbench.yaml`，SHA256 `9c2cc36b69c53e83cb4f60b0aa809768996373f4c3dddb3a202b78e93899bf8e`；配置缺失/被修改时预检拒绝执行。
- system / instance 提示词与上游 DefaultAgent.run() + DockerEnvironment 模板变量逐字比对通过。system_information 使用上游同源的控制端 platform.uname()，不添加 ARL 任务说明。命令使用官方环境变量、bash -lc，启动前初始化上游 Git 用户和邮箱。
- 使用官方结构化工具反馈、格式错误模板、长输出首尾截取、剩余步数及时间提醒；ProgramBench 轨迹按上游行为移除重复 raw_output。本地 Qwen 渲染视图仍转换参数字典，采样 token/logprob/mask 原样保留。
- ProgramBench 单环境 / 混合 / shell 部署示例改为 1000 步、6 小时 agent 时限、命令 180 秒、20 CPU / 60GB、容器 7 小时。环境级 step/timeout/resources 可显式覆盖并记入轨迹；后续已改为未指定时使用环境默认（ProgramBench 1000）。
- 按独立 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 命令结束；上限退出也归档完整工作区并调用已有官方 v1.2.5 评分器，不使用 SWE 的 stdout patch 提交规则。模型仍应按官方提示词提交 Git 改动和提供 compile.sh。
- 全量 147 项测试通过，包括 ProgramBench 官方提示词、平台变量、工具反馈、期限/步数结束归档、worker 超时/完整输出与采集训练轨迹。SWE 与既有混合训练回归也通过。15 个实验阶段配置预检全部通过。
- 实际 Qwen3.6 tokenizer 渲染验证通过，2107→2168 tokens，工具消息只新增一个工具调用；结果 `runs/programbench_official_alignment/tokenizer_check.json`。
- 真实远程 Docker 编译失败负例通过：`runs/programbench_official_compile_failure/result.json`，compile_failed、reward=0、infra_error=false，profile=upstream_programbench。冒烟显式覆盖步数为 4，未运行模型/GPU，不能当作模型成绩。
- 真实远程 Docker 可编译占位提交也通过：`runs/programbench_official_stub/result.json`，官方 429 项行为测试通过 35 项，score=0.08158508158508158、infra_error=false、Submitted；实际 20 CPU / 60GB，步数显式为 4。对应 `arl-pb-93e6b07bd16142e684f015644ecd0b4a` 的 agent/评分容器已确认清理。这是合成提交，不是模型成绩。
- 本地差异仍包括 Qwen token 采样、生成/context 上限、显式采样参数和更严格 Docker 隔离；默认评分 CPU/分支并发可配置。官方自动作弊审计尚未接入，行为分数不是完整 leaderboard 合规审核。

## 环境与 GRPO 统一闭环（2026-11-10）

用户明确目标：所有当前环境都由当前模型在各自 harness 下生成用于 GRPO 的 on-policy 数据，再更新同一模型。当前注册的五个环境已全部接入该路径，其他仅有 benchmark 目录的项目仍不算已接入环境。

- 配套：CUDA → arl_agent；SWE/ProgramBench → mini_swe_agent；IFEval → direct；MultiCodeIF → multicodeif_self_repair。arl_agent 的 CUDA 专用边界保留。
- 补齐 MultiCodeIF 的本地环境 `arl/envs/instruction_following/multicodeif.py`：固定上游检查器、官方修复模板、本地 token 采样、strict/fraction 终局奖励。各轮 datum 共用 trajectory unit，按同题不同轨迹做 GRPO 组内基线；反馈与旧代码在新 prompt 中 mask=0，只有真实采样动作训练。
- MultiCodeIF 保留 API/offline 文本接口。EnvironmentSpec.backend_interfaces 按后端选择接口；API/offline 不声明训练 trace，明确禁止作为 GRPO 后端。原 `experiments/multicodeif.yaml` 继续用于 API；新增 `experiments/multicodeif_grpo.yaml` 用于本地 collect/train/evaluate。
- 约束检查缺失/不完整/错误时丢弃整条轨迹；数据按任务或多层约束链固定划分，显式双文件检查链级重叠。启动前探测所选检查类别，依赖缺失/缺少 judge 明确报错。训练示例使用已准备的 Python 命名/数据结构规则；不评代码功能/编译。
- 新增统一边界 `arl/algorithms/rollout_data.py`：检查 token/logprob/mask、返回数组长度、同题分组、轨迹 unit、有限回报与正确性标签。缺失/不齐/非有限 logprob 或非法 mask 在执行工具前丢弃；采样中途失败也丢弃之前的轨迹片段，不能把故障当模型负例。
- train 在线循环仍是 rollout → 终局 reward → finalize_ppo_data → train_step/PPO loss → optimizer → 参数同步 → 下一批。collect 的导出数据用于审计/外部数据处理，默认 train 不重放历史 datums 或纯文本回答。
- env_max_turns 默认改为 None，继承环境默认 CUDA 16 / SWE 250 / ProgramBench 1000；显式设置保留。IFEval 固定单轮；MultiCodeIF 使用 multicodeif_max_rounds 默认 3。修正非 CUDA 环境显示 CUDA 性能指标的问题。
- 新增 `experiments/all_environments_grpo.yaml`，五环境按权重选择全局批次、共享模型与优化器，每批有效更新后同步参数。GPU 部署仍必须走既有队列；本次没有启动 GPU 作业。
- 验证：完整 `python3 -m unittest discover -s tests -p 'test_*.py' -q`，157 项通过。五个真实环境类与真实 harness 在同一 GRPO 驱动中完成 20 次真实 CPU autograd/Adam/PPO 更新，21 次参数快照（初始 + 每次更新）；逐条检查下一批使用相应更新版本。CudaEval/Docker/IFEval 服务为测试后端，MultiCodeIF 使用固定官方检查器。这不是 Qwen GPU 模型训练或模型成绩。
- 单独 MultiCodeIF CPU 策略完成 2 次真实 GRPO 更新；官方两轮失败→修复通过、组内正负优势、API/本地路由及导出训练 token 均有回归覆盖。
- 实际 Qwen3.6 tokenizer + 脚本生成 + 官方检查器冒烟：4 个 datum，returns=[1,1,0,0]、units=[(0,0),(0,0),(1,0),(1,0)]、每段 13 个 action tokens，正负优势均存在。结果 `runs/environment_grpo_alignment/multicodeif_token_smoke.json` 和同目录 trajectories/；未运行模型推理。
- 已检查既有单环境/API/混合配置和新增配置；五个注册环境的 local train 配套均包含 token_ids/logprobs/loss_mask。

## 2026-11-13 SWE 宿主机 master / 两 worker 落地

按用户要求采用训练容器内运行 ARL/vLLM、宿主机提供容器服务的结构。`.70:18770` 为 master，`.70:18771` 与 `.71:18771` 为 worker；systemd 单元分别为 arl-swe-master 和 arl-swe-worker。复用宿主机 Docker，未采用 DinD，也未改 Docker data-root。两机合计最多 8 条未关闭轨迹，其中最多 2 条评分；每条轨迹固定到原 worker。

脚本与配置统一在 `script/swe_worker/`：cluster.py、cluster.local.json、prepare.py、cluster_smoke.py、ctl.py、README.md。cluster.py 支持 deploy、prepare、sync-images、status、sync-artifacts、stop；新增机器修改拓扑再 deploy --worker。当前 SSH 经 .70 使用其 host-to-host 凭证，不复制私钥。同步镜像支持共享 crane 缓存，或从 seed 通过 docker save/load 分发并核对 image ID，已实测实际传输和清理测试镜像。

共享缓存已从 xing 私有缓存复制到 `/root/share/swe/crane_cache`；宿主机对应 `/mnt/efs-013e7987/share/swe/crane_cache`。评分产物归档到共享 `/share/swe/runs/<service-id>/<workspace>`；各机本地状态在 `/mnt/ai_data/swe/service/state`。token 在 service/token，0600，不能写进日志或训练配置。中途遇到共享盘临时 ENOSPC，随后写入恢复；服务已增加本地结果保留、pending 标记及 sync-artifacts 补传，不能把归档失败当模型 reward=0。

验证：完整 unittest 176 项通过；两机真实同题工作区隔离、固定路由、官方正例 1 / 负例 0、共享产物归档和清理均通过；运行中取消/命令超时也通过。详细报告 `/root/ai_data/swe/client-smoke/cluster_validation.json`。

另用 Qwen2.5-0.5B，GPU0 训练/GPU1 采样，在单机服务阶段跑完 1 批真实交互、1 次有效 optimizer 更新、NCCL 同步及 checkpoint 保存：`/root/ai_data/swe/client-smoke/gpu_v3`。该 smoke 使用 optimizer offload 和辅助长度惩罚以验证非零更新，SWE reward=0，不是能力成绩；前一轮纯 SWE 小模型零优势触发梯度保护、未更新，记录保留在 gpu_v2。修复了单 rank 首次 NCCL 同步未初始化采样器的问题并添加回归测试。

正式大规模训练尚未启动。标准 Python SWE-bench 后端已接服务；ProgramBench/Pro 尚未接该服务。多租户鉴权、policy-version 固定、跨机器恢复半条轨迹和调度公平性尚未实现。部署会重启相应服务，应在训练停止后操作；查看 README 再扩节点。

## 2026-11-13 服务代码隔离（覆盖上节的源码路径）

用户要求源码只保存在 .70 宿主机并统一部署。独立项目现位于宿主机 `/root/workspace/SweContainerService`，包括 swe_service/、scripts/、configs/、tests/、README.md 和 swe-cluster 入口；没有 ARL 依赖。.70 master/worker 直接运行该项目源码；.71 仅接收 `/mnt/ai_data/swe/service/code` 运行文件，没有源码项目。旧 .70 `/mnt/ai_data/swe/{master,service}/code` 副本已清理，状态、token、镜像和日志均保留。

ARL 中移除了 service.py / master.py / worker.py、服务端测试和部署目录 `script/swe_worker/`。只保留 HTTP 客户端、环境和训练配置，以及 `script/swe_client/` 的客户端检查脚本。endpoint/token 不变。本地/SSH 兼容模式现在调用宿主机已有的 swe_worker_script，默认 .70 的 `/root/workspace/SweContainerService/swe_service/worker.py`，不再发送服务端源码；直接调用 .71 时需指定其 `/mnt/ai_data/swe/service/code/worker.py`。

在 .70 执行：`cd /root/workspace/SweContainerService && ./swe-cluster status|deploy|prepare|sync-images|sync-artifacts`。拓扑为 configs/cluster.local.json。独立服务测试 20 项通过，迁移后 ARL 全套 159 项通过；独立项目双机官方正负例评分 1/0、固定路由和清理通过，报告在宿主机 `/mnt/ai_data/swe/service/migration-validation.json`。ARL 客户端不读服务端源码的回归测试已加入。

## 2026-11-13 master 负载均衡与来源 IP 公平调度

用户要求 master 负载均衡，并按请求 IP 防止大量请求独占、排队时轮询 IP。已在 .70 的独立项目增加 swe_service/scheduling.py，更新 master.py；服务仍在 .70 统一管理，worker 分布 .70/.71。

创建工作区与评分分别使用有界队列：同 IP FIFO、跨 IP round robin；来源为 TCP peer，不信任 client_ip/X-Forwarded-For。默认每 IP 最多 4 个活跃/预留工作区、128 个待处理接入记录，全局 1024；普通 HTTP 每 IP 32 在途/100 请求每秒/突发 200，全局 256 在途。close 使用独立有界预算。排队返回 HTTP429/Retry-After=1，客户端同 ID 重试，30 秒无重试的未领取预留会释放。队列位置和统计在内存，重启不恢复排队顺序；已经分配的 worker 归属继续持久化，不迁移运行中的工作区。

worker 选择考虑镜像就绪、可达/指纹兼容、硬额度、占用率、评分/CPU/内存压力和 weight。master 计入尚未被 worker 快照观察到的预留并按 workspace 去重，防止并发突发超分配；相同压力的节点轮换。状态和镜像探测短暂缓存并合并，库存缓存有界。配置在宿主机项目 configs/cluster.local.json 的 scheduling 与 worker.weight。add-worker 已支持 --weight，且有排队/预留时拒绝部署。

ARL 客户端仅更新了明确 429 的安全重试：exec/patch 也可能在执行前被 IP 预算拒绝，可重试；网络错误不重放。训练端不包含服务实现。

验证：宿主机服务端 53 项测试、ARL 全套 160 项测试通过。使用两个真实本地 TCP 源地址创建同一镜像的 8 个工作区，两机各 4 个，每 IP 各 4 个；验证跨 IP 不饥饿、同 IP FIFO、预留、固定路由和清理，报告在宿主机 /mnt/ai_data/swe/service/fairness-validation.json。官方双机正负例再次得到 1/0，报告 balancing-grade-validation.json。独立源 IP 压测脚本 scripts/check_fairness.py；规则和默认额度详见 docs/BALANCING.md，镜像多容器说明补入 docs/IMAGES.md。未启动正式大规模训练。

## 2026-11-13 集群 HTTP 状态页

已在 .70 master 的现有 18770 端口增加浏览器状态页 GET /、认证 JSON GET /api/status（别名 /status）、轻量存活 GET /healthz。原 POST /rpc 保持兼容。网页外壳公开，集群数据使用原 Bearer Token；Token 只在页面内存，不写 URL/cookie/本地存储。master 新增 status_view 只读动作，不因 GET 观测调用 schedule 分配等待任务，已有 RPC status 的调度语义保留。

独立源码在 .70 项目 swe_service/monitoring.py 与 web/{index.html,status.js,status.css}；只重启了 master，worker 未重启。页面每 2 秒刷新 worker 状态/压力/容量、各 IP 的活跃/排队/预留/请求统计、活跃任务及来源 IP。新增 observed_at 与 snapshot_age_seconds；活跃任务记录增加 client_ip。使用 CSP、自有静态资源、textContent 渲染和 no-store，日志不记录 URL query。

验证：服务端 63 项测试通过，包括认证、只读、GET/HEAD、旧 POST RPC、资源白名单、请求预算；JavaScript 在 V8 中检查语法、DOM 绑定、表格渲染与认证失败复位。真实 HTTP 创建了临时工作区并验证任务及来源 IP 出现、清理后消失，两 worker 在线。记录在宿主机 /mnt/ai_data/swe/service/http-status-validation.json。说明在 .70 项目 docs/HTTP_STATUS.md，README 已加入口。


## 2026-11-16 参考 swe-qwen-collect 对接 SWE env

- 对照 `/root/share/swe-qwen-collect/collect.py` 的 HTTP `/rpc` 协议，复用现有 `SweTaskSource` / `DockerWorkspace`。默认 `script/train_swe_grpo.conf` 及 swe/mixed/mixed_programbench/all_environments_grpo 实验配置从旧 .90 SSH 切到 `http://10.0.134.70:18770`；token 文件 `/root/ai_data/swe/service/token`，远程 Verified 数据路径不变。local.conf 为默认配置的兼容入口。
- 保留 mini_swe_agent 2.4.6 官方提示词、stdout patch 提交与 GRPO token/logprob/mask 链路，不引入示例采集器的 JSON command/done 提示词。
- 底层 SWE 27 项、mini-swe-agent 14 项、实验阶段 8 项测试通过；SWE 三阶段与三份混合训练配置共 6 项预检通过，shell 语法与 smoke Python 编译通过。沙箱内线程/网络操作挂起，mini 和实验回归及真实连接在获准的沙箱外执行。
- 真实 `script/smoke_swe_env.py --harness mini_swe_agent --negative` 通过：`sympy__sympy-21847`，workspace `arl-swe-b07394e405b44a5fbf3e482fffde4b7f`，worker `node92`，patch 181 bytes，reward=0，infra_error=false；FAIL_TO_PASS 0/1，PASS_TO_PASS 9/9。评分报告：`/mnt/efs-013e7987/share/swe/runs/swe-node92/arl-swe-b07394e405b44a5fbf3e482fffde4b7f/logs/run_evaluation/arl-swe-b07394e405b44a5fbf3e482fffde4b7f/arl/sympy__sympy-21847/report.json`。
- 退出后 master status 的 active=[]，pending_starts=0，工作区已回收。健康检查当次显示 25 个 reachable workers。未启动 GPU 训练；本次为可信负例接口验证，不是模型能力成绩。


## 2026-11-16 SWE 切换 Claude Code harness，继续使用本地 Qwen

用户明确要求继续本地 Qwen 并支持采集/GRPO。新增 `arl/harnesses/claude_code.py`，运行真实 `/root/.local/bin/claude` 2.1.226，通过回环 Anthropic Messages API 暂停 CLI 请求，由 ARL 原采样器生成；XML/Hermes 调用转为 Anthropic tool_use，唯一 MCP 工具 `mcp__swe__bash` 调用现有远程 SWE 工作区。CLI 最终回复/预算结束提交完整工作区 diff，复用官方 grader。原始 token/logprob/mask、同题分组、trajectory unit、optimizer 和参数同步路径保留；没有从 CLI 文本日志伪造概率。

SWE 注册默认、单环境/混合 YAML 和 shell 训练配置均切为 claude_code。ProgramBench 仍用 mini-swe-agent；SWE 显式选 mini_swe_agent 仍可用。Claude CLI 使用 bare 最小模式、独立临时 HOME、禁用所有 builtin 工具，仅经 MCP 远程读写/运行 bash；不加载用户 credentials/hooks/plugins/skills/CLAUDE.md 或启动子 agent。count_tokens 为估算，实际窗口和训练使用 Qwen tokenizer。版本不匹配时预检失败；使用 ARL_CLAUDE_CODE_BIN 可指定固定版本二进制。

验证：真实 CLI 的工具/反馈/最终提交、格式/截断不执行、预算结束、命令中取消清理、infra 标记、采集 artifact 对齐和 GRPO 优势测试通过；五环境共享 CPU optimizer 的 20 次真实更新/逐次参数同步测试通过，SWE 使用 Claude Code 的 MCP 动作。Qwen3.6-35B-A3B 实际 tokenizer 渲染两轮为 583→638 tokens，第二轮包含远程工具反馈。

真实远程 `--harness claude_code --negative` 通过：sympy__sympy-21847，node92，workspace `arl-swe-fe939babf2304f8d8fdc4925214eac74`，patch 181 bytes，resolved=false、reward=0、infra_error=false、PASS_TO_PASS 9/9。报告 `/mnt/efs-013e7987/share/swe/runs/swe-node92/arl-swe-fe939babf2304f8d8fdc4925214eac74/logs/run_evaluation/arl-swe-fe939babf2304f8d8fdc4925214eac74/arl/sympy__sympy-21847/report.json`。这是脚本化可信负例，不是模型成绩；未启动新的 GPU 训练。操作说明见 `harnesses/claude_code/README.md`。

本次最终全量回归：170 项测试全部通过（`python3 -m unittest discover -s tests -p test_*.py`）；6 项单环境/混合阶段配置预检通过。master status 确认冒烟 workspace 已回收。所有含 SWE 的实验输出目录追加 `_claude_code`，避免与旧 harness 身份混用。


## 2026-11-16 同步 ARM GRPO 训练改动，保留环境适配

按用户要求，以 `/root/workspace/ARM` 当前工作区为源（HEAD `021af58`，包含未提交训练改动），只同步 GRPO 训练侧。共享 `ppo_data`、loss、train_step、参数同步 helper、打包后负载重排原本已一致；本次补齐：

- `think_cost_free_tokens` 每回复免费额度，保留真实 sampled token、forced token 排除和轨迹终局正确性标签；以及通用 `think_budget` 两段式生成、插入 token 的 mask 和采样请求接线。
- sampling replay 配置同时控制 vLLM processed logprobs 和 trainer 的 temperature/top-k/top-p loss 分布；开始运行时显式复位旧环境变量。混合训练使用统一 replay 开关、每环境自己的采样参数。
- MoE router replay：sample/completer 类型、transition、trajectory datum、可选采集导出、trainer 前向/重算及指标。与 ARM 相同的 vLLM buffer 补丁预检，默认关闭。
- 固定 RMSNorm 的配置、worker 和 Triton 内核；支持的 5120 维 Qwen3.5/3.8 默认启用，与 ARM 相同。
- 共卡模块（KV-only sleep/wake、CUDA IPC 权重同步、训练显存余量）及 service 初始化、恢复、训练/评估阶段切换。launcher 支持 COLOCATE=1，采样卡必须是训练卡子集；默认保留本仓库既有分卡布局 COLOCATE=0，不复制 ARM 的机器/GPU/数据集配置。
- W&B 默认代理训练日志改动；显式空 KERNEL_RL_WANDB_PROXY 可关闭。

保留 ARL 的 logits reshape 显存修复、单 rank lazy sampler 首次同步修复和 policy/metadata 恢复修复。没有同步 ARM 的 Kevin 环境、逐轮评分/折扣、RM/value-head/PPO 支路、curriculum 或其他 SFT/蒸馏算法。环境/harness Python 文件散列对照仅 `arl/envs/mixture.py` 的训练 option 白名单有变化（新增免费额度/思考预算字段）；所有环境实现、任务源、奖励和 harness 文件均保持字节一致。`arl/agent/rollout.py` 仅增加训练 metadata 与计费/采样开关，原终局奖励、截断和轨迹分段语义保留。

最终验证：184 项全量测试通过；21 个实验阶段配置预检通过；shell 语法/Python 编译通过。15 个共享训练文件加 ppo_data 共 16 个文件与 ARM 按命名空间归一后一致；train_step、get_updated_sampling_client、dummy datum、负载重排 helper AST 一致。CPU LocalTrainingClient 的真实 forward/backward replay 概率/梯度、路由梯度/布局/传递、免费额度/强制 token mask 通过；五环境共享 CPU optimizer 完成 20 次更新并验证每批 sample→train 阶段调用。共卡设备操作为 mock 控制流验证，未运行新的 GPU 训练/共卡硬件测试。核对清单与测试日志：`runs/arm_training_sync_20261116/manifest.json`、`tests.log`。


## 2026-11-16 SWE 前 10 题 Qwen3.6 启动配方

用户要求用 `/models/Qwen/Qwen3.6-35B-A3B` 在 SWE 前 10 题训练和测试验证算法。
新增 `script/train_swe10.sh` / `script/swe10.py`，默认同 10 题底座测试→5 epochs
GRPO（group=8/batch=10）→final policy 测试→comparison.json。测试每题 4 条，前后预算和
采样参数一致，split=train/test_fraction=0，标为 training_set_diagnostic，不改环境重叠保护。
默认复用已有 8 train + 8 sampler 的分卡配置；也支持显式 GPU 列表/共卡。没有启动 GPU。

已准备 `runs/swe10_qwen36_v1`：服务端原始顺序前 10 条均为 Astropy，IDs 与 sha256
在 tasks.json；完整 grader 记录仅保存在 0600 的 tasks.private.jsonl。真实 10 题镜像
preflight 通过。启动：`./script/train_swe10.sh --run-dir runs/swe10_qwen36_v1`。
脚本保存各阶段配置/日志，不覆盖既有阶段输出；--check 仅预检、不联网、不启动模型。
对比时检查采样数量/任务 ID，汇报 infra/dropped、optimizer steps、active token 与非零
梯度信号；没有更新信号或 infra 污染则 inconclusive。使用官方 0/1 reward，无辅助惩罚。


## 2026-11-16 SWE10 保留用户 ARM 命令中的可用参数

新增 `script/train_swe10_aligned.sh`；SWE10 入口开放 replay、packing/head chunks、
late tokens、thinking、warmup、优势屏蔽和保存选项，预设逐项保留用户命令的训练值。
8 卡共卡/DP2，batch8/group8/epochs10，eval16，maxturn4，40960/40960，pack32768/
head8，think32768/lambda.005/free8192，warmup30/.5，replay 两项与 mask 两项 True，
save_optimizer/trajectories/best_solutions False。3 个 KERNEL_RL 前缀保持原值。
SWE10 测试现在遵循 SAMPLER_DP；关闭完整轨迹时 `run_native` 写最小 outcomes.jsonl，
比较器优先读取它，仍能检查失败/数量并汇报逐题结果。训练内部 eval_at_start=False，
完整流程的独立 baseline/after 比较继续保留。未引入 Kevin/KernelBench 环境行为。

发现当前共享 vLLM 0.19.0+cu130 缺少 router buffer 修复。新增项目内隔离覆盖层
`arl/models/vllm_overlay.py`，只替换 scheduler/model-runner 两处 buffer 大小为
num_blocks × attention block_size，其他文件链接原安装，不修改共享包。
SWE10 开启 router replay 时自动准备并把覆盖层加到子进程 PYTHONPATH，记录散列。
CPU 验证物理 slot 上界、包解析位置、重复准备、原安装不变；未知源码/版本拒绝修改。
--check 无 GPU/网络/写入副作用；实际 GPU 训练尚未启动。

用户随后确认 router replay 补丁已打上。已重新读取当前 `/usr/local/lib/python3.12/site-packages/vllm` 两处源码，均有 ARM 标记和正确的 num_blocks×attention_block_size 范围；prepare_overlay() 返回 None，当前启动配方直接复用已修复安装，不启用隔离覆盖层。对齐配方 --check 通过，未启动 GPU。

SWE10 新增 --harness claude_code|mini_swe_agent（默认 claude_code），普通/对齐启动器均可用。所选 harness 贯穿 baseline/train/after 及 comparison 身份；两组配置各 3 阶段预检，不启动 GPU。不同 harness 必须用独立 run-dir。


## 2026-11-16 ARM 风格 key=value 和外部异步 SWE 评估

用户要求另一台机器从共享 checkpoint 异步评估，并参考 ARM 实现；另要求不使用 -- 风格。
已新增 `script/4b_train_qwen36.sh`（SWE10 对齐配方别名），所有 SWE10 入口支持 ARM 风格
key=value（大写 GPU/runtime，num_epochs/eval_num_samples 等别名），仍支持
原 --flag。布尔规范化、重复后者覆盖，无 shell eval，未知/KernelBench 环境参数明确报错。
对齐配方不再写死 run-dir，省略即自动生成。

`eval_async=True eval_every=5`：训练端发布 async_eval/spec.json 固定评估语义与任务散列，
每 N batch 保存 checkpoint，只跑训练，不在本机先测/后测，不等待评估。独立 baseline
仍为训练效果比较而在评估机排队；内部 eval_at_start=False 保留。
`arl/experiments/eval_service.py` 从 ARM rlhf/ppo/eval_service.py 改编，保留队列/glob、
GPU 池/UUID lease、probe、心跳锁、重试、进程组/orphan 清理、状态、redo 存档；GPU lease
源码复用到 arl/utils/gpu_lease.py，沿用共享 UUID.lock 协议以协作 ARM/CudaMaster。
SWE 适配读取 root/train/checkpoints.jsonl 和冻结 spec，去掉 KernelBench health/kwargs。
最终周期点与 final 同 batch 时只评 final。

评估机入口 `script/eval_swe10_async.sh add RUN GPUS=auto NGPUS=8 SAMPLER_DP=2 ...`。
每个点用 `arl.experiments.async_swe_eval` 子进程，验证冻结任务散列/已发布权重，沿用同一
SWE source/harness/采样预算并写 progress/results/metrics/done；infra/dropped 不写 done，
服务负责重试。base/final 完成自动生成 comparison。源任务和模型需两端同绝对路径；
示例共享根 /root/share/arl-runs。SWE_EVAL_TOKEN_FILE 可指定评估机令牌文件路径。
未启动服务、未启动训练或评估 GPU，仅做配置/CPU 验证。

本次最终验证：224 项全量测试通过；ARM 风格的完整可用参数命令（含 eval_async=True）配置预检通过，save_every=5；CPU 验证无本机同步测试启动、部分索引不消费、周期/final 去重、done→comparison、UUID 租约、PID namespace 和真实进程组清理。没有启动 daemon 或 GPU 作业。记录在 runs/swe10_async_validation_20261116。


## 2026-11-16 默认全 GPU、采样 DP=4

用户要求训练和采样都默认使用全部 GPU，SAMPLER_DP=4。SWE10 parser 默认 all/all、
colocate=True、DP4；对齐 preset 去掉固定 8 卡列表、DP 改 4。all 尊重 CUDA_VISIBLE_DEVICES，
否则 nvidia-smi 枚举，必要时回退 torch 设备数。显式列表仍可覆盖。
SWE10 默认 PACK_TOKENS=32768；按 ARM launcher 的 packing+rebalance 条件放开
batch_size<world_size 的训练校验，沿用原 dummy/重分配/lockstep 训练实现，不重复题目。
4 个真实 CPU/Gloo rank、2 个初始任务 rank 测试证明重排后各 rank 都有 active datum，
且原分母权重的目标总和保持一致。本机 check=True 显示16训练rank/all16采样、DP4。
没有启动 GPU 作业。异步评估机 GPU/NGPUS/DP 参数仍独立配置。

用户随后询问 max_turns：当前4b对齐配方默认4，指模型回复/动作反馈轮数，不是GRPO固有限制；
ARM原命令也设置4，但它每轮完整改kernel，SWE通常每轮一个工具动作，两者预算不等价。
未擅自更改 max_turns；可传 max_turns=250 使用原SWE环境预算。

全 GPU 默认配置最终验证：228 项全量测试通过，本机预检解析训练/采样均为0–15、16 rank、SAMPLER_DP4。真实4 rank CPU/Gloo 小 batch 重分配测试通过；GPU 作业未启动。记录：runs/swe10_all_gpu_validation_20261116。

用户要求去掉 save_best_kernels 兼容名：SWE10 现只接受 save_best_solutions=True/False；旧名明确报错，文档和测试已更新。

## 2026-11-16 训练/评估独立选题

SWE launcher 开放 task_limit（别名 train_task_limit）、train_instance_ids、train_ids_file，
及 eval_dataset、eval_task_limit、eval_instance_ids、eval_ids_file；全部支持 key=value。
默认仍前10题，未传任何评估选题参数则沿用训练选题。显式 IDs 按原字符串和顺序选取，
重复/缺失/首尾空白报错；文件支持逐行 IDs 或 JSON 字符串数组，同侧列表/文件互斥。
训练 records 仍 tasks.private.jsonl，独立评估则 eval_tasks.private.jsonl，均0600。
公开 manifest 记录两侧 IDs/重叠/sha，比较报告依重叠情况区分诊断与独立评估。
异步 spec 固定两侧文件 hash；worker 校验评估 records 并按评估 IDs 汇总。
顺便修复 expected_batches：GRPO 的 _batches 先展平 epochs 后分批，应为
ceil(train_count * epochs / batch_size)，不是 ceil(train_count / batch_size) * epochs。
删去 batch_size<=10 限制，适配任意题数；旧默认10题的 immutable manifest 可继续读取。
CPU 选题测试5项及既有 SWE 配方/异步/GPU默认回归28项通过；实际入口 check=True
验证独立2题训练/1题评估且未启动训练。选题改变需用新 run 目录。
本次最终全量回归：233 项测试全部通过（34.7 秒），日志 /tmp/swe-selection-full-tests.log。

用户使用 eval_at_start=True eval_async=True 被原 launcher 校验拒绝：已移除该拒绝，
True 对应已有的独立 baseline 流程，内部 GRPO eval_at_start 仍 False 防止内部切分/重复评估。
False/省略仍保留默认前后比较 baseline。异步 spec baseline_requested=True，服务排队 base，
训练端不等待。用户完整命令加 check=True 已通过（10题训练/10题评估，无网络/GPU启动）；
29 项 SWE 配方/异步/选题回归通过。尚未启动用户实际训练。

## 2026-11-16 W&B 项目及实验命名

参考 ARM script/_lib.sh 的 run_stamp/next_run_dir 和训练脚本的目录取名方式：
SWE/4b 默认 ROOT/runs/<模型basename规范化>_swe_grpo_<MMDD>_vN。
Qwen3.6-35B-A3B => qwen36_35b_a3b_swe_grpo_1116_v1；已存在则v2，mkdir原子占位。
显式run_dir/log_path不自增，支持strftime日期。check仅预览，不建目录、不占版本。
SWE训练默认 wandb_project=ARL，wandb_name=run.name（不是train子目录名）；
可wandb=False关闭。GRPOConfig默认项目ARL，所有现有训练.conf项目均统一ARL。
共享ml_log默认name取日志目录basename，W&B本地文件放log_dir；SWE显式传父实验名。
本地metrics.jsonl继续保存，W&B错误仍告警并继续训练，凭据沿用用户SDK/环境。
异步评估metrics仍保留本地，没有加入跨机器上传到训练W&B run的流程。
46项命名/日志/配方/异步/训练同步测试通过；实际W&B SDK离线init/log/finish通过，
真实入口check确认ARL项目及新目录名，无GPU/网络训练启动。未测试真实在线上传。
最终全量回归237项通过（34秒，WANDB_MODE=disabled避免测试上传）；日志 /tmp/arl-wandb-full-tests.log。

用户进一步要求简化命名：SWE/4b现为 qwen36_1116_v1，去掉模型规模和swe_grpo字段。
Qwen按系列/版本简称（Qwen3.6=>qwen36，Qwen3.5=>qwen35），其他模型保留规范化basename。
目录及W&B实验名仍一致，project=ARL、日期和原子版本自增保留。文档和命名回归已更新。

## 2026-11-16 取消 SWE 独立 prompt cap 及全局 late-token 机制

用户要求取消固定prompt budget，随后追加取消max_tokens_late相关机制。
SWE/4b移除prompt_budget入口，新增max_model_len（sampling_max_model_len别名），
默认0从原始模型config.json读取总上下文（支持text_config）。上下文推导输入容量=
window-max_tokens；显式window不能超过模型声明上限，必须给输入留空间。
Qwen3.6配置262144，max_tokens40960时输入容量221184，不再在32768处压缩。
训练/baseline/final/异步spec全部明确冻结sampling_max_model_len与推导的prompt_budget_tokens。
内部rollout保留该推导容量防止超模型窗口，两个harness共享render/rollout实现。
取消max_tokens_late及switch阈值/耐心字段、GRPO单向latch和动态cap状态，
所有训练/评估直接使用各环境cfg.max_tokens；各launcher/conf、mixture白名单已清理。
旧SWE命令需删max_tokens_late和prompt_budget等参数，不静默兼容。
50项针对性测试通过，含长历史>32K不压缩、实际总窗口边界、两harness全阶段对齐、
窗口越界/未知检查和废弃参数拒绝。完整用户命令去掉late参数加check已预检。
没有启动GPU训练，仍需在实际显存条件下运行。
最终全量回归241项通过（31.3秒）；日志/tmp/arl-context-full-tests.log。真实入口check确认262144总窗口/221184输入容量，各阶段一致。

## 2026-11-16 首次真实单题采样完成

用户要求找一道题不加额外限制采样。实际运行已完成，非CPU假测：
run=/root/workspace/ARL/runs/qwen36_1116_v1，task=astropy__astropy-12907，
本地/models/Qwen/Qwen3.6-35B-A3B，mini_swe_agent 2.4.6，一题一条轨迹。
GPU13,14 TP2，获取共享租约后运行。8,9显存空但被其他任务预留，未使用/未终止他人任务。
一次性driver sample_one.py在run目录，用LocalServiceClient inference_only加载真实vLLM，
注入标准run_native。每次请求动态max_tokens=262144-实际prompt长度，think_budget=0，
prompt仅受总模型窗口限制；env_max_turns=2147483647作为现有harness的实际不触及上限。
保留工具60s、官方评分1800s超时。未改生产训练/采样默认值以模拟此不设额外生成上限的探针。
预启动两次未形成轨迹：一次GPU租约冲突退出；一次发现inference_only需要显式
KERNEL_RL_SAMPLER_TP=2，停止自己的启动进程组后重启。实际仅采样了1条轨迹，没有重采挑结果。
最终官方resolved=True、infra_error=False、Submitted；FAIL_TO_PASS 2/2，PASS_TO_PASS13/13。
33轮，累计生成7555token，单轮最大842，最大输入21627；全部stop，0次length截断。
生成耗时合计107.27秒（不含工具/评分/冷启动）。首次PPU Triton/DeepGEMM编译较久。
核心修改separable.py的_cstack：右侧矩阵块从填1改为填right。
完整trajectory.txt、solution.patch、grade.json、summary.json已整理；原始responses和
sample/trajectories/rank0.jsonl.gz、sample/outcomes.jsonl、sample.log均保留。
进程成功退出0，GPU13/14显存回到1MiB、租约已释放。run v1已实际占用，下次默认v2。

## 进行中：跨请求 prefix caching 对照实验

用户请求实验验证。生产默认未修改。实验目录 runs/qwen36_1116_v2（/tmp/arl-prefix-run指向它）。
run_probe.py 持有 GPU13/14共享租约，顺序启动 prefix_worker.py off / on_align 两个进程，
exec session=30251，主日志probe.log，各自off.log/on_align.log；最终probe.done。
两组均原模型、TP2、262144窗口、PIECEWISE、raw_logprobs、greedy(seed42)、每探针输出上限64。
on组通过进程内_extra_engine_args覆写 enable_prefix_caching=True,mamba_cache_mode=align。
不动共享vLLM安装或生产默认。当前off在PPU首次Triton调优，torch.compile缓存已加载；
536649/536650是本实验off的worker，主worker536364（仅供识别，发信号前必须重新核实）。
prompts.json从v1真实mini-swe轨迹重建7个实际输入（turn0/2/10/15/16/17/32），
长度逐个核对原始轨迹一致；15->16->17保留完整旧prompt前缀。额外两条共享前缀分支。
探针覆盖实际历史增长、21K长前缀重复3次、并发双分支、GPU-only embedding符号翻转模拟权重变化。
嵌入权重只在本实验GPU进程内neg_，第二次neg_精确恢复，不写checkpoint；
先故意不清cache测旧KV，再reset测与off新权重一致，重复命中，再恢复/reset测原输出。
各请求持久化off.jsonl/on_align.jsonl，含tokenIDs/logprobs/真实num_cached_tokens/TTFT/总延迟。
reset事件在*.events.jsonl，各模式完成*.done.json；失败*.failure.json。
分析脚本analyze.py已写好，待两组完成运行 python runs/qwen36_1116_v2/analyze.py 生成comparison.json。
固定权重输出、真实cache命中、更新后reset一致性和恢复应分别报告；不声称验证了完整GRPO或colocate unmap/wake。
必须等真实结果，结束后确认GPU显存和租约释放。每60秒内给用户简短进展。

### Prefix caching 实验已全部完成（含补充复查）

v2主实验两个进程各19次请求，均exit0；comparison.json/report.txt已生成。
GPU13/14 TP2，真实mini-swe历史1650-21627token，greedy单探针最多64生成token。
重复21627输入命中21120token：TTFT中位数1.3033s->0.1175s（11.10x），
整个54token请求中位数1.8208s->0.6557s（2.78x）。并发两分支输出均匹配off。
跨配置17个有效对照15个token序列相同，actual_turn10/17分别从token9/10起分歧；
零命中请求也有logprob差异，不可宣称两配置数值等价。
GPU-only embedding符号翻转不reset命中旧10560token且第一token即与新权重off参考不同；
reset后命中0且输出匹配off新权重参考，后续cached repeat在on内logprob完全一致，恢复也成功。
为排除缓存污染，追加on_align_focus独立进程（exec45568已exit0），逐个reset后冷/热配对：
turn0/10/15/16/17/32，6/6 token AND logprob完全一致（max_abs_delta=0），
两个分歧case冷算也复现，且与首个on进程输出一致。说明测到的复用本身正确，
但align配置与原off计算路径有稳定差异；未进一步宣称差异具体数学根因。
补充文件on_align_focus.jsonl/focus.log/analyze_focus.py，report.txt汇总方法、数字、局限。
生产默认未开启，checkpoint未写回，未运行完整GRPO或colocate KV unmap/wake验证。
所有实验子进程已退出，GPU13/14各1MiB，租约已释放；不要重复启动已完成实验。

## 进行中：用户怀疑 ARM 的 RMSNorm autotune / 分块导致 greedy 差异

已核对 ARM README.md 164行及r3lm/models/fixed_rmsnorm*.py，ARL代码仅包名不同，完全一致。
现有ARM固定内核只支持qwen3_5/qwen3_5_text + hidden_size5120（27B）；
当前Qwen3.6-35B-A3B是qwen3_5_moe_text,hidden_size2048，engine_args(config)={}，未启用旧补丁。
已明确告诉用户这是疑点而非确认根因，正在真实GPU验证。
实验目录 runs/qwen36_1116_v3（/tmp/arl-rms-run），exec session49580运行run_probe.py，
持有GPU13/14租约；依次norm_unit.py、off_fixed2048、on_align_fixed2048，模型进程日志各自命名。
norm_unit已exit0：GemmaRMSNorm2048，BF16，torch.compile，rows1/16/32/128/512/1056/1650/2048，
原生compiled与实验fixed逐位相同，各batch首行相同，2048整块vs1056+992分块changed_elements=0。
此unit只直接测无residual路径；固定worker实现也覆盖residual路径，真实模型将进一步检查。
实验rms_probe_kernel.py仿ARM，仅H2048/Rblock2048/xblock1/warps8/stages1的opaque customop。
rms_probe_worker.FixedRMSProbeWorker在模型加载/编译/捕获前安装，匹配2048的GemmaRMSNorm，
其他维度及gated norm不改；worker输出RMS_PROBE_FIXED_2048_ACTIVE层数和status.json。
配置进additional_config使编译cache哈希包含实验标记；不修改生产模块或共享vLLM安装。
prefix_worker.py基于v2，两个模式都固定2048norm，分别prefix off/align；
6个真实promptturn0/10/15/16/17/32逐个清cache后冷/热配对64token greedy。
预写analyze.py，待两组完成运行 python runs/qwen36_1116_v3/analyze.py，
比较native_off->fixed_off、native_on->fixed_on、fixed_off->fixed_on及同引擎冷/热。
原生参照在v2/off.jsonl和v2/on_align_focus.jsonl。需等真实模型结果再判断根因，不能仅unit排除。

### RMSNorm 检查已完成，定位到 runtime prefill 分块路径

v3全部进程成功exit0，原始配对fixed_norm两模式完成；GPU13/14租约释放且各1MiB。
实际模型两GPU各81个2048 GemmaRMSNorm已验证固定内核生效；fixed_off vsfixed_align仍3/6 greedy不同。
因此不能称“扩展旧ARM norm补丁就修好了”。固定norm本身也改变原生部分输出（融合边界变化）。
追加norm_residual.py GPU单测（exec91006已exit0）：compiled residual path与eager/fixed有BF16差异，
最大0.03125，但compiled2048整块vs1056+992切分仍0差异；不能说所有RMSNorm数值路径相同。
原v2 off/on有33/33 Inductor subgraph cache_key完全相同（文件整体目录hash不同），已记录。
决定性控制run_matched.py/prefix_worker_matched.py（exec34576已exit0）：
不改RMSNorm、不启用prefix caching、不改模型编译/profiling范围2048，
仅进程内Scheduler.__init__后设置max_num_scheduled_tokens=1056。
spawn在EngineCore里重复安装实验patch，没有修改site-packages。
matched_scheduler_560871.json实测记录original_scheduled_tokens2048,scheduled_tokens1056,
max_num_batched_tokens2048,block_size1056,prefix_cachingFalse。
6个prompt（turn0/10/15/16/17/32）全部与v2原生align cold输出token AND logprob逐位相同，max_delta0。
包含之前分歧的turn10/17。故本次差异可由runtime prefill切块复现，不需假设cache复用错误或不同RMSNorm加载测速配置。
尚未逐层定位首个不同的算术算子，不能把GDN或某个norm算子断言为唯一根因。
报告runs/qwen36_1116_v3/report.txt、comparison.json；finalize.py汇总了所有控制和局限。
生产默认、ARM代码、共享vLLM、checkpoint文件均未修改。不要重复启动已完成实验。

## 2026-11-16 prefill参数正式接入，采样4×TP4

用户要求设置预算并使用4个TP4；工作中追问1056来源，已解释TP2实测块大小与Mamba align边界。
SWE/4b新增prefill_tokens、prefill_batch_tokens及对应大写别名，默认0保持vLLM行为。
prefill_tokens>0时编译预算默认max(2048,prefill_tokens)，可显式prefill_batch_tokens=2048。
冻结到GRPOConfig sampling_prefill_tokens/sampling_prefill_batch_tokens，全训练/独立baseline/final/
异步spec一致，经过LocalServiceClient->LocalTrainingClient->_sampling_cfg->SamplingEngine。
新增arl/models/prefill.py(纯配置验证/合并，保留fixed RMSNorm additional_config)及
prefill_scheduler.BudgetAsyncScheduler(继承vLLM AsyncScheduler，通过scheduler_cls与serialized config接入)。
只改self.max_num_scheduled_tokens，编译预算保留max_num_batched_tokens；保留async scheduling。
启动日志打印实际block_size/预算；Mamba align模式要求调度预算是实际block_size整数倍，且不超过引擎预算。
预算包括prefill和decode每副本共享token，不限制对话长度/输出长度；并发/尾块仍可能改变每请求实际块长，
不能宣称相同预算保证跨TP/并发逐位输出相同。预算配置不会自动启用prefix caching。

修复原生单进程inference_only只走_own_gpus_for_rank忽略DP显式GPU池的问题：
WORLD_SIZE1+显式KERNEL_RL_SAMPLER_GPUS时使用_sampler_replicas，支持MultiSamplingEngine；
多进程旧推理/无显式GPU池仍保留rank-local TP切片。训练原DP路径保持。
部分副本启动失败会shutdown已建引擎并恢复父进程环境，避免新推理多副本路径泄漏。

预检验证16张采样卡+SAMPLER_DP4=>四组[0-3],[4-7],[8-11],[12-15]，每组TP4；
check输出sampler_tp和sampler_replicas。没有启动实际16卡模型/训练。
另外做了真实vLLM AsyncEngineArgs.create_engine_config配置检查（只读metadata，未加载模型）：
TP2=>block_size1056，TP4=>block_size528，且mamba_block_size同值，
新scheduler_cls/async_scheduling/编译预算2048/序列化预算1056均被真实vLLM接受。
因此用户4×TP4配置中1056是每副本两个528token块，不是普遍魔数；528/1584也对应一/三块预算。
检查日志/tmp/arl-prefill-config-check.log，拓扑/tmp/arl-prefill-topology.json，
入口预检/tmp/arl-prefill-launch-check.json，临时metadata租约已释放。

6项新增测试覆盖预算字段与旧RMS配置合并、实际块对齐、真实SchedulerConfig解析、
全阶段预算传递、训练/单进程推理4个TP4引擎构造(mock，无GPU加载)、失败清理、旧推理TP切片。
247项全量回归通过（48.4秒，WANDB_MODE=disabled），日志/tmp/arl-prefill-full-tests.log；git diff --check通过。
用户原命令可保留SAMPLER_GPUS=all SAMPLER_DP=4，在16张可见GPU上追加prefill_tokens=1056 prefill_batch_tokens=2048。
异步评估也要4×TP4时，服务add指定NGPUS=16 SAMPLER_DP=4，预算从冻结spec继承。

## 2026-11-16 enable_prefix_caching / mamba_cache_mode正式开放

用户明确要求开放两个参数。SWE/4b parser新增 enable_prefix_caching=True/False（默认None跟随vLLM模型），
mamba_cache_mode=none/align/all（默认none）；支持对应大写别名。未显式开启时原默认保持。
两个字段进入GRPOConfig、全阶段options/独立评估白名单、ServiceClient/TrainingClient/_sampling_cfg，
最终由SamplingEngine转成真实AsyncEngineArgs。异步spec冻结相同配置，无需shell环境传播。
新增arl/models/prefix_cache.py纯配置校验：align/all必须显式True；Qwen3_5系列（含当前MoE）
禁止all，提示use align；align显式要求chunked prefill。SamplingEngine在创建引擎前校验模型配置。
保留prefill预算和fixed RMSNorm additional_config合并，未改变已有权重更新/共卡cache reset流程。
修正SamplingEngine.reset_prefix_cache旧注释“所有V1默认开启”为按模型默认/显式开关实际行为。

用户原命令可追加 enable_prefix_caching=True mamba_cache_mode=align，
与SAMPLER_GPUS=all SAMPLER_DP4(16卡=>4×TP4)、prefill_tokens1056 prefill_batch_tokens2048组合。
关闭需enable_prefix_caching=False mamba_cache_mode=none。生产入口现已能显式开启，默认未自动开。
完整GRPO共卡生命周期仍未实测，没有在这次参数开放任务里启动训练。

6项新增回归：默认/组合校验、Qwen all提前拒绝、三阶段计划一致、TrainingClient配置存储、
实际vLLM AsyncEngineArgs构造捕获（mock引擎不加载GPU模型）、异步spec持久化。
既有4副本构造测试也验证每个副本都得到相同prefix开关/mamba模式。
253项全量测试通过（37.5秒，WANDB_MODE=disabled），git diff --check通过。
真实SWE/4b入口check=True通过4×TP4+预算+True/align，未联网/未启动GPU；
日志/tmp/arl-prefix-switch-full-tests.log，入口预检/tmp/arl-prefix-switch-check.json。

## 2026-11-16 用户要求再次检查当前设置

复核实际完整命令（原训练选项+cache True/align+prefill1056/2048），入口check=True通过：
16卡分四组TP4，train/eval各10题，总上下文262144，未启动GPU/未联网。
预检文件/tmp/arl-reviewed-settings-check.json；本机GPU当前均被占用，未做真实16卡GRPO试跑。
源码复核vLLM pause_scheduler abort等待/清理、Scheduler cache reset、router replay按真实attention
KV块slot读取整个prefix路由，补丁attention gid/block size匹配；未宣称新的GPU replay组合实测。
共卡顺序pause_generation(mode=abort,clear_cache=True)全部完成再kv_sleep，
唤醒顺序kv_wake全部完成->resume全部完成->reset各副本cache。

发现保护不足并修复：SamplingEngine现在记录_require_prefix_cache_reset（显式True），
reset_prefix_cache返回False时显式开cache会raise RuntimeError停止，拒绝旧策略KV继续采样。
默认未显式开cache时保留旧warning行为。MultiSamplingEngine任一副本失败会传播异常。
新增tests/test_prefix_cache_lifecycle.py共4项：显式失败停止、旧warning行为、任一副本失败传播、
四副本共卡pause+clear/unmap/map/resume/reset完整调用顺序；实际constructor测试也断言保护flag已接通。
257项全量回归通过（37.6秒），git diff --check通过；/tmp/arl-reviewed-settings-full-tests.log。

明确告知用户四个副本缓存互相独立、按最少inflight分流，无跨轮固定副本亲和性，
缓存命中率/整体提速应实跑测量，不能把v2单引擎11x TTFT当成四副本训练整体收益。
完整GRPO+colocate+prefix+router replay的16卡生命周期尚未GPU实跑，不能宣称已经验证。

## 2026-11-16 跨轮轨迹级副本亲和性已实现

用户问能否保证跨轮返回同一副本；已接入默认自动生效，无需新增命令参数。
MultiSamplingEngine维护affinity_key->replica，后续轮次无视临时负载变化保持原副本。
新轨迹按(n_inflight,已绑定轨迹数)选副本。只保证归属，不保证物理KV永不被驱逐。
SamplingClient在整个sample_async开始时解析一次实际副本，思考/回答两段也固定，
即使未提供affinity key也不会在两段之间重选。无key的普通请求仍按负载选择。
SamplingParams尾部新增可选affinity_key（保持旧位置参数顺序）。

DistributedSamplingClient新增begin/end_affinity_scope、来源rank/客户端UUID命名空间，
root转发时把origin+scope+trajectory绑定键传给SamplingClient，SampleRequest原始对象不原地修改。
run_agent_rollout对真实DistributedSamplingClient自动开始scope，每条轨迹保持scope:tid键；
request id改为scope.tid:turn（只对这条真实链路，Fake/插件旧client保持原tid:turn），避免并发scope同名请求。
每次正常/异常/取消收尾取消并await本scope剩余采样任务，再释放本scope绑定，不新增collective。
作用于Claude Code/mini SWE等共享rollout、训练及独立评估；权重更新仍按原流程清KV。
绑定在rollout作用域结束清除，下批重新按负载分配；长轨迹后期可能导致副本负载不齐，这是保持亲和性的代价。

新增tests/test_sampler_affinity.py共7项：新轨迹均衡/后续负载变化不迁移、不同rank/客户端键隔离、
两段生成固定（含无key）、作用域取消隔离与释放、真实两轮Fake SWE rollout8轨迹、异常收尾，
以及4个真实CPU/Gloo rank（仅2个rank有任务、2个空rank），根节点四个假引擎，8条轨迹两轮，
验证所有轨迹同副本、每引擎4次生成、全局收尾无绑定/任务泄漏。
264项全量回归通过（47.1秒），git diff --check通过；/tmp/arl-affinity-full-tests.log。
没有启动GPU模型或训练；不声称已实测16卡亲和性吞吐。
