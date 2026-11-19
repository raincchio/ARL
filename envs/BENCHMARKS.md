# Benchmark 目录

按给定清单登记 **26 个 benchmark（coding 15 个、instruction_following 11 个）**。各版本独立建目录。
IFEval 已接入官方规则判分和原生 GRPO，ProgramBench 已接入官方提交包判分，mini-swe-agent 与 GRPO 适配待完成；MultiCodeIF 已部分接入数据、约束评测和文本轨迹采集。其余 benchmark 主要为目录骨架。

默认调用 `benchmark()`；只有显式指定 `--mode train` 才选择 `train()`，IFEval 已可训练，其余新增 benchmark 提示“训练接入待完成”。
待完成入口退出码为 2，不会下载数据、连接服务、初始化模型或启动训练。MultiCodeIF 默认检查缓存数据清单，指定 `--config` 后运行相应评测。

```bash
python3 -m arl.benchmarks --list
python3 -m arl.benchmarks --list --category instruction_following
python3 -m arl.benchmarks --list --category coding --group repository_development --json
python3 -m arl.benchmarks coding.swe_bench_verified
python3 -m arl.benchmarks instruction_following.ifeval --mode train
```

每个目录包含 `README.md` 和 `entry.py`，后者分别定义 `benchmark(**kwargs)` 与 `train(**kwargs)`。
名称、分类、路径和状态的统一注册清单为 `arl/benchmarks/catalog.json`。

| 分类 | 子类 | Benchmark | 调用标识 |
|---|---|---|---|
| `coding` | 仓库级开发 | [SWE-bench Verified](coding/benchmarks/repository_development/swe_bench_verified/README.md) | `coding.swe_bench_verified` |
| `coding` | 仓库级开发 | [SWE-bench Pro](coding/benchmarks/repository_development/swe_bench_pro/README.md) | `coding.swe_bench_pro` |
| `coding` | 仓库级开发 | [SWE-bench Pro V2](coding/benchmarks/repository_development/swe_bench_pro_v2/README.md) | `coding.swe_bench_pro_v2` |
| `coding` | 仓库级开发 | [SWE-bench Multilingual](coding/benchmarks/repository_development/swe_bench_multilingual/README.md) | `coding.swe_bench_multilingual` |
| `coding` | 仓库级开发 | [DeepSWE v1.1](coding/benchmarks/repository_development/deepswe_v1_1/README.md) | `coding.deepswe_v1_1` |
| `coding` | 仓库级开发 | [FrontierSWE](coding/benchmarks/repository_development/frontierswe/README.md) | `coding.frontierswe` |
| `coding` | 仓库级开发 | [FrontierCode v1.1](coding/benchmarks/repository_development/frontiercode_v1_1/README.md) | `coding.frontiercode_v1_1` |
| `coding` | 仓库级开发 | [CursorBench 4.0](coding/benchmarks/repository_development/cursorbench_4_0/README.md) | `coding.cursorbench_4_0` |
| `coding` | 终端任务 | [Terminal-Bench 2.1](coding/benchmarks/terminal_tasks/terminal_bench_2_1/README.md) | `coding.terminal_bench_2_1` |
| `coding` | 终端任务 | [Terminal-Bench 4.0](coding/benchmarks/terminal_tasks/terminal_bench_4_0/README.md) | `coding.terminal_bench_4_0` |
| `coding` | 从零生成项目 | [NL2Repo-Bench](coding/benchmarks/project_generation/nl2repo_bench/README.md) | `coding.nl2repo_bench` |
| `coding` | 从零生成项目 | [ProgramBench](coding/benchmarks/project_generation/programbench/README.md) | `coding.programbench` |
| `coding` | 竞赛编程 | [LiveCodeBench v6](coding/benchmarks/competitive_programming/livecodebench_v6/README.md) | `coding.livecodebench_v6` |
| `coding` | 竞赛编程 | [Codeforces Rating](coding/benchmarks/competitive_programming/codeforces_rating/README.md) | `coding.codeforces_rating` |
| `coding` | CUDA 算子优化 | [KernelBench](coding/benchmarks/kernel_optimization/kernelbench/README.md) | `coding.kernelbench` |
| `instruction_following` | 通用指令遵循 | [IFEval](instruction_following/benchmarks/general/ifeval/README.md) | `instruction_following.ifeval` |
| `instruction_following` | 通用指令遵循 | [IFBench](instruction_following/benchmarks/general/ifbench/README.md) | `instruction_following.ifbench` |
| `instruction_following` | 复杂多约束 | [FollowBench](instruction_following/benchmarks/multi_constraint/followbench/README.md) | `instruction_following.followbench` |
| `instruction_following` | 复杂多约束 | [ComplexBench](instruction_following/benchmarks/multi_constraint/complexbench/README.md) | `instruction_following.complexbench` |
| `instruction_following` | 多轮、多语言 | [Multi-IF](instruction_following/benchmarks/multi_turn_multilingual/multi_if/README.md) | `instruction_following.multi_if` |
| `instruction_following` | 指令优先级 | [IHEval](instruction_following/benchmarks/instruction_hierarchy/iheval/README.md) | `instruction_following.iheval` |
| `instruction_following` | 结构化输出 | [IFStruct](instruction_following/benchmarks/structured_output/ifstruct/README.md) | `instruction_following.ifstruct` |
| `instruction_following` | Agent 指令遵循 | [AgentIF](instruction_following/benchmarks/agent/agentif/README.md) | `instruction_following.agentif` |
| `instruction_following` | 代码指令遵循 | [IFEvalCode](instruction_following/benchmarks/code/ifevalcode/README.md) | `instruction_following.ifevalcode` |
| `instruction_following` | 代码指令遵循 | [CodeIF](instruction_following/benchmarks/code/codeif/README.md) | `instruction_following.codeif` |
| `instruction_following` | 代码指令遵循 | [MultiCodeIF](instruction_following/benchmarks/code/multicodeif/README.md) | `instruction_following.multicodeif` |

名称和版本按给定清单保留，具体评测和训练接入在各目录的 `entry.py` 中完成。

现有 `coding.swe` / `coding.cuda` 是已实现的训练环境，不是本清单中的官方 benchmark 入口；原有命令仍保留。
`coding.swe_bench_verified` / `coding.kernelbench` 的独立 benchmark 入口仍标记为待完成，不会默默转到已有训练环境。
IFEval、ProgramBench 名称已注册为可训练环境；其余本清单中的 benchmark 名称传给 GRPO 的 `env` 时仍提示训练接入待完成。

MultiCodeIF 已部分接入：支持 API/离线预测、官方约束检查和 self-repair，范围与用法见 [MultiCodeIF](instruction_following/benchmarks/code/multicodeif/README.md)。功能执行和编译不计分，训练入口仍待完成。

IFEval 的训练、官方 strict/loose 检查与数据划分见 [IFEvalEnv](instruction_following/IFEvalEnv/README.md)。

ProgramBench 的官方评分器、Docker cleanroom、训练与混合示例见 [ProgramBenchEnv](coding/ProgramBenchEnv/README.md)。
