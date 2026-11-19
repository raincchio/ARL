# 环境分类

环境分为两类，训练器统一通过 `arl.agent.protocol` 使用 `AgentEnv` 和 `TaskSource`。

新增 benchmark 清单按子类放在 `envs/<category>/benchmarks/`：coding 15 个，instruction_following 11 个。
benchmark 目录的默认入口为数据检查/评测；实际模型在线训练使用实验入口。当前注册的 CUDA、SWE、ProgramBench、IFEval、MultiCodeIF 均通过各自 harness 接入本地 GRPO；其他仅有目录的 benchmark 不自动成为可训练环境。完整目录见 [Benchmark 目录](BENCHMARKS.md)。

```text
envs/
├── coding/
│   ├── CudaEnv/                  CUDA 算子编写与性能优化
│   ├── SweEnv/                   SWE-bench 仓库修复
│   ├── ProgramBenchEnv/          从零重建参考程序与官方行为测试
│   └── benchmarks/               仓库、终端、项目生成、竞赛、算子 benchmark 目录
└── instruction_following/
    ├── IFEvalEnv/                官方约束检查与单轮 GRPO 环境
    └── benchmarks/               通用、多约束、多轮多语言、优先级、结构化、agent、代码指令 benchmark

arl/envs/
├── coding/
│   ├── cuda.py                  CudaEnv 的训练适配
│   └── swe.py                   SweEnv 的训练适配
├── instruction_following/
│   └── ifeval.py                IFEval 的训练适配
├── registry.py                  环境元数据和延迟加载
└── __init__.py                  分类查询、名称解析和工厂入口
```

| 分类 | 环境名 | 内容 | 兼容名称 |
|---|---|---|---|
| `coding` | `coding.cuda` | CUDA 算子优化；CudaTasks 题库、CudaEval 判分 | `cuda` |
| `coding` | `coding.swe` | SWE-bench 软件仓库修复；官方 harness 判分 | `swe` |
| `coding` | `coding.programbench` | 根据参考程序从零生成项目；官方行为测试 | `programbench` |
| `instruction_following` | `instruction_following.ifeval` | IFEval 官方规则检查，单轮指令遵循 GRPO | `ifeval` |

配置中的 `env` 选择具体环境，例如 `env=coding.swe`，分类由注册信息确定。
`ops6k`、`kernelmix`、`verified`、`lite` 等属于各环境的任务集，通过 `train_set` / `test_set` 选择。

已有 `env=cuda` / `env=swe` 配置以及 `arl.envs.cuda` / `arl.envs.swe` 导入仍然兼容。
独立环境包的路径已迁移到 `envs/coding/`；外部设置 `CUDAENV_ROOT` 或安装包时请使用新路径。

注册表可直接查询，查询时不会导入 Docker、SWE-bench 或具体环境的依赖：

```python
from arl.envs import get_env_spec, list_environment_categories, list_environments

list_environment_categories()              # ('coding', 'instruction_following')
list_environments('coding')                 # ('coding.cuda', 'coding.swe', 'coding.programbench')
list_environments('instruction_following')   # ('instruction_following.multicodeif', 'instruction_following.ifeval')
get_env_spec('swe').category                 # 'coding'
```

新增环境：将独立环境包放入对应的 `envs/<category>/`，在 `arl/envs/<category>/` 实现适配器，
并在该分类的 `ENVIRONMENTS` 中添加 `EnvironmentSpec`。适配器的 `make_source_from_config`
负责将训练配置转换为自己的 TaskSource 配置；公共训练循环不负责选择环境的构造参数。

详细用法：[CudaEnv](coding/CudaEnv/README.md)、[SweEnv](coding/SweEnv/README.md)。

环境与 harness 的配套关系和分阶段模型选择见 [实验配置](../experiments/README.md)。CUDA 默认 `arl_agent`，SWE/ProgramBench 默认 `mini_swe_agent`，IFEval 默认 `direct`，MultiCodeIF 默认 `multicodeif_self_repair`；五者均接入本地 on-policy GRPO。

首个 instruction following 训练环境：[IFEvalEnv](instruction_following/IFEvalEnv/README.md)，支持 strict 全约束 0/1 奖励或 fraction 比例奖励。

ProgramBench 已支持采集、原生 GRPO、评估及混合训练，详见 [ProgramBenchEnv](coding/ProgramBenchEnv/README.md)。

训练闭环：当前策略 → 对应 harness → 环境执行及判分 → token/logprob/mask + 终局 reward → 按题/轨迹计算 GRPO 优势 → PPO loss/优化器 → 同步新参数再采样。五环境共享策略配置见 `experiments/all_environments_grpo.yaml`。env_max_turns 未指定时使用各环境默认值：CUDA 16、SWE 250、ProgramBench 1000；MultiCodeIF 独立使用 multicodeif_max_rounds（默认 3）。
