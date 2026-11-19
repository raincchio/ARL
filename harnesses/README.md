# Harness

Harness 决定模型的执行流程，环境负责题目与判分。具体配套由注册表校验，禁止未接通的 harness 静默退回其他实现。

| Harness | 配套环境 | 当前状态 |
| --- | --- | --- |
| arl_agent | CUDA 算子 / KernelBench（coding.cuda） | 原生采集、GRPO、评估已接入 |
| direct | IFEval | 单轮生成、官方判分及 GRPO 已接入 |
| multicodeif_self_repair | MultiCodeIF | 本地 token 采集、自修复 GRPO/评估已接；保留 API/离线文本评测 |
| mini_swe_agent | SWE 可选、ProgramBench 默认 | 本地采集、GRPO、评估已接入；GPU 训练尚未实跑 |
| codex | 外部 agent harness | 待完成 |
| claude_code | SWE 默认 | 真实 CLI + 本地 Qwen token 桥，采集 / GRPO / 评估 |

`direct` 是仓库内部的单轮生成适配名称，不是 lm-evaluation-harness。公共采样、token/logprob/mask 数据处理可复用，但不同 harness 的入口、环境配套及状态分别记录。

实验配置可分别选择 collect/train/evaluate 的 env、harness 和 model；见 [实验说明](../experiments/README.md)。mini-swe-agent 安装与配置差异见 [适配说明](mini_swe_agent/README.md)。
