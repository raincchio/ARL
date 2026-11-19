# mini-swe-agent harness

标识：`mini_swe_agent`。状态：**本地模型 collect / train / evaluate 已接入**。

固定上游 mini-swe-agent **2.4.6**。`entry.py` 通过 `arl/harnesses/mini_swe.py` 调用真实的
`DefaultAgent.step()`、上游动作校验、观察格式化和提交检测。模型由 ARL 本地采样器驱动，
工具连接现有 SWE / ProgramBench 隔离工作区，奖励仍由各环境的官方判分器给出。

```bash
bash script/setup_mini_swe_agent.sh
python3 -m arl.experiments experiments/programbench.yaml --phase collect --check
python3 -m arl.experiments experiments/swe.yaml --phase train --check
```

依赖仅下载到 `.cache/benchmarks/mini-swe-agent`；使用运行时已有的 pydantic>=2、jinja2、
rich、python-dotenv、platformdirs，不修改共享 Python。缺失或版本不符在预检报错。

SWE 直接加载该发行包的 `config/benchmarks/swebench.yaml`，并校验 SHA256
`9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e`。
system / instance 提示词按官方模板原样渲染，task 只传公开的 problem_statement，
不插入 ARL 的仓库说明、提示词或轮数描述。使用官方 bash 工具、结构化工具结果、
长输出截取和格式错误模板。上游消息保留 API 格式；渲染给 Qwen 时将工具参数转为字典。

SWE 按官方提示词创建并检查 `patch.txt`，然后执行
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`。
只评分该命令实际输出的 patch，不自动收集整个工作区 diff。
纯文本、原生 submit 工具及截断命令不会执行提交；未提交就达到步数/上下文上限或格式
错误上限时，按空提交计分。完整历史不压缩。

SWE 示例已设为官方的 250 步、每条命令 60 秒；环境变量及 bash 解释器也从官方配置读取，
包括 `BASH_ENV=/root/.bashrc`。`env_max_turns`、`swe_command_timeout` 仍可显式覆盖，
轨迹记录实际预算；env_max_turns 未指定时继承 SWE 默认 250 步。

保留的本地运行差异：使用 Qwen 本地采样器及 chat template，API 3 美元预算关闭
（不虚构本地美元费用），生成/context 上限仍由实验配置指定，Docker 保留现有隔离限制。
SWE evaluate 示例使用 temperature=1、top_p=1、top_k=-1；上游 v2 不设置 API temperature，
各服务默认值可能不同。因此这里对齐官方提示词与执行/提交协议，不宣称完整复现榜单运行。
上游配置哈希、profile 及实际覆盖值均写入轨迹。

ProgramBench 直接加载同一发行包的 `config/benchmarks/programbench.yaml`，SHA256 为
`9c2cc36b69c53e83cb4f60b0aa809768996373f4c3dddb3a202b78e93899bf8e`。
官方 system / instance 提示词原样渲染，包括参考程序使用边界、compile.sh、Git 提交要求；
system information 使用上游同源的控制端 `platform.uname()`。启动时初始化官方 Git 身份，
命令使用官方环境变量及默认 `bash -lc`。工具反馈、长输出、剩余步数/时间警告均使用官方模板。

ProgramBench 单环境/混合/训练部署示例改为 1000 步、6 小时、命令 180 秒、20 CPU / 60GB；
容器存活 7 小时。`env_max_turns`、`programbench_command_timeout`、资源仍可显式覆盖，
轨迹记录实际值。env_max_turns 未指定时继承 ProgramBench 默认 1000 步。
通过独立的 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 结束，随后归档整个工作区；
达到步数/时间上限时也沿用上游的归档行为，不使用 SWE 的 stdout patch 语义。

ProgramBench 同样保留本地 token 采样、生成/context 预算及更严格的 Docker 隔离；官方
agent cost_limit 本来就是 0。evaluate 示例 temp=1 / top_p=1 / top_k=-1，官方 API 未设置这些
参数，不能把服务默认值视为一定相同。评分使用固定官方 ProgramBench v1.2.5；尚未接入
自动作弊审计，行为测试通过率不等同于完整 leaderboard 合规审核。

collect 导出实际采样的 token_ids / logprobs / loss_mask；GRPO 复用现有优化器与每次更新后
的采样参数同步。消息、上游版本、agent 类型、环境身份和官方评分保存在轨迹中。
目前仅支持 local 后端，未提供独立 OpenAI-compatible API 运行入口。

Benchmark 决定题目和评分；harness 决定 agent 如何调用模型、使用工具和管理轨迹。
各自独立选择，不将某个模型或 benchmark 固定绑定到本 harness。

统一约定见 [Harness 目录](../README.md)。

CPU 验证覆盖真实上游 agent、token 导出、GRPO 优势及混合训练共享优化器/参数同步调用；
尚未进行真实 GPU 优化器训练。远程 Docker 冒烟使用合成响应，不能当作模型成绩。
离线数据检查与提交包判分仍不执行 agent。`arl_agent` 仅适用于 CUDA / KernelBench。

上游接口说明：[DefaultAgent](https://mini-swe-agent.com/latest/reference/agents/default/)。
