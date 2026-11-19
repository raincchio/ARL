# Instruction following

此分类用于指令遵循任务及其环境、数据集和判分器。

`benchmarks/` 已按给定清单搭好 7 类、11 个独立 benchmark 目录：

- 通用指令遵循：IFEval、IFBench
- 复杂多约束：FollowBench、ComplexBench
- 多轮、多语言：Multi-IF
- 指令优先级：IHEval
- 结构化输出：IFStruct
- Agent 指令遵循：AgentIF
- 代码指令遵循：IFEvalCode、CodeIF、MultiCodeIF

默认调用 benchmark；MultiCodeIF 的数据和约束评测已部分接入，其余评测与全部训练入口仍待完成，当前没有训练 TaskSource。
用 `python3 -m arl.benchmarks --list --category instruction_following` 查看调用标识。

完整清单与调用约定见 [Benchmark 目录](../BENCHMARKS.md)。
