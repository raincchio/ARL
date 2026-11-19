# Codex harness

标识：`codex`。状态：**接入待完成**。

此目录用于 Codex 的运行适配、prompt/tool 配置和输出转换。
`entry.py:run()` 是统一调用入口，当前明确提示待完成，不启动 agent 或安装外部工具。

Benchmark 决定题目和评分；harness 决定 agent 如何调用模型、使用工具和管理轨迹。
各自独立选择，不将某个模型或 benchmark 固定绑定到本 harness。

统一约定见 [Harness 目录](../README.md)。
