# Claude Code harness

标识：`claude_code`。支持 SWE 的本地 Qwen collect / GRPO train / evaluate；SWE 默认使用此 harness。
运行真实 Claude Code CLI **2.1.226**，不调用 Claude API、不替换成 mini-swe-agent。

## 运行与依赖

需要固定版本的 Claude Code 可执行文件和 Python `aiohttp`。当前机器的 CLI 为
`/root/.local/bin/claude`。依次查找 `ARL_CLAUDE_CODE_BIN`、PATH、该默认路径；
缺失或版本不匹配会在配置预检时报错。可用官方安装方式安装指定版本，或配置
`ARL_CLAUDE_CODE_BIN=/absolute/path/to/claude`。不自动升级 CLI。

```bash
python3 -m arl.experiments experiments/swe.yaml --phase collect --check
python3 -m arl.experiments experiments/swe.yaml --phase collect
python3 -m arl.experiments experiments/swe.yaml --phase train
python3 -m arl.experiments experiments/swe.yaml --phase evaluate
# 原 shell 训练入口也已切换；GPU 分配仍由该脚本的参数控制
./script/train_swe_grpo.sh
```

切换 harness 后使用新的输出目录，避免与旧 mini-swe-agent 的 run identity 混用。
显式 `harness=mini_swe_agent` 仍可运行原实现。ProgramBench 保持 mini-swe-agent。

## 本地模型与远程工具

每条轨迹创建独立的 Claude Code 进程、临时 HOME 和回环 HTTP endpoint。
CLI 请求 Anthropic Messages API 时暂停，ARL 将实际 system/messages/tool schema
转换成 Qwen chat template，由现有本地 on-policy sampler 生成。原始采样 token、
logprob 和 loss mask 继续走公共 rollout / trajectory GRPO 数据链；不从 CLI 文本
日志重建训练概率。CLI 中的模型路由名 `arl-qwen` 只指向本地桥，实际权重仍由
实验的 `models.policy.name` 或训练的 `POLICY` 决定。

Qwen 的 XML/Hermes 工具调用转换成 Anthropic tool_use（支持 JSON 和 SSE 响应），
由 Claude Code 调用唯一的 `mcp__swe__bash`。MCP 经现有 SWE master 在同一远程
`/testbed` 执行，结果交给 CLI，CLI 再发下一轮模型请求。最终文本回复、步数上限或
上下文预算结束时收集完整工作区 diff，调用原官方 grader，最终 resolved 为 0/1。
空补丁为 0；CLI、模型桥或工作区基础设施错误不作为模型负例。

CLI 使用 `--bare`、空 built-in tools、独立配置和严格 MCP 配置；不加载用户凭证、
hooks、plugins、项目 CLAUDE.md、skills 或会话历史。Read/Edit/Bash 等本机工具关闭，
读写文件、自测通过远程 bash 完成；不启用子 agent。此配置使用 Claude Code 的最小
模式和 MCP 工具流程，和完整交互式 Claude Code 的工具集不同。
模型格式错误或截断时不执行工具，提供重试反馈并计入轮数。CLI 管理自己的历史，
ARL 不静默压缩其请求；超出 prompt_budget_tokens 时结束轨迹。count_tokens 接口
为 CLI 的预算估算，训练 token 与窗口检查使用实际 Qwen tokenizer。

## 验证

```bash
python3 -m unittest discover -s tests -p test_claude_code.py
python3 script/smoke_swe_env.py \
  --endpoint http://10.0.134.70:18770 \
  --token-file /root/ai_data/swe/service/token \
  --dataset remote:/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl \
  --harness claude_code --negative
```

单元/集成测试启动真实 CLI，以脚本化采样和假工作区检查工具反馈、格式错误、终止、
基础设施失败、原始 logprob、同题 GRPO 分组和优势。上面的远程冒烟使用可信负例，
不表示 Qwen 的解题成绩。正式 GPU Qwen 训练需另行实跑。

CLI 与 MCP 参数参考：[CLI reference](https://code.claude.com/docs/en/cli-reference)、
[MCP](https://code.claude.com/docs/en/mcp)。本地 Qwen 适配由本仓库维护；Anthropic
[不支持通过 gateway 路由到非 Claude 模型](https://code.claude.com/docs/en/llm-gateway)。
