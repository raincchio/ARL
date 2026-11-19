# CudaEnv — CUDA-Agent 工作区环境

分类：`coding`；训练环境名 `coding.cuda`（兼容 `cuda`）。训练适配器为 `arl/envs/coding/cuda.py`。

给 agentic RL 用的多文件 CUDA 优化环境，从 CudaAgent 的 `r3lm/envs/agent_workspace_env.py`
抽出来独立成包，**与训练框架解耦**（不依赖 torch / tokenizer / r3lm），评测后端升级到
CudaEval 的 `/repo/evaluate`（CudaTasks 题库）。

```
CudaTasks (题库: tasks/<set>/registry/<id>/)  ──►  CudaEnv (工具循环 + 奖励)  ◄──  训练器 (ARL)
                         ▲                                │ POST /repo/evaluate
                         └────── CudaEval (判分/防作弊/基线, GPU 集群) ◄┘
```

## 和 CudaAgent 版的差别

| | CudaAgent `AgentWorkspaceEnv` | CudaEnv |
|---|---|---|
| 评测端点 | `/workspace/evaluate`（已降为集群内部端点） | **`/repo/evaluate`**（对外唯一端点） |
| 题目 | jsonl 里的 `code`，客户端传 `ref_code` | CudaTasks 注册表 `instance_id`；ref/cfg/基线/受保护文件**全由服务端从注册表取** |
| 工具 | 6 个 | 同 6 个（schema、system prompt 逐字相同，RFT checkpoint 可直接用）+ 可选 `analyze`（ws_version 2 的 `acu` roofline 剖析） |
| 接口 | r3lm `Env`（token 级，绑 renderer） | **消息层**：`messages()` / `step(text)`，token 预算由训练器处理 |
| 历史 | Kevin 式：每轮剥思考 + 老轮次压缩（而且剥不掉，见下） | **harness 式**：思考进 `reasoning_content` 原样回放（Qwen3.8 模板缺省 `preserve_thinking`），历史逐字不变；只在超预算时逐级降级 |
| 奖励 | best-of-trajectory 里程碑 | 同（`reward_mode="best"`）；可选 `"final"` 交卷时按 SWE-bench 口径对最终工作区判分 |
| 留档 | — | `summary()` 带最好成绩时的工作区快照 `best_files` |

## 接口

```python
from cudaenv import CudaEvalClient, CudaWorkspaceEnv, EnvConfig, load_task_set

tasks = load_task_set("ops6k")            # 或 "kernelmix" / "kernelmix:l1" / "toy" / instances*.jsonl
client = CudaEvalClient()                 # CUDAENV_EVAL_ENDPOINT, 缺省 http://10.0.134.90:8000
env = CudaWorkspaceEnv(tasks[0], client, EnvConfig(max_turns=16))

await env.reset()
while not env.done:
    msgs = env.messages(aggressiveness=0) # system + 题面 + 完整历史 (含思考); 超预算再 +1 降级
    # 训练器: apply_chat_template(msgs, tools=env.tools) 超预算就 aggressiveness += 1 重取
    text, truncated = ...                 # 生成; truncated = stop_reason == "length"
    out = await env.step(text, truncated=truncated)
# out.reward ∈ {-1,1,2,3}; out.infra_error=True -> 丢样本; env.summary() 留档
```

- 一轮里没有任何工具调用的完整回复 = 交卷；截断且无工具调用的轮次不算交卷（烧一轮 + 提醒）。
- 每轮最多执行 4 个工具调用；路径白名单 `kernels/*.{cu,cuh,cpp,h}` + `model_new.py`。
- `profile`/`analyze` 服务端固定先 verify，进奖励的 speedup 一定是刚验证过的 .so 测出来的。

## 为什么不剥思考（Qwen3.8 模板）

Qwen3.8 的 `chat_template.jinja` 只认 assistant 消息的 `reasoning_content` 字段，**不会**从 content 里
拆 `</think>`（Qwen3 会）；`preserve_thinking` 缺省就是保留历史思考，也就是 agent 用的交错思考。
生成 prompt 以 `<think>\n` 结尾，采样回复里只有闭标签 `</think>`。CudaAgent 的环境拿只认成对标签的
正则去剥，一个都没剥掉，又把原文整段当 content 塞回，于是 RFT run1 的每个历史轮都渲染成了
"空 think 块 + 整段推理 + 游离的 `</think>`"。

这里按模板约定拆成 `reasoning_content` + `content`（`tools.split_reasoning`）原样回放。档 0 下
**下一轮 prompt 文本 = 上一轮 prompt + 本轮回复原文 + 工具结果**，训练器可以只追加 token，
一条轨迹合成一个训练序列（ARL 就是这么做的）。超预算时的降级档位：
1 = 老轮次（最近 `keep_full_turns` 轮之外）去掉思考；2 = 再压成一行摘要；3+ = 丢最老的轮次。

## 题库

训练用 `ops6k`（5019 题，`CudaAgent/data/ops6k/train.jsonl` 生成，带实测 eager/compile 基线），
评测用 `kernelmix`（280 题），冒烟用 `toy`。生成：

```bash
cd /root/workspace/CudaTasks && PYTHONPATH=.:../CudaEval python3 -m cudatasks.generate \
    --ops6k ../CudaAgent/data/ops6k/train.jsonl --set ops6k
```

注册表必须放在 CudaEval 主节点也能读到的共享盘上（`/root/workspace/CudaTasks/tasks` 是缺省猜测路径）。

## 自测

```bash
python3 tests/test_env_offline.py         # 离线 (假客户端), 8 项
python3 scripts/smoke_live.py --analyze   # 真集群: toy-axpby 标准解 + 作弊被拒, 约几分钟 (看排队)
```

## 已知限制

- 服务端 `/repo/evaluate` 把 `workspace_id` 固定成 `repo-<instance_id>`，同题的 GRPO 兄弟轨迹会粘到
  同一个评测节点（编译缓存命中更好，但一组 8 条挤一台 16 卡节点）。
- 工作区在内存里，没有 bash；要换成真实目录 + bash 沙箱，在 `env._execute` 加工具即可，评测仍走同一端点。
