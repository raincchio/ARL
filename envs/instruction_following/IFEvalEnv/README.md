# IFEval 训练环境

环境名 `instruction_following.ifeval`（别名 `ifeval`），默认 harness 为 `direct`。已接入原生 collect、GRPO train 和 evaluate：单条轨迹生成一次回答，不调用工具；ARL 现有采样器保存 token、旧策略 logprob 和 mask，环境只返回判分与奖励。

判分直接调用 [Google Research 官方实现](https://github.com/google-research/google-research/tree/master/instruction_following_eval)，固定 commit `e6890f85757dd84e27ca6df2dd30651dafad28e0`。上游检查代码、541 条公开题目和许可证按 SHA-256 校验后存入 `.cache/benchmarks/ifeval-upstream/`，不复制进版本库。检查器依赖在独立 venv 中安装，句子分割资源位于独立 NLTK 缓存。

```bash
# 仓库根目录；网络需要时配置 PIP_INDEX_URL / HTTPS_PROXY
./script/setup_ifeval.sh
# 当前机器内部 pip 源不可用时可显式选公网源
PIP_INDEX_URL=https://pypi.org/simple ./script/setup_ifeval.sh
# 官方数据及全部检查器预检，不加载模型
python3 -m arl.benchmarks ifeval
# 分别配置采集、训练和评估，先仅校验
python3 -m arl.experiments experiments/ifeval.yaml --phase train --check
python3 -m arl.experiments experiments/ifeval.yaml --phase collect
python3 -m arl.experiments experiments/ifeval.yaml --phase evaluate
# 按实际部署修改模型和 GPU 设置，再通过 GPU 调度运行
./script/train_ifeval_grpo.sh
```

[分阶段配置](../../../experiments/ifeval.yaml) 与 [训练启动配置](../../../script/train_ifeval_grpo.conf) 均默认使用 `/models/Qwen/Qwen3.6-35B-A3B`。实验入口沿用现有训练部署的 sampler/DeepSpeed 设置；启动脚本提供独立的训练/采样卡配置，不自行排队或选空闲卡。

## 奖励与回答

- `ifeval_reward_mode=strict`（默认）：官方 strict 全约束通过时奖励 1，否则为 0。
- `ifeval_reward_mode=fraction`：奖励为 strict 检查通过数 / 总约束数；correct 标签仍表示全部 strict 检查通过。
- 同时保存官方 strict/loose 的逐约束结果。最终回答会移除 `</think>` 之前的推理；示例默认关闭 thinking。
- 生成截断记 0，不因合格前缀获得奖励；截断 token 是否参与训练仍遵循公共 truncated_turn 设置。
- 检查器超时、依赖缺失或运行错误属于 infra_error，丢弃训练轨迹，不作为模型答错。启动前会用全部选定任务预检官方检查器。

这些规则检查验证可机器检查的指令约束，不能替代回答内容质量评估。

## 数据划分

`train_set=ifeval` 使用官方公开题目，也可指定同格式本地 JSONL；`test_set` 可指定独立文件。显式双文件会检查 key 和 prompt 重叠。单文件按固定 key 排序、seed 随机划分，再应用 task_limit/test_limit；同一数据集调整 limit 不会改变 held-out 身份。

默认 test_fraction=0.1、ifeval_split_seed=0 得到 **487 train / 54 held-out**。这里将公开 benchmark 题目用于训练，评估结果应标记为内部划分成绩；不能作为未训练过 IFEval 的全量 benchmark 成绩。用户已有独立训练集时可用 train_set/test_set 替换。

提示消息只含原始 prompt，不额外暴露 instruction_id_list、kwargs 等判分元数据；超出 prompt 预算时不裁剪题目。

配置项：`ifeval_python`、`ifeval_upstream` 可更换已安装的检查器 Python/官方文件目录，目录内容仍需通过固定哈希校验；`ifeval_timeout=60` 为每次 worker 超时，`eval_max_concurrency` 控制并发。

## 离线评测

```json
{"predictions":"/abs/predictions.jsonl","output_dir":"/abs/new_output"}
```

预测格式为 `{"key":1001,"response":"..."}` 或官方 `{"prompt":"原始完整 prompt","response":"..."}`。使用 `python3 -m arl.benchmarks ifeval --config config.json` 输出 strict/loose 的 prompt、instruction accuracy。默认 inspect；不连接 API 或启动训练。本地模型生成评估通过 experiments 的 evaluate 阶段执行。

验证：`python3 -m unittest discover -s tests -p test_ifeval.py -v` 覆盖数据划分、防重叠、官方正负判分、真实检查器到 token rollout/GRPO advantage、采集导出、截断与判分错误处理；无需 GPU。
