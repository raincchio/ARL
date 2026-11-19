# MultiCodeIF

分类：`instruction_following` / 代码指令遵循，同时与 coding 相关。标识：`instruction_following.multicodeif`（短名 `multicodeif`）。

**支持文本轨迹采集和部分评测，GRPO 训练接入待完成。** 已支持官方数据读取、静态约束判分、可配置独立 LLM judge、API 生成、离线预测和多轮 self-repair。默认命令检查数据清单，不启动模型；指定模型或预测文件后执行评测。

本接入检查指令约束，不执行生成代码，不汇报功能正确率或编译通过率。报告明确记录 `functional_correctness=not_evaluated`、`compilation=not_evaluated`，不能把约束通过率当作完整 coding benchmark 成绩。

## 官方实现与依赖

[官方仓库](https://github.com/SYSUSELab/MultiCodeIF)，固定 commit `faafada5356ce81d228793f3787b27615cb763bf`。运行时复用其检查函数和 self-repair 模板，不复制或修改官方源码。上游快照和依赖安装在被 git 忽略的 `.cache/benchmarks/`。

```bash
./script/setup_multicodeif.sh
python3 -m arl.benchmarks multicodeif
```

`setup_multicodeif.sh` 仅克隆固定版本及安装静态检查依赖，不修改训练环境的 site-packages。可按机器网络配置设置 `PIP_INDEX_URL` 和代理。当前快照实际读取到单层 1,526 条、四个层级共 660 条（165 条链）；报告使用实际数据数量，不套用 README 中的规模描述。

兼容处理在独立 worker 中完成：

- 上游 eager Guesslang 导入改为按需导入；没有 Guesslang/TensorFlow 的 `language_type` 检查记为 unavailable。
- 使用 Tree-sitter 0.25.2 支持上游语法包的 ABI 15，通过兼容层提供原有 `Language.query(...).captures(...)` 接口。语法包版本沿用上游锁定值。
- 上游写死的生成 API 改接配置的 judge，仍使用官方判分提示词；judge 必须明确返回 YES/NO，异常输出记为评测错误。
- 未配置 judge、未接入语言版本工具链、检查器异常都单独报告，不当作模型回答错误，不把未运行检查当作通过。
- 保留官方检查语义；例如当前 Rust 数据结构 size 检查存在大小写匹配限制，不在适配层私自修正结果。因此报告还需要结合具体检查和错误明细阅读。

## API 生成与多轮修复

```bash
python3 -m arl.benchmarks multicodeif \
  --config envs/instruction_following/benchmarks/code/multicodeif/config.api.example.json
```

先将示例里的 `base_url`、`model` 改为实际 OpenAI 兼容服务。`base_url` 应以 `/v1` 结尾；本接入不自动使用外部付费服务。`MULTICODEIF_API_KEY` 仅在服务需要鉴权时设置。模型目录 `/models/Qwen/Qwen3.6-35B-A3B` 的本地小规模评测可通过 `script/eval_multicodeif_local.py` 在集群调度器分配的 GPU 上运行；脚本持有共享 GPU 租约，并在结束后回收自己的临时推理服务。

`max_rounds=3` 表示总共最多三次回答。首次只给模型任务 prompt，不泄露 expected answer；失败后复用官方 `self_repair/repair.py` 的反馈模板，提供失败约束、上一轮代码和原始任务。已通过的任务不再生成；无法完成判分的任务不伪造修复信号。多轮离线预测也可以通过 `responses` 提供。

这里的轮次是同一任务的 self-repair；`suite=multi_level` 则是逐级累加的约束集。两者可分别使用或组合。多层数据按完整的前缀链判分，而不只检查最后一条新增约束。

需要语义 judge 的项目可在配置中增加：

```json
{
  "judge_base_url": "http://your-judge-service/v1",
  "judge_model": "your-independent-judge",
  "judge_api_key_env": "MULTICODEIF_JUDGE_API_KEY"
}
```

按需设置环境变量 `MULTICODEIF_JUDGE_API_KEY`。目标模型与 judge 分开配置；没有独立 judge 时，相关项目保留 unavailable。judge 模型和 API 生成参数都会记录在 provenance 中。

## 离线预测

先导出只包含 task ID 和 prompt 的任务，不含判分答案：

```bash
python3 -m arl.benchmarks multicodeif \
  --config envs/instruction_following/benchmarks/code/multicodeif/config.export.example.json
```

预测为 JSON 数组或 JSONL，每条按 task ID 对齐：

```json
{"task_id":"single_level/code_convention_naming_convention:1","response":"class DatabaseConnector:\n    def connect(self): pass\n    def disconnect(self): pass\n    def execute_query(self): pass"}
```

多轮离线格式用 `responses: ["first answer", "repair answer"]`。缺预测、重复 task ID 或缺修复轮回答会明确报错/记录错误，不默默错位。也可将 `predictions` 指向**单个模型**的官方生成目录（如上游 `results/single_level/deepseek-r1`），读取其 `response=[[prompt, code]]` 格式。

将 `config.offline.example.json` 中预测路径改好后执行：

```bash
python3 -m arl.benchmarks multicodeif \
  --config envs/instruction_following/benchmarks/code/multicodeif/config.offline.example.json
```

## 报告与范围

每轮落盘 `trajectories.jsonl` 与 `report.json`，另有固定版本及参数的 `provenance.json`。默认输出 `runs/multicodeif_<timestamp>_<id>/`；指定 `output_dir` 时目录必须尚不存在，防止覆盖上次结果。

- `checks_requested` / `checks_scored` / `coverage`：请求检查数、实际完成数和覆盖率。
- `constraint_accuracy_scored`：已完成检查中的约束通过率。
- `all_constraints_passed_scored`：完成判分的任务中，全部约束同时通过的比例。
- `complete=false`：存在 unavailable、评测错误或生成错误。此时的分数只代表已评分子集。
- `functional_correctness` / `compilation`：当前均为 `not_evaluated`。

按约束、语言、limit 选出的子集结果，以及不同 judge / 采样设置，均不能直接作为论文完整复现成绩。

```bash
python3 -m unittest discover -s tests -p test_multicodeif.py
python3 -m arl.experiments experiments/multicodeif_grpo.yaml --phase train --check
```

## Harness 与分阶段实验

默认 harness 为 `multicodeif_self_repair`，可通过 benchmark CLI 的 `--harness` 显式指定。采集与评估可各自指定模型，见 [实验配置](../../../../../experiments/README.md) 和 `experiments/multicodeif.yaml`。采集保存实际输入 messages、回答与约束反馈，可用于后续 SFT/蒸馏数据构造；当前不包含 SFT trainer。

MultiCodeIF 本地训练入口已接通：`experiments/multicodeif_grpo.yaml`。环境位于 `arl/envs/instruction_following/multicodeif.py`，由 multicodeif_self_repair harness 使用本地采样器产生 token/logprob/mask；官方失败反馈与旧代码仅作为下一轮 prompt，mask 为 0。每题多条独立轨迹构成 GRPO 组，同一修复轨迹的多个 datum 共享终局奖励和 trajectory unit。默认 strict 0/1，可选 fraction；不可用/错误判分丢弃。API/offline 文本接口继续独立保留，不能用于 GRPO 更新。

直接 benchmark 默认使用全部选定数据；实验入口的 collect 默认 train、evaluate 默认 test，通过 task_split/test_fraction/split_seed 确定性划分，多层约束的同一链保持在同一侧。实验配置使用顶层 split 指定，避免与 options 重复。
