# 实验：分别指定 env、harness 和模型

`python3 -m arl.experiments CONFIG --phase collect|train|evaluate` 每次运行一个阶段，省略 phase 默认 evaluate。阶段独立配置，可以继承 defaults，再覆盖 env、harness、model、split 和 options。模型名引用 models 中的配置。

```bash
# 只检查配置与能力，不创建模型或工作区
python3 -m arl.experiments swe.yaml --phase collect --check
# 查看目录中已配置的配套关系
python3 -m arl.experiments --matrix
# 从仓库根目录执行示例
python3 -m arl.experiments experiments/all_environments_grpo.yaml --phase train --check
python3 -m arl.experiments experiments/swe.yaml --phase collect
python3 -m arl.experiments experiments/swe.yaml --phase train
python3 -m arl.experiments experiments/swe.yaml --phase evaluate
```

第一条命令需在本目录执行，其余执行示例假设位于仓库根目录。配置中本地文件及 output_dir 相对配置文件所在目录解析；模型权重路径、远程数据集路径按原样传递。示例机器地址和 API 地址需要按部署调整；训练不会自动启动。

| Env | 默认 harness | collect | train | evaluate |
| --- | --- | --- | --- | --- |
| coding.cuda | arl_agent | token 轨迹 | GRPO | 支持 |
| coding.swe | claude_code | token 轨迹 | GRPO 接入 | 官方 resolved 判分 |
| coding.programbench | mini_swe_agent | token 轨迹 | GRPO 接入 | 官方提交包判分 |
| instruction_following.ifeval | direct | token 轨迹 | GRPO | 官方 strict/loose 检查 |
| instruction_following.multicodeif | multicodeif_self_repair | 本地 token 轨迹；API 文本反馈 | 本地自修复 GRPO | 官方约束检查、自修复 |
| 其他 benchmark | 待配置 | 待完成 | 待完成 | 待完成 |

SWE 已默认使用真实 Claude Code 2.1.226 CLI，经本地 token 桥连接 Qwen，见 [Claude Code 适配说明](../harnesses/claude_code/README.md)。Codex 的独立入口仍待完成。mini-swe-agent 固定 2.4.6，通过本地采样器连接真实上游 DefaultAgent；先运行 `bash script/setup_mini_swe_agent.sh`，依赖缺失或版本不符会在预检报错。`external_harness_pending.yaml` 保留文件名，现演示 collect 使用 mini-swe-agent、evaluate 使用 Claude Code 的独立选择。

`arl_agent` 仅用于 CUDA 算子/KernelBench，本地模型；IFEval 使用独立的 `direct` 单轮生成 harness。`multicodeif_self_repair` 支持 local token 采样训练，另保留 OpenAI-compatible API 和显式 offline 文本评测。兼容性由 env 的 supported_harnesses、接口及 harness 的 operations、model_backends、trace_fields 共同校验。

## 采集与训练数据

当前训练统一使用“当前策略 → 环境对应 harness → 环境执行/官方判分 → GRPO 优势 → PPO 更新 → 同步新参数”的在线循环。
`collect` 的 datums.jsonl 用于轨迹审计/导出，`train` 每批重新用当前策略采样，不默认重放历史采集文件或 API 文本。
环境返回任务消息和终局 reward；harness 管理交互；公共 rollout 保留采样 token/logprob/mask，提示词、工具结果与修复反馈不进入 loss。
缺失/不齐/非有限 logprob 的轨迹在执行工具前丢弃；采样失败或官方判分不完整也丢弃，不当成模型负例。
算法入口再次检查 token 对齐、同题分组、trajectory unit 与奖励标签一致性。

[五环境混合配置](all_environments_grpo.yaml) 已明确每个环境的默认 harness。一个全局 batch 选择一个环境，各 rank 同步执行；
共享模型/优化器，每次有效更新后同步参数，下一批才开始采样。环境权重控制选批概率，不缩放 reward/loss。
`env_max_turns` 默认 None，按 CUDA 16、SWE 250、ProgramBench 1000 继承环境预算，显式配置仍可覆盖；
IFEval 单轮，MultiCodeIF 使用独立 multicodeif_max_rounds。

原生 collect 只采样，不创建优化器。输出包括 experiment.json（env/harness/model 身份）、invocations/（每次调用配置）、manifest.json（状态与能力）、datums.jsonl、轨迹与汇总。每个 datum 保存输入 token、目标 token、旧策略 logprob、loss mask、奖励、正确性以及任务/轨迹标识。失败的采集会将 manifest 标为 failed。

MultiCodeIF 本地 collect 保存 datums.jsonl，包含真实 token/old-policy logprob/loss mask；本地 train 由同一 harness 在线采样、自修复及官方约束判分，并在每次优化后同步参数。见 [本地 GRPO 配置](multicodeif_grpo.yaml)。原有 [API 配置](multicodeif.yaml) 的 collect 仍保存 samples.jsonl，用于文本数据整理；API/offline 输出明确不能用于 GRPO，未提供 SFT trainer。

当前 train 执行原生 on-policy GRPO：训练时重新采样，更新参数。它不会自动读取之前 collect 的文件。`validate_collection(..., require_training_trace=True)` 能检查配套关系、状态、数据对齐及 token 轨迹，但不构成离线 PPO 导入入口。

## 数据划分与身份

collect/train 默认 split=train，evaluate 默认 split=test；原生环境仅支持这两类，MultiCodeIF 可显式指定 all 用于完整 benchmark。MultiCodeIF 按 split_seed 和 test_fraction 确定划分，多层约束的同一链保持在同一侧，limit 在划分之后应用。比较阶段需保持相同数据源、筛选条件和划分参数。官方 SWE Verified 数据自身名为 test；这里的 train/test 是实验内部再划分。

collect/evaluate 要求新的输出目录，防止混入旧结果。train 可恢复同身份目录，拒绝不同 env/harness/model 身份。分布式训练必须指定共享 output_dir，并使用原有 torchrun/训练部署所需的 GPU 和 sampler 设置；配置文件本身不分配 GPU。

`models.*.api_key_env` 引用密钥环境变量。options 不允许覆盖模型和路由，也不允许隐式导入离线训练数据。离线预测必须显式使用 backend=offline。

IFEval 分阶段配置见 [ifeval.yaml](ifeval.yaml)。先运行 `script/setup_ifeval.sh` 安装隔离检查器；公开题目内部划分为 487/54，训练后的内部评估与完整公开 benchmark 需区分记录。

## 多环境混合训练

`train.environments` 接受多个环境及其配套 harness、weight、options。完整配置见 [mixed.yaml](mixed.yaml)，示例将 SWE 与 IFEval 按 7:3 的批次采样概率混合，共享一个模型和优化器。

```yaml
train:
  model: policy
  output_dir: ../runs/mixed_train
  options:
    batch_size: 8
    group_size: 8
    mixture_steps_per_epoch: 100
  environments:
    - id: swe
      env: coding.swe
      harness: mini_swe_agent
      weight: 7
      options:
        train_set: /abs/swe.jsonl
        max_tokens: 4096
    - id: ifeval
      env: instruction_following.ifeval
      harness: direct
      weight: 3
      options:
        train_set: ifeval
        enable_thinking: false
        max_tokens: 2048
```

```bash
python3 -m arl.experiments experiments/mixed.yaml --phase train --check
# 配置现有训练/采样 GPU 部署后执行；多 rank 使用 torchrun
python3 -m arl.experiments experiments/mixed.yaml --phase train
```

- **每批一个环境**：所有 rank 使用相同的环境选择和全局题目批次，再将题目分摊到各 rank。每题的 GRPO 组独立计算，不把不同环境的题放进同一组。
- **权重是选择批次的概率**，会归一化，不是 loss 或 reward 的乘数，也不保证短跑严格满足比例。环境样本耗尽时重新打乱并循环使用；小题库会被重复采样。各环境原始奖励保持自己的口径，不做跨环境奖励缩放。
- **训练长度**：mixture_steps_per_epoch 指定每轮批数，默认 0 表示 ceil(各环境训练题数总和 / batch_size)。混合训练中的 epoch 是这段采样计划，不代表每个题库完整遍历一遍。
- **可复现与恢复**：shuffle_seed、num_epochs、mixture_steps_per_epoch、batch_size、环境配置和数据顺序不变时，重新构造相同批次计划；start_batch 跳过其前缀。修改这些设置会改变续跑含义，身份检查会拒绝部分关键计划参数变化。optimizer 是否恢复仍由 resume_optimizer/load_checkpoint_path 指定。
- **配置范围**：模型、学习率、优化器、batch_size 和 checkpoint 配置属于共享 options；数据源、划分、harness、奖励模式、group_size、生成与 prompt 预算可按环境覆盖。允许同一 env 配不同题库，但 id 必须唯一；只接受目前已接通训练的组合。不同环境的生成上限扩容状态分别保存，采样窗口覆盖各自的 prompt+生成预算。
- **记录与评估**：轨迹附带具体 env/harness/environment_id。训练指标增加 env/<id>/...，周期评估依次运行所有环境自己的 held-out 集，输出 eval/<id>/...。不合并不同环境的奖励为统一 benchmark 分数。
- 现有单环境 `train.env` 写法保持兼容。显式 train.env 与 train.environments 二选一。此版本的 environments 列表用于训练；独立 collect/evaluate 阶段仍各自选一个 env。周期评估所有训练环境不改变独立 evaluate 阶段的选择。

示例仍需先完成 IFEval setup，并保证 SWE Docker 服务可用。CUDA 算子环境也可作为 mixture 成员，需提供其 CudaEval 服务和数据源。

ProgramBench：[单环境配置](programbench.yaml)、[SWE 50% + IFEval 30% + ProgramBench 20% 混合配置](mixed_programbench.yaml)。ProgramBench 示例先选 3 个已准备任务做 2/1 内部分割，正式使用前按需要准备并选择更大的任务集。

ProgramBench、SWE 及其混合示例已通过 mini_swe_agent 的预检。collect 导出实际 token/logprob/mask，train 复用共享优化器及更新后的采样参数同步；evaluate 使用相同上游 agent 和各环境官方评分。

当前 harness 边界：arl_agent 仅用于 coding.cuda；IFEval 使用 direct；SWE 默认 claude_code（可选 mini_swe_agent），ProgramBench 使用 mini_swe_agent。真实上游 agent、SWE+IFEval 混合调度、共享优化器与参数同步调用已通过 CPU 联调；没有启动真实 GPU 训练。两者分别加载固定版本官方 swebench.yaml / programbench.yaml 的提示词及工具反馈。SWE 示例为 250 步 / 命令 60 秒、输出 patch；ProgramBench 为 1000 步 / 6 小时 / 命令 180 秒 / 20 CPU / 60GB、归档整个工作区。本地采样、预算及隔离差异见 [mini-swe-agent 说明](../harnesses/mini_swe_agent/README.md)。
