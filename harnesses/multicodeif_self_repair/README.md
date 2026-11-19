# MultiCodeIF self-repair harness

标识：`multicodeif_self_repair`。状态：本地 collect / GRPO / evaluate 已接入；API/离线文本评测保留。

与 `instruction_following.multicodeif` 配套：首次生成只接收任务 prompt，后续失败轮使用官方反馈模板。
本地模型经公共 token 采样器生成，环境使用固定版本官方检查器给奖励；API 与离线预测走独立文本接口。
判分仍在 benchmark/worker 中完成，不由 harness 改写。

本地 collect 导出真实 token / old-policy logprob / loss mask；prompt、官方反馈与反馈中出现的旧代码
全部 mask=0，只有当前采样动作进入 loss。自修复按官方模板重新构造输入，前缀改变时产生新的 datum，
但同一轨迹的各 datum 共用终局奖励与 trajectory unit；GRPO 组内基线按同题不同轨迹计算。

默认最多三轮，全部约束通过即结束；strict 终局奖励为 0/1，fraction 使用最后一轮通过比例。
不可用/错误/不完整判分是 infra 并丢弃整条轨迹；截断回答终止并交由统一截断 mask 策略处理。
初始输入不暴露 evaluator 答案；只有失败后的官方反馈会提供其要求的约束信息。
多层约束链整体划分 train/test，限制题数不会重切划分，显式双文件拒绝重叠。

```bash
python3 -m arl.experiments experiments/multicodeif_grpo.yaml --phase train --check
python3 -m arl.experiments experiments/all_environments_grpo.yaml --phase train --check
```

本地路径可用于共享优化器与逐批参数同步；API/offline 文本输出没有训练轨迹，明确禁止用于 GRPO。
需要 LLM judge 或 Guesslang/工具链的约束必须配置对应依赖；启动前探测所选检查类别，缺失会报错。
目前奖励只评官方代码约束，不评代码功能或编译。已验证 CPU 优化器更新，尚未正式运行 GPU GRPO。
