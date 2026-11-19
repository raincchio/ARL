# ProgramBench

- 分类：coding / project_generation
- 标识：coding.programbench（别名 programbench）
- 默认 harness：mini_swe_agent（本地 collect/train/evaluate 已接入，GPU 训练尚未实跑）
- 状态：官方任务清单、源码提交包评测、底层环境协议已接入；agent collect/GRPO/evaluate 通过实验入口接入
- 官方版本：v1.2.5，cleanroom 镜像 task_cleanroom_v6

```bash
./script/setup_programbench.sh
python3 -m arl.benchmarks programbench
python3 -m arl.experiments experiments/programbench.yaml --phase train --check
```

默认仅 inspect 数据清单，不启动模型。现有提交包使用 config 中的 submissions_dir 在 Docker 宿主机评分；本地模型生成/训练使用实验入口。完整准备方式、奖励与数据划分见 [ProgramBenchEnv](../../../ProgramBenchEnv/README.md)。
