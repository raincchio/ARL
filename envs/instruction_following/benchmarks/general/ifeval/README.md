# IFEval

- 分类：instruction_following / general
- 环境与 benchmark 名称：`instruction_following.ifeval`，别名 `ifeval`
- Harness：`direct`
- 训练：已接入原生 GRPO
- 评测：支持官方 strict/loose 离线判分；本地模型生成评估通过实验入口

```bash
./script/setup_ifeval.sh
python3 -m arl.benchmarks ifeval
python3 -m arl.experiments experiments/ifeval.yaml --phase train --check
```

完整数据划分、奖励、训练启动和验证说明见 [IFEvalEnv](../../../IFEvalEnv/README.md)。默认 benchmark 仅检查数据与判分器；训练需显式提供实验配置或 model_name/log_path。
