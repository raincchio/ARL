# IHEval

- 分类：`instruction_following` / 指令优先级（`instruction_hierarchy`）
- 标识：`instruction_following.iheval`
- 默认模式：`benchmark`
- 评测接入：待完成
- 训练接入：待完成

```bash
python3 -m arl.benchmarks instruction_following.iheval
python3 -m arl.benchmarks instruction_following.iheval --mode train
```

当前入口只提示待完成，不启动评测、下载数据或运行训练。后续在 `entry.py` 的
`benchmark()` 接入原 benchmark，在 `train()` 接入训练，并更新注册清单的接入状态。
