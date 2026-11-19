# LiveCodeBench v6

- 分类：`coding` / 竞赛编程（`competitive_programming`）
- 标识：`coding.livecodebench_v6`
- 默认模式：`benchmark`
- 评测接入：待完成
- 训练接入：待完成

```bash
python3 -m arl.benchmarks coding.livecodebench_v6
python3 -m arl.benchmarks coding.livecodebench_v6 --mode train
```

当前入口只提示待完成，不启动评测、下载数据或运行训练。后续在 `entry.py` 的
`benchmark()` 接入原 benchmark，在 `train()` 接入训练，并更新注册清单的接入状态。
