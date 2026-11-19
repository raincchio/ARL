# FrontierSWE

- 分类：`coding` / 仓库级开发（`repository_development`）
- 标识：`coding.frontierswe`
- 默认模式：`benchmark`
- 评测接入：待完成
- 训练接入：待完成

```bash
python3 -m arl.benchmarks coding.frontierswe
python3 -m arl.benchmarks coding.frontierswe --mode train
```

当前入口只提示待完成，不启动评测、下载数据或运行训练。后续在 `entry.py` 的
`benchmark()` 接入原 benchmark，在 `train()` 接入训练，并更新注册清单的接入状态。
