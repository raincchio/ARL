# KernelBench

- 分类：`coding` / CUDA 算子优化（`kernel_optimization`）
- 标识：`coding.kernelbench`
- 默认模式：`benchmark`
- 评测接入：待完成
- 训练接入：待完成

相关环境基础：`coding.cuda`。现有训练环境不代表本 benchmark 的评测或训练入口已完成。

```bash
python3 -m arl.benchmarks coding.kernelbench
python3 -m arl.benchmarks coding.kernelbench --mode train
```

当前入口只提示待完成，不启动评测、下载数据或运行训练。后续在 `entry.py` 的
`benchmark()` 接入原 benchmark，在 `train()` 接入训练，并更新注册清单的接入状态。
