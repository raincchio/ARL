# ProgramBench 环境

`coding.programbench`（别名 `programbench`）默认且当前配置的 agent harness 为 `mini_swe_agent`，已通过真实上游 DefaultAgent 和本地 token bridge 接通 collect/train/evaluate。环境提供独立 Docker 工作区与官方判分；不允许使用 arl_agent 替代。

使用 [ProgramBench 官方 v1.2.5](https://github.com/facebookresearch/ProgramBench/tree/v1.2.5)，commit `27f02157c785f8da3647aa6dbbe6b9137f99f10e`，镜像标签 `task_cleanroom_v6`。启动前验证官方 Python 源码、200 个任务的 task.yaml/tests.json 哈希；排除上游 testorg 测试夹具。数据中的 ignored branches/tests 由官方评分器处理，不自行重写测试通过率计算。

## 运行机制

每条轨迹创建独立容器，以 `agent` 用户在 `/workspace` 操作。容器 `--network none`，不挂载宿主机源码、Docker socket、测试或密钥。默认 mini-swe-agent 使用固定版本官方 programbench.yaml 的提示词、单个 bash 工具及结构化工具反馈，观察参考 `./executable` 及文档，编写原始代码和 `compile.sh`，按提示提交 Git 改动并通过 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 结束。提示明确禁止取得原项目源码、复用/包装参考二进制及对参考程序反编译、反汇编、跟踪。底层环境 bash/submit 工具仍保留用于独立协议测试。

提交时按官方方式归档完整工作区，在新容器中调用 `_evaluate_instance` 编译并运行行为测试。参考 executable 无读取权限时 tar 可以返回非零；只要产生归档，继续官方评测，并保留 archive_warning。这与现有官方 baseline 导出方式一致。

- 默认 `programbench_reward_mode=fraction`：官方非忽略测试通过比例，范围 0–1。
- `strict`：全部通过才得 1 分。
- correct/resolved 始终表示全部有效测试通过，不受奖励模式影响。
- compile_failed / copy_executable_failed 属于模型提交失败，计 0；官方测试缺失、分支错误、system_error、超时及评分不一致作为 infra_error，丢弃训练轨迹。
- 默认 mini-swe-agent 已验证 token、旧策略 logprob、mask 与 GRPO 数据路径；真实 GPU 训练尚未实跑。官方判分细节只存到可信日志，不反馈给模型。

当前不包含官方 baseline 的自动源码复用/作弊审计；分数表示行为测试结果，不等同于已完成官方 leaderboard 合规审核。默认 mini-swe-agent 已对齐 2.4.6 官方提示词、工具反馈、Git 初始化和 agent 生命周期；本地采样、生成/context 上限及额外隔离差异见 [harness 说明](../../../harnesses/mini_swe_agent/README.md)。底层原生工具循环用于环境测试，不能当作默认 agent 执行路径。

## 环境准备

本机安装官方 Python 包到隔离缓存（不启动模型、不下载镜像或全量测试 blob）：

```bash
PIP_INDEX_URL=https://pypi.org/simple ./script/setup_programbench.sh
python3 -m arl.benchmarks programbench
```

实际 agent/评分需要 Linux x86_64 Docker 宿主机。宿主机需安装同一官方版本，预先准备所选 `programbench/<id>:task_cleanroom_v6` 镜像及官方 `ProgramBench-Tests` blobs。`programbench_blob_dir` 应指向每任务 blob 目录的父目录。可使用官方 blob sync 或 HuggingFace 固定 revision 下载；本次复用宿主机 blob revision `de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5`，预检会记录实际 revision。评分不在线下载测试。

单环境 YAML 的采集、训练和评估均选择 mini_swe_agent，已通过预检；先运行 `bash script/setup_mini_swe_agent.sh`。真实 GPU 训练尚未实跑。提交包离线评分不运行 agent，报告会注明 harness_execution=not_run_offline_grading。

当前示例复用已部署的 `root@10.0.134.90`、ProgramBench venv 和 blobs；修改配置即可使用本地 Docker 或另一台已准备的机器。

```bash
# 远程数据清单，不启动工作区
python3 -m arl.benchmarks programbench --config envs/coding/benchmarks/project_generation/programbench/remote_inspect.json
# 检查训练配置，不连接 Docker 或加载模型
python3 -m arl.experiments experiments/programbench.yaml --phase train --check
# 启动脚本含 GPU 部署设置，需按集群规则调度
./script/train_programbench_grpo.sh
```

示例配置：[单环境](../../../experiments/programbench.yaml)、[SWE + IFEval + ProgramBench 混合](../../../experiments/mixed_programbench.yaml)、[训练部署](../../../script/train_programbench_grpo.conf)。混合示例按 50% / 30% / 20% 的训练批次概率采样，已通过配套预检；每个环境共享优化器但分别记录指标。

## 数据与预算

全量 200 题，单文件按固定 seed 切分，默认环境配置留出 10%（180/20）；裸 GRPO 的通用比例仍为 2%，示例会覆盖。`programbench_task_ids` 可用逗号分隔选择子集；train_set/test_set 可指定本地 JSON ID 列表或 JSONL instance_id 记录。显式双文件拒绝重叠，先划分再应用 task_limit/test_limit。

首轮示例只选 3 个已准备的任务，按 0.34 切成 2 train / 1 held-out。Agent 采用官方 1000 步、6 小时、命令 180 秒；容器 20 CPU / 60GB、存活 7 小时。每轮生成 4096、prompt 预算 32768 是本地采样预算，每 rank 1 个活跃工作区。预算覆盖值会写入轨迹；env_max_turns 未指定时继承 ProgramBench 默认 1000 步。移除 task_ids 后可用全量，但必须先准备全部镜像和 blobs。训练过的公开题目及内部 held-out 成绩应分别标记，不能当作未见的全量 benchmark 结果。

预检验证选定任务的镜像和 blob 目录存在；缺少时明确报错，不自动拉取大量镜像。`programbench_eval_cpus`、`programbench_branch_workers` 控制单任务官方判分资源；并发工作区数是每 rank 的 eval_max_concurrency，所有 rank 会叠加。

## 现有提交包的评测

`submissions_dir` 是 **Docker 宿主机路径**，格式为 `<dir>/<instance_id>/submission.tar.gz`，配置例如：

```json
{
  "action": "evaluate",
  "ssh_host": "root@10.0.134.90",
  "python": "/mnt/ai_data/benchmarks/programbench/.venv/bin/python",
  "blob_dir": "/mnt/ai_data/benchmarks/programbench/blobs",
  "work_dir": "/mnt/ai_data/arl/programbench_runs",
  "task_ids": "lh3__seqtk.94e7070",
  "submissions_dir": "/host/path/to/submissions",
  "output_dir": "/local/path/to/new_report"
}
```

用 `python3 -m arl.benchmarks programbench --config config.json` 执行，输出覆盖率、完整性、已判分任务的平均测试通过比例及 resolved 比例；基础设施错误不算模型失败。原始官方 report 和 submission 保留在 Docker 宿主机，汇总保存在本地。

## 验证

```bash
python3 -m unittest discover -s tests -p test_programbench.py -v
# CPU + 远程 Docker，合成失败提交，不启动模型
python3 script/smoke_programbench_env.py
# 能编译的占位程序，再跑真实行为测试；不代表模型成绩
python3 script/smoke_programbench_env.py --mode stub
```
