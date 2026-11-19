# SWE 服务客户端检查

查看本机实验阶段和轨迹进度（默认选最新 run）：

```bash
python3 script/run_status.py
python3 script/run_status.py runs/qwen36_1116_v1 --watch 5 --gpu
```

新启动的 rollout 每约 2 秒写入各 rank 的 `rollout_progress.rankN.json`，
显示本批轨迹完成数、生成/排队数、环境执行/判分数及已完成轮数。
旧进程只能显示整批结束后落盘的累计结果；读取共享目录时，本机 PID 仅代表执行此命令的机器。

ARL 仅保留训练侧 HTTP 客户端、环境适配和配置。服务源码、服务端测试、节点拓扑与部署工具统一位于 **10.0.134.70 宿主机 `/root/workspace/SweContainerService`**，与本训练仓库分开维护。

训练容器中的检查命令（在 ARL 根目录执行）：

```bash
python3 script/swe_client/ctl.py health
python3 script/swe_client/ctl.py status
python3 script/swe_client/smoke.py
python3 script/swe_client/cluster_smoke.py
```

endpoint 仍为 `http://10.0.134.70:18770`，token 文件仍为 `/root/ai_data/swe/service/token`。`train_swe_grpo.conf`、兼容入口 `train_swe_grpo_local.conf` 以及 `experiments/` 内所有含 SWE 的配置均连接该 master，无需服务端源码。cluster_smoke 的参考补丁仅用于可信评分自检，不用于模型训练。

部署和同步镜像需登录 .70 宿主机操作：

```bash
ssh 10.0.134.70
cd /root/workspace/SweContainerService
./swe-cluster status
./swe-cluster deploy
./swe-cluster deploy --worker node71
./swe-cluster sync-images --worker node71 --limit 20
./swe-cluster prepare --limit 20
```

新增宿主机修改该独立项目的 `configs/cluster.local.json`，再统一部署。.71 仅接收运行文件，不保存源码项目。部署会重启服务，应在训练停止后操作。

旧本地/SSH 后端现在调用宿主机已有的 worker 脚本，不再携带服务端源码。默认 `swe_worker_script=/root/workspace/SweContainerService/swe_service/worker.py` 适用于 .70；若直接调用其他 worker 的 stdio 入口，显式指定该机的运行路径（例如 .71 的 `/mnt/ai_data/swe/service/code/worker.py`）。

## 与 swe-qwen-collect 的接口对应

参考 `/root/share/swe-qwen-collect/collect.py`，共用 `/rpc` 的
`tasks → start → exec → patch/grade → close` 协议。ARL 的 `SweTaskSource`
负责加载远程任务，`DockerWorkspace` 保持同一轨迹的 workspace 名称，master
负责 worker 分配和后续路由。HTTP 429 在请求预算内重试；传输错误不重放命令。
令牌仅从文件或 `SWE_WORKER_TOKEN` 读取，不写入模型消息或轨迹。

ARL 默认使用 `claude_code` harness：真实 CLI 调用本地 Qwen 采样器，经 MCP 在远程工作区执行 bash，结束后提交工作区 diff。可选 `mini_swe_agent` 仍按其官方提示词输出 patch 后评分。
参考采集器的 JSON command/done 提示词不替换所选 harness。
collect 导出 token/logprob/mask，train 继续在线 GRPO。

从仓库根目录检查配置和真实环境（不占 GPU）：

```bash
python3 -m arl.experiments experiments/swe.yaml --phase collect --check
python3 script/smoke_swe_env.py \
  --endpoint http://10.0.134.70:18770 \
  --token-file /root/ai_data/swe/service/token \
  --dataset remote:/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl \
  --harness claude_code --negative
```

第二条创建独立工作区，通过 Claude Code 最终回复提交一个不修复问题的补丁，预期
reward=0、infra_error=false，结束后回收容器。`--gold` 可替换 `--negative`
验证参考补丁得 1；仅用于可信评分自检。
