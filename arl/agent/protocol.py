"""agentic RL 的环境协议 (鸭子类型, 训练器不 import 任何环境包的基类).

一条轨迹 = 一个 ``AgentEnv`` 实例上的多轮对话。环境只说**消息**, token 级的事 (渲染、
预算、logprob、mask、advantage) 全在 ARL 这边:

    class AgentEnv:
        tools: list[dict]                          # chat template 的 tools= (OpenAI/hermes 形状)
        done: bool
        async def reset(self) -> None
        def messages(self, aggressiveness: int = 0) -> list[dict]
            # 本轮完整 prompt (system + 任务 + 历史)。aggressiveness 越大历史压得越狠;
            # 训练器从 0 往上试, 直到渲染结果塞进 prompt 预算。
        async def step(self, text: str, *, truncated: bool = False) -> StepOut
            # text = 本轮生成的解码文本 (不含开头的 <think>, 见 Qwen3.x 生成 prompt);
            # truncated = 本轮撞了生成上限 (stop_reason == "length")。
        async def finish(self, reason: str = "") -> StepOut
            # 训练器要求提前收尾 (上下文塞不下 / 采样失败), 奖励照常结算。
        def summary(self) -> dict                  # 留档用, 可选

    StepOut: .reward (只在 done 时非 0 也行) / .done / .metrics: dict / .infra_error: bool
             (done 时 True = 整条轨迹没被裁决, 丢样本) / .tool_results (留档, 可选)

    class TaskSource:                              # 题目 + 环境工厂
        def train_tasks(self) -> list
        def test_tasks(self) -> list
        def task_key(self, task) -> str            # 日志 / 留档主键
        def make_env(self, task, uid: str) -> AgentEnv
        async def close(self) -> None              # 可选

奖励按**轨迹**算 (一个终局标量), 同一条轨迹的每一轮共享它 —— adv_unit=trajectory、
组键 = 题目, 和 ARM 4c (adv_unit=trajectory, Dr-GRPO) 同一口径。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentEnv(Protocol):
    tools: list[dict[str, Any]]

    @property
    def done(self) -> bool: ...

    async def reset(self) -> None: ...

    def messages(self, aggressiveness: int = 0) -> list[dict[str, Any]]: ...

    async def step(self, text: str, *, truncated: bool = False) -> Any: ...

    async def finish(self, reason: str = "") -> Any: ...


class TaskSource(Protocol):
    def train_tasks(self) -> list[Any]: ...

    def test_tasks(self) -> list[Any]: ...

    def task_key(self, task: Any) -> str: ...

    def make_env(self, task: Any, uid: str) -> AgentEnv: ...
