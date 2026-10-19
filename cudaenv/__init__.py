"""CudaEnv —— CUDA-Agent 式多文件工作区环境 (agentic RL 用), 与训练框架解耦.

一条轨迹 = 一个 agent 在一道 CudaTasks 题 (``registry/<instance_id>/``) 的工作区里多轮调用工具:

    write_file / edit_file / read_file    操作**内存里的**工作区 (kernels/* + model_new.py)
    compile / verify / profile [/ analyze] 整包 files POST 给 CudaEval ``/repo/evaluate``

判定口径 (受保护文件、静态检查、dispatch 审计、基线、里程碑) 全在服务端; 本包只做
"消息 <-> 工具 <-> 评测"的编排, 不依赖 torch / tokenizer / 任何训练框架。

训练框架看到的接口是**消息层**的 (见 ``env.CudaWorkspaceEnv``):

    env = CudaWorkspaceEnv(task, client=CudaEvalClient(endpoint))
    await env.reset()
    while not env.done:
        msgs = env.messages(aggressiveness=0)        # 本轮 prompt (历史已压缩), 交给 chat template
        out = await env.step(response_text, truncated=False)
    out.reward / env.summary()

token 级的东西 (渲染、预算、logprob、mask) 是训练器的事: 预算塞不下就调大
``aggressiveness`` 重取 ``messages``。
"""

from .client import CudaEvalClient
from .env import CudaWorkspaceEnv, EnvConfig, StepOutput
from .tasks import CudaTask, load_task_set, load_instances
from .tools import TOOLS, ANALYZE_TOOL, parse_tool_calls, tool_schemas

__all__ = [
    "CudaEvalClient",
    "CudaWorkspaceEnv",
    "EnvConfig",
    "StepOutput",
    "CudaTask",
    "load_task_set",
    "load_instances",
    "TOOLS",
    "ANALYZE_TOOL",
    "parse_tool_calls",
    "tool_schemas",
]

__version__ = "0.1.0"
