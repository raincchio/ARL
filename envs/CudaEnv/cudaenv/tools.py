"""工具 schema (OpenAI/hermes 形状, 交给 chat template 的 ``tools=``) + 工具调用解析.

六个基础工具与 CudaAgent ``agent_workspace_env.TOOLS`` **逐字相同** (RFT 出来的 checkpoint
就是在这份 schema 渲染出的 prompt 上训的, 改一个字都是换分布)。``analyze`` (CudaEval
ws_version 2 的 ``acu`` roofline 剖析) 是可选的第七个, 缺省不开。
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace. Writable paths: "
                       "kernels/*.cu, kernels/*_binding.cpp (and .h/.cuh), model_new.py.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact, unique string in an existing file. Fails if "
                       "old_string is not found or matches more than once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        }, "required": ["path", "old_string", "new_string"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a workspace file (model.py, model_new.py, kernels/*).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "compile",
        "description": "Compile the workspace (all kernels/*.cu + bindings) into the "
                       "cuda_extension module on the GPU sandbox. Returns compiler "
                       "errors on failure.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "verify",
        "description": "Run correctness verification: ModelNew vs the reference Model "
                       "on 5 random inputs (atol=rtol=1e-2). Compiles first if needed.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "profile",
        "description": "Verify correctness, then measure performance: your extension vs "
                       "PyTorch eager vs torch.compile, plus a per-kernel time table. "
                       "Only profiled results count toward the final score.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

ANALYZE_TOOL: dict[str, Any] = {"type": "function", "function": {
    "name": "analyze",
    "description": "Verify correctness, then collect a roofline-level hardware profile of "
                   "your kernels: compute / memory throughput %, achieved vs theoretical "
                   "occupancy, registers per thread, shared memory per block, grid/block. "
                   "Use it to decide whether a kernel is compute- or memory-bound.",
    "parameters": {"type": "object", "properties": {}},
}}

FILE_TOOLS = {"write_file", "edit_file", "read_file"}
HEAVY_TOOLS = {"compile", "verify", "profile", "analyze"}


def tool_schemas(enable_analyze: bool = False) -> list[dict[str, Any]]:
    return TOOLS + ([ANALYZE_TOOL] if enable_analyze else [])


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def parse_tool_calls(text: str, names: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """抽出工具调用 -> (calls, 解析错误)。

    主格式是 Qwen3.5/3.8 chat template 的 XML 函数调用::

        <tool_call>
        <function=write_file>
        <parameter=path>
        kernels/foo.cu
        </parameter>
        </function>
        </tool_call>

    hermes JSON (``<tool_call>{"name":..., "arguments":{...}}</tool_call>``) 兜底。
    解析失败的条目作为反馈回给模型, 也是 RFT 过滤"非法工具调用"的信号。
    """
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1)
        fm = _FUNC_RE.search(raw)
        if fm:
            name = fm.group(1)
            args = {pm.group(1): pm.group(2) for pm in _PARAM_RE.finditer(fm.group(2))}
        elif raw.lstrip().startswith("{"):
            try:
                obj = json.loads(raw)
            except Exception as e:
                errors.append(f"invalid tool_call JSON ({type(e).__name__}: {e}); "
                              f"snippet: {raw[:120]!r}")
                continue
            name = obj.get("name")
            a = obj.get("arguments")
            args = a if isinstance(a, dict) else {}
        else:
            errors.append(f"unparseable tool_call (expected <function=...> block); "
                          f"snippet: {raw[:120]!r}")
            continue
        if name not in names:
            errors.append(f"unknown tool {name!r} (available: {sorted(names)})")
            continue
        calls.append({"name": name, "arguments": args})
    return calls, errors


def split_reasoning(text: str) -> tuple[str, str]:
    """回复原文 -> (reasoning_content, content), 给 chat template 的 assistant 消息用。

    Qwen3.x 的生成 prompt 以 ``<think>\\n`` 结尾, 采样出来的回复里**没有**开头的
    ``<think>``, 只有 ``</think>``。Qwen3.8 的模板**不**从 content 里拆 ``</think>``
    (Qwen3 会), 只认 ``reasoning_content`` 字段, 缺了就渲染一个空 think 块再接 content ——
    CudaAgent 把原文整段当 content 塞回去, 历史轮于是成了 "空 think 块 + 整段推理 + 游离的
    </think>" (2026-10 在 RFT run1 的 prompt 里查实)。这里按模板的约定拆开:
    没有 ``</think>`` (思考写到一半被截断) 的整段算 reasoning。
    """
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
    else:
        reasoning, content = text, ""
    reasoning = reasoning.split("<think>", 1)[1] if "<think>" in reasoning else reasoning
    return reasoning.strip(), content.strip()


def strip_think(text: str) -> str:
    """只留可见回复 (工具调用 + 正文)。"""
    return split_reasoning(text)[1]
