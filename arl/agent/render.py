"""消息 -> token: chat template (带 tools=) + prompt 预算.

两条路:

  * **追加** (``continue_prompt``, harness 的常态): 环境在压缩档 0 下历史逐字不变
    (CudaEnv: 思考进 reasoning_content, 模板原样回放), 于是新 prompt 的文本 =
    上一轮 prompt 文本 + 本轮回复原文 + 新增部分 (工具结果 + 下一轮的生成提示)。只把
    新增部分 tokenize 接在**采样出来的那串 token** 后面 —— 模型看到的历史就是它自己生成的
    token, 不经过 decode/encode 往返; 整条轨迹是一条前缀递增的序列, 训练时合成一个 datum。
  * **重渲染** (``fit``): 文本前缀对不上 (回复里有模板会规整掉的空白、环境压缩了历史),
    或者追加后超预算, 就整段重渲染, 从压缩档 0 往上试到塞进预算 —— 这一轮起开一个新 datum。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_EFFORT_UNSET = object()


class AgentRenderer:
    """``apply_chat_template(messages, tools=..., add_generation_prompt=True)``。

    ``reasoning_effort``: Qwen3.8 模板的一维 (xhigh/medium/low, 缺省 xhigh = 往 system
    顶上塞一句"想仔细点")。None = 不传, 用模板缺省 —— CudaAgent RFT 数据就是这么渲染的;
    ARM 4c 用 "medium" (中性档, 一个字都不加)。换了就是换 prompt 分布。
    """

    def __init__(self, tokenizer, *, enable_thinking: bool | None = True,
                 reasoning_effort: str | None = None):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort or None
        self._extra_ok = True

    def render_text(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
        return self._apply(messages, tools, tokenize=False)

    def encode(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
        return _ids(self._apply(messages, tools, tokenize=True))

    def _apply(self, messages, tools, tokenize: bool):
        kwargs: dict[str, Any] = dict(add_generation_prompt=True, tokenize=tokenize)
        if tools:
            kwargs["tools"] = tools
        extra: dict[str, Any] = {}
        if self.enable_thinking is not None:
            extra["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        if extra and self._extra_ok:
            try:
                return self.tokenizer.apply_chat_template(list(messages), **kwargs, **extra)
            except TypeError:
                self._extra_ok = False
                logger.warning(f"chat template rejected {sorted(extra)}; rendering without them")
        return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def fit(self, env, budget: int, max_aggressiveness: int = 4
            ) -> tuple[list[int] | None, str | None, int]:
        """整段重渲染: 从压缩档 0 往上试 -> (prompt ids, 文本, 档位); 最狠一档也超预算 -> ids=None。"""
        for a in range(max_aggressiveness + 1):
            text = self.render_text(env.messages(a), env.tools)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= budget:
                return ids, text, a
        return None, None, max_aggressiveness

    def continue_prompt(self, env, prev_ids: list[int], prev_text: str, resp_ids: list[int],
                        budget: int) -> tuple[list[int], str] | None:
        """追加: (prev_ids + resp_ids + tokenize(新增部分), 新文本); 前缀对不上或超预算 -> None。"""
        full = self.render_text(env.messages(0), env.tools)
        expected = prev_text + self.tokenizer.decode(resp_ids, skip_special_tokens=False)
        if not full.startswith(expected):
            return None
        delta = self.tokenizer.encode(full[len(expected):], add_special_tokens=False)
        ids = list(prev_ids) + list(resp_ids) + delta
        if len(ids) > budget:
            return None
        return ids, full

    def stop_token_ids(self) -> list[int]:
        ids: list[int] = []
        tid = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
            ids.append(tid)
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        return list(dict.fromkeys(ids))

    def decode(self, tokens: list[int]) -> str:
        # <tool_call>/<think> 在 Qwen3.x 词表里不是 special token, skip_special_tokens 留着它们
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def _ids(out) -> list[int]:
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)
