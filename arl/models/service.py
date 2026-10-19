"""Local service client, replacing ``tinker.ServiceClient``.

Constructs :class:`~arl.models.training_client.LocalTrainingClient` instances
for new full-parameter runs or resumed-from-state runs. Extra training/sampling
knobs that the Tinker ``ServiceClient`` API doesn't carry (learning rate, GPU
memory split, context length) are passed via the constructor by the training loop.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def model_max_positions(model_name: str) -> int | None:
    """模型 config.json 里的 max_position_embeddings (读不到就 None)。

    多模态模型 (Qwen3.6-27B 的 qwen3_5) 把它放在 ``text_config`` 下, 顶层只有
    architectures/model_type —— 只看顶层会得到 None, 于是窗口不夹, 到 vLLM 那里才炸。
    """
    path = os.path.join(model_name, "config.json")
    if not os.path.isfile(path):
        return None  # HF hub id / 非标准目录: 不猜, 让 vLLM 自己报
    try:
        with open(path) as f:
            cfg = json.load(f)
        v = cfg.get("max_position_embeddings")
        if not v:
            v = (cfg.get("text_config") or {}).get("max_position_embeddings")
        return int(v) if v else None
    except Exception:  # noqa: BLE001 - 读不出来不该拦住训练/评估
        return None


def sampling_window(model_name: str, peak_tokens: int, requested: int = 0,
                    prompt_floor: int = 8192) -> int:
    """vLLM 引擎的上下文窗口 (prompt + 生成)。

    ``peak_tokens`` = 这个进程整场会用到的最大生成上限 (训练侧还要算上后期放宽的那一档,
    见 ppo_loop 的 max_tokens_late)。``requested`` 非 0 时就用它, 否则按老惯例取 2x 峰值。

    无论哪种都夹在模型的 ``max_position_embeddings`` 以内: 超了 vLLM 是**直接拒绝启动**
    (vllm/config/model.py 的 derived max_model_len 检查), 不是降级, 而且要等权重加载完
    才炸 —— QwQ-32B 是 40960, 所以 22528 x 2 = 45056 这种配置必须在这里先夹住。

    夹完之后留给 prompt 的余量不足 ``prompt_floor`` 就直接退出: multi_turn 的最后一轮
    prompt 带着上一轮的完整 kernel + 评测反馈, 挤没了只会在采样时静默截断。
    """
    win = requested or peak_tokens * 2
    hard = model_max_positions(model_name)
    if hard and win > hard:
        logger.warning(
            f"sampling window {win} exceeds {model_name}'s max_position_embeddings={hard}, "
            f"clamped to {hard} (so the prompt budget is {hard - peak_tokens})")
        win = hard
    if win - peak_tokens < prompt_floor:
        raise SystemExit(
            f"\npeak generation cap {peak_tokens}, sampling window {win} -> only "
            f"{win - peak_tokens} tokens left for the prompt (minimum {prompt_floor}).\n"
            f"  -> Lower the generation cap, or specify a larger window explicitly (if the model can take it).\n")
    return win


class LocalServiceClient:
    def __init__(self, base_url: str | None = None, **training_kwargs: Any):
        # base_url is accepted for signature-compatibility and ignored (self-hosted).
        self.base_url = base_url
        self.training_kwargs = training_kwargs

    async def create_training_client_async(self, model_name: str):
        from arl.models.training_client import LocalTrainingClient

        kwargs = dict(self.training_kwargs)
        kwargs.pop("model_name", None)
        return LocalTrainingClient(model_name=model_name, **kwargs)

    def _read_meta(self, path: str) -> dict:
        meta_path = os.path.join(path, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return {}

    async def create_training_client_from_state_async(self, path: str):
        from arl.models.training_client import LocalTrainingClient

        meta = self._read_meta(path)
        model_name = meta.get("model_name") or self.training_kwargs.get("model_name")
        if model_name is None:
            raise ValueError(
                f"Cannot resume from {path}: model_name unknown (no meta.json and none provided)"
            )
        kwargs = dict(self.training_kwargs)
        kwargs.pop("model_name", None)
        # Full policy weights live either at ``path`` itself or under ``path/policy``.
        policy_dir = os.path.join(path, "policy")
        weights_dir = policy_dir if os.path.isdir(policy_dir) else path
        # Sampling-only: there is no engine to scatter weights into, and ``weights_dir``
        # is a standard HF checkpoint (_save_full_model writes config+weights+tokenizer),
        # so point vLLM straight at it instead of loading a trainer-side copy.
        if kwargs.get("inference_only"):
            return LocalTrainingClient(model_name=weights_dir, **kwargs)
        client = LocalTrainingClient(model_name=model_name, **kwargs)
        client.load_policy_weights(weights_dir)
        # The critic is saved next to policy/ in every checkpoint (value_head.pt, weights +
        # its Adam state). Resuming must pick it up -- otherwise the run silently restarts
        # from value_init_path (or a random head). It overrides value_init_path on purpose.
        if getattr(client, "_has_value_head", False):
            vh = os.path.join(path, "value_head.pt")
            if os.path.isfile(vh):
                client._load_value_head(path)
                logger.info(f"[resume] value net restored from {vh}")
            else:
                logger.warning(f"[resume] {vh} not found -- the value net starts from "
                               f"value_init_path / a fresh init, not from this checkpoint")
        return client

    async def create_training_client_from_state_with_optimizer_async(self, path: str):
        client = await self.create_training_client_from_state_async(path)
        if os.path.isdir(os.path.join(path, "ds_state")) or os.path.isfile(
            os.path.join(path, "latest")
        ):
            client.load_state(path)
        return client
