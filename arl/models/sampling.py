"""vLLM-backed sampling, replacing ``tinker.SamplingClient``.

``SamplingEngine`` wraps a single vLLM ``AsyncLLMEngine`` (base model loaded once).
``SamplingClient`` samples from it and implements
``sample_async(prompt, num_samples, sampling_params)`` returning sequences with
per-token logprobs and a stop reason -- exactly what ``LocalTokenCompleter``
consumes.

Concurrent ``generate`` coroutines are batched by the engine, so many rollouts
sample in parallel with no external lock. For on-policy full-parameter training,
the policy weights are hot-reloaded **in place** each step via ``reload_weights``
(vLLM's official RLHF weight-update path) -- the engine, its KV cache and CUDA
graphs stay alive; only the model tensors are overwritten. The one thing that does
NOT survive a weight update is the prefix cache: it must be dropped explicitly
afterwards, see ``SamplingEngine.reset_prefix_cache``.
"""

from __future__ import annotations

import inspect
import logging
import os

import torch

from arl.models import types as tinker

logger = logging.getLogger(__name__)

_FINISH_TO_STOP_REASON = {"stop": "stop", "length": "length", "abort": "stop"}


def _init_weight_update_group_worker(worker, host: str, port: int, world_size: int) -> bool:
    """Run INSIDE the vLLM EngineCore worker (via collective_rpc).

    Joins a ``StatelessProcessGroup`` of ``world_size = 1 + tp_size`` members:
    the trainer process is rank 0, TP worker ``i`` is rank ``1 + i`` (each
    worker computes its own group rank from its TP rank -- collective_rpc runs
    this on EVERY worker). Used for the DISAGGREGATED weight sync: the trainer
    and workers live on DIFFERENT GPUs, so CUDA IPC can't alias memory across
    them -- instead the trainer NCCL-broadcasts each full param into this group;
    every worker receives it and ``load_weights`` TP-slices it locally.
    The communicator is stashed on the worker so per-step updates reuse it.
    """
    import torch
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    rank = worker.rank + 1  # trainer holds group rank 0
    pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
    worker._kr_weight_comm = PyNcclCommunicator(pg, device=torch.cuda.current_device())
    return True


def _nccl_update_weights_worker(worker, metadata: list) -> int:
    """Run INSIDE the vLLM EngineCore worker (via collective_rpc).

    ``metadata`` is a list of ``(name, dtype_str, shape)`` describing the params
    the trainer is about to broadcast, in order. For each, allocate a receive
    buffer on the worker's GPU, receive it over the weight-update NCCL group
    (trainer is src=0), then copy into the live model via ``load_weights`` (which
    applies the HF->vLLM name/qkv-fusion mapping). Streaming in trainer-sized
    groups keeps peak GPU bounded.
    """
    import torch

    comm = worker._kr_weight_comm
    model = worker.model_runner.model
    # 多模态包装模型 (语言塔挂在 self.language_model 下, 如 Qwen3.6 的
    # *ForConditionalGeneration): 训练侧发来的是 HF 文本类的名字 (model.layers... /
    # lm_head.*), 而包装类的 hf_to_vllm_mapper 只认 **checkpoint 格式**
    # (model.language_model.* / lm_head.* / model.visual.*) —— model.layers... 谁也
    # 不认, AutoWeightsLoader 直接报 unexpected。这里把文本名折回 checkpoint 格式
    # (lm_head.* 两边同名, 不动); 纯文本模型 (QwQ 等, 无 language_model 属性) 不改名。
    remap_mm = hasattr(model, "language_model")
    stream = torch.cuda.current_stream()
    weights = []
    for name, dtype_str, shape in metadata:
        buf = torch.empty(tuple(shape), dtype=getattr(torch, dtype_str), device=torch.cuda.current_device())
        comm.broadcast(buf, src=0, stream=stream)
        if remap_mm and name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        weights.append((name, buf))
    stream.synchronize()
    model.load_weights(weights)
    torch.cuda.synchronize()
    return len(weights)


def _read_model_config(model_name: str) -> dict:
    """读 HF 目录里的 config.json (读不到就返回 {}, 让 vLLM 自己去处理)。"""
    import json

    path = os.path.join(model_name.split(":")[0], "config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:  # noqa: BLE001 -- 读配置失败不该拦住起引擎
        logger.warning(f"[sampler] cannot read {path} ({e}), skipping the model-specific engine argument tuning")
        return {}


def _extra_engine_args(model_name: str) -> dict:
    """按模型 config.json 决定两个额外的 vLLM 引擎参数。两个都能用环境变量强制覆盖。

    ── ① 多模态模型: 屏蔽视觉/音频塔 (``language_model_only=True``) ──────────────
    Qwen3.6-27B 这类 ``*ForConditionalGeneration`` 的权重里带一个 vision tower, 而
    KernelBench 的输入全是纯文本, 一张图都不会喂进去。不关它的代价:
      * vision tower 的权重照样加载, 白占显存;
      * vLLM 起引擎时按**最大尺寸的假图**做多模态 profiling —— 这个模型的
        preprocessor_config 写着 longest_edge=16777216 (16M 像素), profiling 出来的
        activation 峰值会把 KV cache 挤掉一大块。
    vLLM 0.19 的 ``language_model_only`` = 把所有模态的 limit 设成 0, 视觉塔的子模块
    直接被换成 StageMissingLayer (根本不 init、不加载权重), 日志里会打
    "running in text-only mode"。老版本 vLLM 没这个字段, 退回等价的
    ``limit_mm_per_prompt={"image": 0, "video": 0}``。
    这一项只在**纯文本**任务下才对: 哪天真要喂图, 设 KERNEL_RL_LANGUAGE_MODEL_ONLY=0。

    ── ② 混合线性注意力模型: CUDA graph 退到 PIECEWISE ─────────────────────────
    Qwen3.6 是 linear_attention (GDN) + full_attention 4:1 的混合结构。vLLM 默认
    ``cudagraph_mode=FULL_AND_PIECEWISE``, 而这台机器 (PPU-ZW810E) 上 **FULL 图会静默
    产出垃圾**: 2026-08-13 实测 TP=8, 同一条 prompt 采 256 token, 输出全是 token 0
    ("!!!!!..."), 不报任何错。同一份权重换成 PIECEWISE 或 eager 就正常, 且三者里
    PIECEWISE 最快 (256 token: PIECEWISE 11.5s / eager 24.3s, 2.1x):
        FULL_AND_PIECEWISE (默认)  -> 全 "!"      ← 假成绩, 而且不会报错
        PIECEWISE                  -> 正常, 11.5s
        eager                      -> 正常, 24.3s
    所以对带 linear_attention 的模型自动降到 PIECEWISE。KERNEL_RL_CUDAGRAPH_MODE 可
    强制指定 (FULL_AND_PIECEWISE / PIECEWISE / NONE), 换 vLLM/驱动后想复测就用它。
    非混合模型 (QwQ-32B 等) 一律不动, 保持原来的 FULL_AND_PIECEWISE。
    """
    import dataclasses

    from vllm import AsyncEngineArgs

    cfg = _read_model_config(model_name)
    text_cfg = cfg.get("text_config") or {}
    extra: dict = {}

    # ── ① 多模态 -> 纯文本 ──────────────────────────────────────────────────
    is_mm = any(k in cfg for k in ("vision_config", "audio_config", "image_token_id"))
    raw = os.environ.get("KERNEL_RL_LANGUAGE_MODEL_ONLY", "").strip().lower()
    if raw in ("1", "true", "yes"):
        lang_only = True
    elif raw in ("0", "false", "no"):
        lang_only = False
    else:
        if raw:
            logger.warning(
                f"[sampler] KERNEL_RL_LANGUAGE_MODEL_ONLY={raw!r} could not be parsed (expected 1/0), falling back to auto-detection"
            )
        lang_only = is_mm
    if lang_only:
        fields = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
        if "language_model_only" in fields:
            extra["language_model_only"] = True
        else:  # 老 vLLM: 等价写法
            extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        logger.info(f"[sampler] {model_name} is multimodal; loading the language tower only (text-only mode)")

    # ── ② 混合线性注意力 -> PIECEWISE ───────────────────────────────────────
    layer_types = cfg.get("layer_types") or text_cfg.get("layer_types") or []
    is_hybrid = any("linear_attention" in str(t) or "mamba" in str(t) for t in layer_types)
    mode = os.environ.get("KERNEL_RL_CUDAGRAPH_MODE", "").strip().upper()
    if not mode and is_hybrid:
        mode = "PIECEWISE"
        logger.info(
            f"[sampler] {model_name} is a hybrid linear-attention model; lowering cudagraph_mode to PIECEWISE "
            f"(a FULL graph silently emits all-zero tokens on this machine; see the notes in _extra_engine_args)"
        )
    if mode:
        extra["compilation_config"] = {"cudagraph_mode": mode}
    return extra


class SamplingEngine:
    """Owns the vLLM AsyncLLMEngine. Created once; weights reloaded in place per sync."""

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        gpu_memory_utilization: float = 0.3,
        max_model_len: int | None = None,
        # False = 开 CUDA graph (默认, 与 training_client._sampler_enforce_eager 一致;
        # 热更新的原地性已实测验证, 见那边的 docstring)。
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # Allow pickle fallback in collective_rpc so we can ship the NCCL
        # group-init callable to the worker. Trusted local RLHF payload -- no
        # untrusted input crosses this boundary.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(dtype, "auto")
        logger.info(f"[sampler] loading vLLM AsyncLLMEngine for {model_name}")
        # 采样端回传的 logprob 口径。默认 raw_logprobs = 全词表、不除温度 (老行为);
        # processed_logprobs = 除温度、top_k/top_p 截断并重新归一之后的 log p —— 采样分布
        # 回放 (ppo_loop 的 sampling_replay) 要的 old logprobs, 由 ppo_loop 经这个环境变量
        # 打开。只改回传的数, 不改怎么采样。
        lp_mode = os.environ.get("KERNEL_RL_SAMPLER_LOGPROBS_MODE", "").strip()
        extra_lp = {"logprobs_mode": lp_mode} if lp_mode else {}
        if lp_mode:
            logger.info(f"[sampler] logprobs_mode={lp_mode}")
        args = AsyncEngineArgs(
            model=model_name,
            dtype=dtype_str,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            disable_log_stats=True,
            # 按模型 config.json 追加的参数 (多模态 -> 只留语言塔; 混合线性注意力 ->
            # cudagraph_mode=PIECEWISE)。非这两类模型时是空 dict, 行为不变。
            **_extra_engine_args(model_name),
            **extra_lp,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self._req_counter = 0
        # 在飞请求数 (generate 进/出各记一次), MultiSamplingEngine 按它挑最闲的引擎。
        self.n_inflight = 0

    @property
    def engines(self) -> list["SamplingEngine"]:
        """按副本枚举 (单引擎 = 它自己)。权重同步按副本各建一组 NCCL, 见
        ``training_client._ensure_weight_comm``。"""
        return [self]

    def _next_request_id(self) -> str:
        # asyncio is single-threaded, so a plain counter is race-free and gives a
        # unique, deterministic request id per sample.
        self._req_counter += 1
        return f"kr-{self._req_counter}"

    async def reload_weights(self, weights_path: str) -> None:
        """Overwrite the engine's model weights in place from an HF checkpoint dir.

        vLLM's worker loads the checkpoint at ``weights_path`` through its normal
        model loader (handling the HF->vLLM name/qkv-fusion mapping) and copies the
        tensors into the running model. The KV cache and CUDA graphs are preserved.

        Note: vLLM's layerwise reload prints benign per-layer "Failed to load
        weights" warnings (its ``load_numel`` counter under-reports). The weights
        ARE applied -- verified empirically by reloading a zeroed checkpoint and
        observing the sampler output change accordingly.
        """
        result = self.engine.collective_rpc(
            "reload_weights",
            kwargs={"weights_path": weights_path, "is_checkpoint_format": True},
        )
        if inspect.isawaitable(result):
            await result
        logger.info(f"[sampler] reloaded policy weights from {weights_path}")

    async def reset_prefix_cache(self) -> None:
        """丢掉所有已缓存的前缀 block。**每次权重更新之后都必须调一次**。

        prefix caching 这份代码从没显式打开过, 但 vLLM V1 对 CUDA 上的 generation 模型
        默认就是开的 (``EngineArgs.enable_prefix_caching`` 默认 None -> ``CacheConfig``
        的 True), 所以它一直在生效。

        而这个缓存住在 EngineCore 进程的 scheduler 里, key 只有 token hash —— 它不知道
        底下的权重换过了。``reload_weights`` / ``update_weights_nccl`` 都是 collective_rpc
        打到 worker 的 model_runner, 只覆盖模型张量, 碰不到 scheduler。不清的话, 每一次
        命中吐出来的都是**旧策略**算的 KV。

        最脏的是 system prompt 那几十个 block: 每条请求都命中, LRU 于是永远淘汰不掉它们,
        整个 run 都在复用第 0 步算出来的 KV。后果是采样分布和顺带记下来的 logprobs 都不是
        当前策略的, PPO/GRPO 的 behavior policy 被静默污染。

        vLLM 只回收没有请求占用的 block: 引擎里还有在跑的请求时这个调用是 no-op 并返回
        False (见 ``scheduler.reset_prefix_cache``)。本项目的 sync 点都卡在两个采样阶段
        之间, 正常不该有在飞的请求 —— 有就告警, 不去抢占它们 (那些生成本来就正在被这次
        权重更新腰斩, 是异步 RL 自带的问题, 不归这里管)。
        """
        ok = await self.engine.reset_prefix_cache()
        if not ok:
            logger.warning(
                "[sampler] reset_prefix_cache had no effect: requests still running in the engine are holding KV blocks."
                "After this weight update, prefix KV computed with the old weights may still be hit."
            )

    def shutdown(self) -> None:
        """Stop the vLLM engine and its background EngineCore subprocess(es).

        Needed so the process can exit: EngineCore procs otherwise linger and
        (under SPMD) the NCCL finalizer path deadlocks. Idempotent / best-effort.
        """
        try:
            self.engine.shutdown()
        except Exception as e:  # noqa: BLE001 -- teardown must never raise
            logger.warning(f"[sampler] engine shutdown failed (ignored): {e}")

    async def init_weight_update_group(self, host: str, port: int, world_size: int) -> None:
        """Have every TP worker join the trainer's weight-update NCCL group.

        Dispatched by the trainer, which itself joins the same group as rank 0
        (worker i joins as rank 1+i, computed worker-side). The workers'
        ``collective_rpc`` call blocks in the rendezvous until the trainer
        creates its own side, so the trainer must run this concurrently with its
        own ``StatelessProcessGroup.create`` (see ``training_client._ensure_weight_comm``).
        """
        result = self.engine.collective_rpc(
            _init_weight_update_group_worker,
            kwargs={"host": host, "port": port, "world_size": world_size},
        )
        if inspect.isawaitable(result):
            await result

    async def update_weights_nccl(self, metadata: list) -> None:
        """Receive a group of broadcast params into the live model (disaggregated).

        The worker blocks on the NCCL recvs until the trainer posts the matching
        broadcasts, so the trainer must NOT await this before it broadcasts -- it
        dispatches this as a task, broadcasts, then awaits (see
        ``training_client._sync_to_sampler_nccl``).
        """
        result = self.engine.collective_rpc(
            _nccl_update_weights_worker, kwargs={"metadata": metadata}
        )
        if inspect.isawaitable(result):
            await result

    async def generate(self, prompt_ids, vllm_params):
        """Stream one request to completion and return the final RequestOutput.

        Concurrent calls are safe and are batched together by the engine.
        """
        from vllm import TokensPrompt

        request_id = self._next_request_id()
        final = None
        self.n_inflight += 1
        try:
            async for out in self.engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids),
                vllm_params,
                request_id,
            ):
                final = out
        finally:
            self.n_inflight -= 1
        return final


class MultiSamplingEngine:
    """DP 份 :class:`SamplingEngine` 副本 (各自 TP, 卡不重叠), 对外同一个接口。

    为什么 (2026-09-06): 采样卡 8 张全给一台 TP=8 引擎时, 解码每层的 all-reduce 要
    8 张卡同步, 而 35B-A3B 激活 3B 的 MoE 每张卡算得很少 —— 通信/launch 开销占大头。
    拆成 2 x TP=4 (每卡权重 17.5GB, KV 还有 60GB+), 两台引擎各吃一半在飞请求, 同步
    伙伴减半、聚合 batch 容量翻倍。qwen36_grpo_0905_v1 实测采样窗口占批墙钟 55%,
    是最大的一块。

    调度: ``generate`` 每次挑在飞请求最少的副本 (SamplingEngine.n_inflight), 一条
    请求从头到尾在同一台引擎上; 副本间没有任何状态共享 —— prefix cache 各自一份
    (同一道题的 8 条轨迹可能分在两边, system prompt 的 KV 各算一次, 代价可忽略)。

    权重同步: 副本各自一组 {trainer, 该副本的 TP worker} 的 NCCL 通信器 (端口顺延),
    trainer 对每组各广播一遍 (见 ``training_client._sync_to_sampler_nccl``); 训练侧
    的 ZeRO-3 gather 只做一次。磁盘路径 (``reload_weights``) 各副本各 reload 一次。

    不用 vLLM 自带的 ``data_parallel_size``: 那条路在 PPU + 混合线性注意力 + MoE 上
    没验证过, 而且它的 collective_rpc / DP 协调器改变了权重同步的拓扑; 这里每台副本
    都是已经跑了几十个 run 的那条单引擎路径, 只是起了两份。
    """

    def __init__(self, engines: list[SamplingEngine]):
        if not engines:
            raise ValueError("MultiSamplingEngine needs at least one engine")
        self._engines = list(engines)

    @property
    def engines(self) -> list[SamplingEngine]:
        return list(self._engines)

    @property
    def n_inflight(self) -> int:
        return sum(e.n_inflight for e in self._engines)

    def _pick(self) -> SamplingEngine:
        # 最闲的副本; 并列取序号小的 (稳定, 便于对日志)。
        return min(self._engines, key=lambda e: e.n_inflight)

    async def generate(self, prompt_ids, vllm_params):
        return await self._pick().generate(prompt_ids, vllm_params)

    async def reload_weights(self, weights_path: str) -> None:
        import asyncio

        await asyncio.gather(*[e.reload_weights(weights_path) for e in self._engines])

    async def reset_prefix_cache(self) -> None:
        import asyncio

        await asyncio.gather(*[e.reset_prefix_cache() for e in self._engines])

    async def init_weight_update_group(self, host: str, port: int, world_size: int) -> None:
        raise RuntimeError(
            "MultiSamplingEngine sets up one weight-sync group per replica: call "
            "init_weight_update_group(host, port + i, ...) on each .engines[i]; see training_client._ensure_weight_comm"
        )

    async def update_weights_nccl(self, metadata: list) -> None:
        raise RuntimeError(
            "MultiSamplingEngine broadcasts weights once per replica: call "
            "update_weights_nccl on each .engines[i]; see training_client._sync_to_sampler_nccl"
        )

    def shutdown(self) -> None:
        for e in self._engines:
            e.shutdown()


class SamplingClient:
    """Samples from a :class:`SamplingEngine` (weights already synced in place)."""

    def __init__(self, engine: SamplingEngine):
        self.engine = engine

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> tinker.SampleResponse:
        from vllm import SamplingParams as VLLMSamplingParams

        prompt_ids = prompt.to_ints()

        stop = sampling_params.stop
        stop_strs: list[str] | None = None
        stop_ids: list[int] | None = None
        if stop:
            if all(isinstance(s, int) for s in stop):
                stop_ids = list(stop)  # type: ignore[arg-type]
            else:
                stop_strs = [s for s in stop if isinstance(s, str)]

        def _vparams(max_tokens: int) -> "VLLMSamplingParams":
            return VLLMSamplingParams(
                n=num_samples,
                temperature=sampling_params.temperature,
                max_tokens=max_tokens,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k if sampling_params.top_k and sampling_params.top_k > 0 else -1,
                stop=stop_strs,
                stop_token_ids=stop_ids,
                logprobs=0,  # return the sampled token's own logprob
                seed=sampling_params.seed,
            )

        budget = int(sampling_params.think_budget or 0)
        if 0 < budget < sampling_params.max_tokens:
            if num_samples != 1:
                raise ValueError("think_budget needs num_samples=1 (each sample is two-stage on its own)")
            seq = await self._sample_with_think_budget(prompt_ids, sampling_params, _vparams)
            return tinker.SampleResponse(sequences=[seq])

        out = await self.engine.generate(prompt_ids, _vparams(sampling_params.max_tokens))

        sequences: list[tinker.SampledSequence] = []
        for comp in out.outputs:
            toks = list(comp.token_ids)
            if comp.logprobs is not None:
                lps = [comp.logprobs[i][tid].logprob for i, tid in enumerate(toks)]
            else:
                lps = None
            stop_reason: tinker.StopReason = _FINISH_TO_STOP_REASON.get(
                comp.finish_reason or "stop", "stop"
            )  # type: ignore[assignment]
            sequences.append(
                tinker.SampledSequence(tokens=toks, logprobs=lps, stop_reason=stop_reason)
            )
        return tinker.SampleResponse(sequences=sequences)

    async def _sample_with_think_budget(self, prompt_ids, sp, make_vparams) -> tinker.SampledSequence:
        """思考预算: 两段式生成, 不靠 vLLM 的 thinking_token_budget。

        第一段只给 ``think_budget`` 个 token。没写满就结束了 = 普通样本, 原样返回。写满了:

          * 里面已经有 ``</think>`` (思考在预算内收了, 卡在写答案) -> 原样接着采, 不插任何东西;
          * 没有 -> 接上 ``think_force_ids`` (过渡语 + ``</think>``), 再采答案。

        第二段的上限 = ``max_tokens`` 减去前面已经用掉的, 所以整条仍 <= max_tokens, 撞上它
        照样是 ``stop_reason="length"``。第二段的 prompt 是第一段的严格延长, 走 prefix cache,
        几乎不重算。

        强塞的 token 不是策略采出来的: logprob 填 0 占位, ``loss_mask`` 记 0 (训练侧按它把
        advantage 置 0, 这几个位置不进 loss / KL / ratio 统计)。前后两段都是策略自己采的,
        照常进 loss。不用 vLLM 自带的 thinking_token_budget, 是因为它要引擎带 reasoning_config,
        且强制的 token 会以 logprob≈0 混进 processed_logprobs, 回放时分不出来。
        """
        budget = int(sp.think_budget)
        force = list(sp.think_force_ids or [])
        end = list(sp.think_end_ids or [])
        if not force or not end:
            raise ValueError("think_budget needs think_force_ids and think_end_ids")

        def _unpack(out):
            comp = out.outputs[0]
            toks = list(comp.token_ids)
            lps = ([comp.logprobs[i][tid].logprob for i, tid in enumerate(toks)]
                   if comp.logprobs is not None else None)
            return toks, lps, comp.finish_reason or "stop"

        toks1, lps1, fin1 = _unpack(await self.engine.generate(prompt_ids, make_vparams(budget)))
        if fin1 != "length":
            return tinker.SampledSequence(
                tokens=toks1, logprobs=lps1,
                stop_reason=_FINISH_TO_STOP_REASON.get(fin1, "stop"))  # type: ignore[arg-type]

        has_end = any(toks1[i:i + len(end)] == end for i in range(len(toks1) - len(end) + 1))
        inject = [] if has_end else force
        rest = sp.max_tokens - len(toks1) - len(inject)
        if rest <= 0:
            return tinker.SampledSequence(tokens=toks1, logprobs=lps1, stop_reason="length")
        toks2, lps2, fin2 = _unpack(await self.engine.generate(
            list(prompt_ids) + toks1 + inject, make_vparams(rest)))
        lps = None if lps1 is None or lps2 is None else lps1 + [0.0] * len(inject) + lps2
        return tinker.SampledSequence(
            tokens=toks1 + inject + toks2, logprobs=lps,
            stop_reason=_FINISH_TO_STOP_REASON.get(fin2, "stop"),  # type: ignore[arg-type]
            loss_mask=[1.0] * len(toks1) + [0.0] * len(inject) + [1.0] * len(toks2),
            think_forced=bool(inject),
        )
