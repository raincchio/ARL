"""CudaWorkspaceEnv —— 消息层的多轮工具环境 (一个实例 = 一条轨迹).

协议 (训练框架按鸭子类型调用, 不需要 import 本包的任何基类)::

    tools: list[dict]                       chat template 的 tools=
    await reset()                           开局
    messages(aggressiveness) -> list[dict]  本轮 prompt (system + 题面 + 压缩后的历史)
    await step(text, truncated=False) -> StepOutput
    done: bool / summary() -> dict          结局与留档

**奖励是轨迹级里程碑** (CUDA-Agent 论文 §reward), 只在最后一步给, 之前每步 0:
    -1  从未通过正确性验证 (含静态检查 / torch 回退 / 作弊)
    +1  验证通过
    +2  且比 PyTorch eager 快 >5%
    +3  且比 torch.compile 也快 >5%
``reward_mode="best"`` (缺省, 论文口径) 取整条轨迹里最好的那次 profile;
``"final"`` 交卷时对最终工作区再判一次分 (SWE-bench 口径, 多一次评测)。
一次真结果都没拿到、全是评测服务失败的轨迹标 ``infra_error`` —— 训练侧应丢掉这条样本。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .client import CudaEvalClient
from .tasks import CudaTask
from .tools import FILE_TOOLS, HEAVY_TOOLS, parse_tool_calls, split_reasoning, tool_schemas

logger = logging.getLogger(__name__)

SPEEDUP_MARGIN = 1.05
_PATH_RE = re.compile(r"^(kernels/[A-Za-z0-9_\-.]{1,80}\.(cu|cuh|cpp|h)|model_new\.py)$")


@dataclass
class EnvConfig:
    max_turns: int = 16
    max_calls_per_turn: int = 4
    enable_analyze: bool = False          # 第七个工具 analyze (服务端 acu); 改了就是换 prompt 分布
    reward_mode: str = "best"             # "best" | "final"
    eval_timeout: float | None = None     # None = 用 task.json 的 timeout
    keep_full_turns: int = 4              # 压缩档 0 下保留原文的最近轮数
    compressed_result_chars: int = 200
    max_compile_err_chars: int = 2400
    max_verify_err_chars: int = 1200
    max_read_chars: int = 12_000
    max_tool_result_chars: int = 14_000


@dataclass
class Turn:
    response_text: str                    # 原文 (含 think), 只进留档
    calls: list[dict[str, Any]]
    results: list[tuple[str, str]]        # (tool name, result text)
    truncated: bool = False


@dataclass
class StepOutput:
    reward: float
    done: bool
    tool_results: list[tuple[str, str]]
    metrics: dict[str, Any]
    infra_error: bool = False             # 只在 done 时有意义: 整条轨迹没被裁决


@dataclass
class _State:
    files: dict[str, str] = field(default_factory=dict)
    history: list[Turn] = field(default_factory=list)
    turn_idx: int = 0
    done: bool = False
    best_milestone: float = -1.0
    best_profile: dict | None = None
    best_files: dict[str, str] | None = None   # 拿到 best_profile 那一刻的工作区快照
    verified_ever: bool = False
    judged: bool = False                  # 至少一次评测拿到了非 infra 的真结果
    infra_events: int = 0
    heavy_calls: int = 0
    eval_seconds: float = 0.0
    final_grade: dict | None = None
    last_result: dict | None = None


class CudaWorkspaceEnv:
    def __init__(self, task: CudaTask, client: CudaEvalClient, config: EnvConfig | None = None):
        self.task = task
        self.client = client
        self.cfg = config or EnvConfig()
        self.tools = tool_schemas(self.cfg.enable_analyze)
        self._tool_names = {t["function"]["name"] for t in self.tools}
        self._system = prompts.system_prompt(self.cfg.enable_analyze)
        self.st = _State()

    # ── 协议 ──────────────────────────────────────────────────────────────
    @property
    def done(self) -> bool:
        return self.st.done

    async def reset(self) -> None:
        self.st = _State()

    def messages(self, aggressiveness: int = 0) -> list[dict[str, str]]:
        """本轮 prompt (harness 式: 历史原样回放, 思考也保留)。

        assistant 轮按 Qwen3.x 模板的约定交 ``reasoning_content`` + ``content`` (见
        ``tools.split_reasoning``), 模板缺省 ``preserve_thinking`` 就是保留 —— 交错思考
        (interleaved thinking) 是这代模板给 agent 用的原生形状。档 0 下历史逐字不变, 所以
        下一轮 prompt = 上一轮 prompt + 本轮回复 + 工具结果, 训练器可以只追加 token
        (一条轨迹一个 datum)。

        只有塞不进预算时才降级, 越往后越伤 (最近 ``keep_full_turns`` 轮始终原样):
          1  更老的轮次去掉思考 (模板渲染成空 think 块)
          2  更老的轮次再把回复和工具结果压成一行摘要
          3+ 保留原样的轮次逐档减半, 更老的整段丢掉 (状态行里仍有全部文件)
        """
        st = self.st
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": prompts.task_message(self.task.ref_code,
                                                             self.cfg.max_turns)},
        ]
        n = len(st.history)
        keep_full = self.cfg.keep_full_turns
        if aggressiveness >= 3:
            keep_full = max(1, keep_full >> (aggressiveness - 2))
        drop_before = n - keep_full if (aggressiveness >= 3 and n > keep_full) else 0
        if drop_before:
            msgs.append({"role": "user",
                         "content": f"[{drop_before} earlier turns omitted to save context; "
                                    f"the workspace below reflects all past edits]"})
        for k, turn in enumerate(st.history):
            if k < drop_before:
                continue
            old = aggressiveness >= 1 and k < n - keep_full
            if old and aggressiveness >= 2:
                names = ", ".join(c["name"] for c in turn.calls) or "(no tool call)"
                msgs.append({"role": "assistant",
                             "content": f"(turn {k}: called {names}; details compressed)"})
                brief = " | ".join(f"{name}: {result[:self.cfg.compressed_result_chars]}"
                                   for name, result in turn.results)
                msgs.append({"role": "tool", "content": brief or "(no result)"})
                continue
            reasoning, content = split_reasoning(turn.response_text)
            msg = {"role": "assistant", "content": content}
            if not old:
                msg["reasoning_content"] = reasoning
            msgs.append(msg)
            for _name, result in turn.results:
                msgs.append({"role": "tool", "content": result})
        return msgs

    async def step(self, text: str, *, truncated: bool = False) -> StepOutput:
        """执行一轮。``truncated`` = 这轮生成撞了 token 上限 (调用方按 stop_reason 判)。"""
        if self.st.done:
            raise RuntimeError("step() after episode end")
        t0 = time.perf_counter()
        st = self.st
        calls, parse_errors = parse_tool_calls(text, self._tool_names)
        results: list[tuple[str, str]] = [("error", f"Tool call error: {e}") for e in parse_errors]
        if len(calls) > self.cfg.max_calls_per_turn:
            results.append(("error", f"Only the first {self.cfg.max_calls_per_turn} tool calls "
                                     f"of a turn are executed ({len(calls)} given)."))
            calls = calls[:self.cfg.max_calls_per_turn]

        eval_s = 0.0
        for call in calls:
            _t = time.perf_counter()
            out = await self._execute(call["name"], call["arguments"])
            if call["name"] in HEAVY_TOOLS:
                eval_s += time.perf_counter() - _t
            results.append((call["name"], out[:self.cfg.max_tool_result_chars]))
        st.eval_seconds += eval_s

        if results:  # 状态行贴在最后一个工具结果后面
            name, last = results[-1]
            results[-1] = (name, f"{last}\n\n{self._status_line()}")

        # 截断且一个工具调用都没有 = 思考写超了预算, 不是交卷: 烧掉这一轮并提醒它说短点
        if truncated and not calls and not parse_errors:
            results.append(("error",
                            "Your response hit the generation token limit BEFORE any "
                            "complete tool call was emitted; this turn is wasted. Keep "
                            "your reasoning brief (a few hundred words) and emit the "
                            "tool call early. Long derivations belong in comments in "
                            "the code you write, not in the reasoning."))

        st.history.append(Turn(text, calls, results, truncated))
        st.turn_idx += 1
        # 完整生成且没有任何 (合法或非法) 工具调用 = 交卷; 或轮数用完
        if (not truncated and not calls and not parse_errors) or st.turn_idx >= self.cfg.max_turns:
            st.done = True

        reward, infra = 0.0, False
        if st.done:
            reward, infra = await self._final_reward()

        metrics = {
            "turn": st.turn_idx - 1,
            "format_ok": float(not parse_errors),
            "n_tool_calls": float(len(calls)),
            "n_invalid_calls": float(len(parse_errors)),
            "truncated": float(truncated),
            "milestone": st.best_milestone,
            "correct": float(st.verified_ever),
            "time/eval": eval_s,
            "time/step": time.perf_counter() - t0,
        }
        if st.done:
            metrics.update(self._episode_metrics(reward, infra))
        return StepOutput(reward=reward, done=st.done, tool_results=results,
                          metrics=metrics, infra_error=infra)

    async def finish(self, reason: str = "") -> StepOutput:
        """外部提前收尾 (训练器: prompt 压到最狠也塞不进预算 / 生成失败)。奖励照常结算。"""
        if self.st.done:
            raise RuntimeError("finish() after episode end")
        self.st.done = True
        reward, infra = await self._final_reward()
        metrics = {"milestone": self.st.best_milestone, "correct": float(self.st.verified_ever),
                   "finished_early": 1.0, **self._episode_metrics(reward, infra)}
        if reason:
            logger.info(f"[cudaenv] {self.task.instance_id}: finished early ({reason})")
        return StepOutput(reward=reward, done=True, tool_results=[], metrics=metrics,
                          infra_error=infra)

    def summary(self) -> dict[str, Any]:
        st = self.st
        p = st.best_profile or {}
        return {
            "instance_id": self.task.instance_id,
            "level": self.task.level,
            "turns": st.turn_idx,
            "milestone": st.best_milestone,
            "verified": st.verified_ever,
            "speedup_vs_eager": p.get("speedup_vs_eager"),
            "speedup_vs_compile": p.get("speedup_vs_compile"),
            "best_files": st.best_files,
            "final_files": dict(st.files),
            "final_grade": st.final_grade,
            "heavy_calls": st.heavy_calls,
            "infra_events": st.infra_events,
            "eval_seconds": st.eval_seconds,
        }

    # ── 工具 ──────────────────────────────────────────────────────────────
    async def _execute(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name in FILE_TOOLS:
                return getattr(self, f"_{name}")(args)
            return await self._heavy(name)
        except Exception as e:  # noqa: BLE001 -- 工具失败变成反馈, 不炸 rollout
            logger.exception(f"tool {name} failed")
            return f"Tool {name} failed internally: {type(e).__name__}: {e}"

    def _write_file(self, args: dict) -> str:
        path, content = args.get("path"), args.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return "write_file needs string 'path' and 'content'."
        path = path.strip()
        if not _PATH_RE.match(path):
            return (f"Illegal path {path!r}. Writable: kernels/*.cu|cuh|cpp|h, "
                    f"model_new.py. Infrastructure files are fixed.")
        existed = path in self.st.files
        self.st.files[path] = content
        return f"{'Overwrote' if existed else 'Wrote'} {path} ({len(content.splitlines())} lines)."

    def _edit_file(self, args: dict) -> str:
        path, old, new = args.get("path"), args.get("old_string"), args.get("new_string")
        if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
            return "edit_file needs string 'path', 'old_string', 'new_string'."
        path = path.strip()
        content = self.st.files.get(path)
        if content is None:
            return f"{path} does not exist (files: {sorted(self.st.files) or 'none'})."
        n = content.count(old)
        if n == 0:
            return f"old_string not found in {path}; read the file and retry with an exact match."
        if n > 1:
            return f"old_string matches {n} times in {path}; provide a unique match."
        self.st.files[path] = content.replace(old, new, 1)
        return f"Edited {path}."

    def _read_file(self, args: dict) -> str:
        path = args.get("path")
        path = path.strip() if isinstance(path, str) else None
        content = self.task.ref_code if path == "model.py" else self.st.files.get(path or "")
        if content is None:
            return (f"{path!r} not found. Available: model.py, "
                    f"{sorted(self.st.files) or '(workspace empty)'}")
        if len(content) > self.cfg.max_read_chars:
            content = content[:self.cfg.max_read_chars] + "\n... (truncated)"
        return f"```\n{content}\n```"

    async def _evaluate(self, tool: str, files: dict[str, str]) -> dict[str, Any]:
        self.st.heavy_calls += 1
        return await self.client.evaluate(
            self.task.instance_id, files, tool, registry=self.task.registry,
            timeout=self.cfg.eval_timeout or self.task.timeout)

    async def _heavy(self, name: str) -> str:
        st = self.st
        if "model_new.py" not in st.files:
            return "The workspace has no model_new.py yet -- write your implementation first."
        res = await self._evaluate(name, dict(st.files))
        if res.get("infra_error"):
            st.infra_events += 1
            return ("The evaluation service failed (this is NOT a problem with your "
                    "code). Retry the same tool call.")
        st.judged = True
        st.last_result = res
        return self._format(name, res)

    # ── 反馈格式化 ────────────────────────────────────────────────────────
    def _format(self, name: str, res: dict) -> str:
        st, c = self.st, self.cfg
        ws = res.get("workspace") or {}
        acts = ws.get("actions") or {}
        violations = ws.get("static_violations") or []
        if violations:
            lines = "\n".join(f"  - {v}" for v in violations[:6])
            return (f"STATIC CHECK FAILED (scores zero until fixed):\n{lines}\n"
                    f"Implement these in your own CUDA kernels instead of calling torch.")
        comp, verify = acts.get("compile") or {}, acts.get("verify") or {}
        err = res.get("error_message") or ws.get("error_message") or "unknown error"
        if not ws.get("ok"):
            if not ws:  # repo 层就失败了 (缺文件之类), 没进 workspace
                return f"{name.upper()} FAILED:\n```\n{err[:c.max_verify_err_chars]}\n```"
            if comp and not comp.get("ok", True):
                return (f"COMPILATION FAILED ({comp.get('elapsed_s', '?')}s):\n"
                        f"```\n{_key_error(err)[:c.max_compile_err_chars]}\n```")
            if verify and not verify.get("ok", True):
                fb = verify.get("torch_fallback") or {}
                if fb:
                    ops = ", ".join(f"{k} x{v}" for k, v in sorted(fb.items()))
                    return (f"VERIFICATION REJECTED -- torch operator fallback detected "
                            f"in ModelNew.forward: {ops}. Every computation must go "
                            f"through your CUDA extension; replace these with your own "
                            f"kernels (this scores zero until fixed).")
                return (f"VERIFICATION FAILED "
                        f"({verify.get('tests_passed', 0)}/{verify.get('tests_total', 5)} "
                        f"trials passed):\n```\n{err[:c.max_verify_err_chars]}\n```")
            return f"{name.upper()} FAILED:\n```\n{err[:c.max_verify_err_chars]}\n```"

        if name == "compile":
            cached = " (cached)" if comp.get("cached") else ""
            return f"Compile OK{cached} in {comp.get('elapsed_s', '?')}s."
        if verify.get("ok"):
            st.verified_ever = True
            st.best_milestone = max(st.best_milestone, 1.0)
        if name == "verify":
            return (f"Verification PASSED "
                    f"({verify.get('tests_passed')}/{verify.get('tests_total')} trials, "
                    f"atol=rtol=1e-2). Run `profile` to measure performance.")
        if name == "analyze":
            return self._format_acu(acts.get("acu") or {})
        return self._format_profile(acts.get("profile") or {})

    def _format_profile(self, p: dict) -> str:
        st = self.st
        vs_eager, vs_compile = p.get("speedup_vs_eager"), p.get("speedup_vs_compile")
        milestone = 1.0
        if vs_eager and vs_eager >= SPEEDUP_MARGIN:
            milestone = 2.0
            if vs_compile and vs_compile >= SPEEDUP_MARGIN:
                milestone = 3.0
        if milestone > st.best_milestone or (
                milestone == st.best_milestone and (
                    st.best_profile is None
                    or (vs_compile or 0) > (st.best_profile.get("speedup_vs_compile") or 0))):
            st.best_milestone = max(st.best_milestone, milestone)
            st.best_profile = p
            st.best_files = dict(st.files)

        lines = [
            "Verification PASSED. Performance:",
            f"  your extension: {p.get('custom_ms')} ms",
            (f"  PyTorch eager:  {p.get('eager_ms')} ms   (speedup x{vs_eager})" if vs_eager
             else f"  PyTorch eager:  {p.get('eager_ms')} ms"),
        ]
        if p.get("compile_ms") is not None:
            lines.append(f"  torch.compile:  {p.get('compile_ms')} ms   (speedup x{vs_compile})")
        elif p.get("tc_error"):
            lines.append(f"  torch.compile:  unavailable for this task "
                         f"({str(p.get('tc_error'))[:120]})")
        tgt = []
        if not vs_eager or vs_eager < SPEEDUP_MARGIN:
            tgt.append("beat eager by >5%")
        elif p.get("compile_ms") is not None and (not vs_compile or vs_compile < SPEEDUP_MARGIN):
            tgt.append("beat torch.compile by >5%")
        lines.append("  target reached -- keep optimizing, every microsecond counts"
                     if not tgt else f"  next target: {', '.join(tgt)}")

        def table(rows, title):
            if not rows:
                return []
            out = [f"  {title}:"]
            for r in rows[:10]:
                out.append(f"    {r.get('cuda_time_us_per_iter', 0):>9.1f} us/iter "
                           f"{r.get('pct', 0):>5.1f}%  {r.get('name', '')[:80]}")
            return out

        lines += table(p.get("custom_kernels") or [], "your kernels (per iteration)")
        lines += table(p.get("eager_kernels") or [], "eager baseline breakdown")
        return "\n".join(lines)

    def _format_acu(self, a: dict) -> str:
        if not a.get("ok"):
            return f"ANALYZE FAILED:\n```\n{str(a.get('error_message'))[:self.cfg.max_verify_err_chars]}\n```"
        ks = a.get("kernels") or []
        if not ks:
            return "Verification PASSED. analyze: no custom kernel launches were captured."
        lines = ["Verification PASSED. Roofline profile of your kernels:",
                 "    compute%  memory%  dram%  occ(ach/theo)  regs  smem(B)   us     grid / block  name"]
        for k in ks[:12]:
            lines.append(
                f"    {_f(k.get('compute_pct')):>8} {_f(k.get('memory_pct')):>8} "
                f"{_f(k.get('dram_pct')):>6}  {_f(k.get('achieved_occupancy_pct')):>5}/"
                f"{_f(k.get('theoretical_occupancy_pct')):<6} {k.get('registers_per_thread', '?'):>5} "
                f"{k.get('shared_mem_per_block_bytes', '?'):>7} {_f(k.get('duration_us')):>6}  "
                f"{k.get('grid', '?')}/{k.get('block', '?')}  {str(k.get('name', ''))[:60]}")
        return "\n".join(lines)

    def _status_line(self) -> str:
        st = self.st
        files = ", ".join(f"{p} ({len(c.splitlines())} lines)"
                          for p, c in sorted(st.files.items())) or "(no files yet)"
        best = "none"
        if st.best_profile:
            p = st.best_profile
            best = (f"custom {p.get('custom_ms')}ms vs eager {p.get('eager_ms')}ms"
                    f" vs compile {p.get('compile_ms')}ms"
                    f" (x{p.get('speedup_vs_eager')} / x{p.get('speedup_vs_compile')})")
        elif st.verified_ever:
            best = "verified correct (not yet profiled)"
        return (f"[workspace: {files} | best result: {best} | "
                f"turn {st.turn_idx + 1}/{self.cfg.max_turns}]")

    # ── 结局 ──────────────────────────────────────────────────────────────
    async def _final_reward(self) -> tuple[float, bool]:
        st = self.st
        if self.cfg.reward_mode == "final":
            if "model_new.py" not in st.files:
                return (-1.0, False) if st.judged or not st.infra_events else (0.0, True)
            res = await self._evaluate("grade", dict(st.files))
            st.final_grade = {k: res.get(k) for k in (
                "resolved_correct", "resolved_fast", "milestone", "speedup_vs_eager",
                "speedup_vs_compile", "tests", "infra_error", "error_message")}
            if res.get("infra_error"):
                return 0.0, True
            return float(res.get("milestone") or -1), False
        if not st.judged and st.infra_events > 0:
            return 0.0, True      # 全是评测服务失败, 这条轨迹没被裁决
        if not st.verified_ever:
            return -1.0, False
        return float(max(st.best_milestone, 1.0)), False

    def _episode_metrics(self, reward: float, infra: bool) -> dict[str, Any]:
        st = self.st
        p = st.best_profile or {}
        m = {
            "episode/reward": reward,
            "episode/infra_error": float(infra),
            "episode/turns": float(st.turn_idx),
            "episode/heavy_calls": float(st.heavy_calls),
            "episode/correct": float(st.verified_ever),
            "episode/faster_eager": float(st.best_milestone >= 2),
            "episode/faster_compile": float(st.best_milestone >= 3),
            "episode/eval_seconds": st.eval_seconds,
        }
        if p.get("speedup_vs_compile"):
            m["episode/speedup_vs_compile"] = float(p["speedup_vs_compile"])
        if p.get("speedup_vs_eager"):
            m["episode/speedup_vs_eager"] = float(p["speedup_vs_eager"])
        return m


def _f(x) -> str:
    return "?" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))


_ERR_LINE_RE = re.compile(r"(error|Error|ERROR)[^\n]*")


def _key_error(err: str) -> str:
    """编译日志里先给出 error 行 (服务端已清洗截断, 这里只把关键行提前)。"""
    keys = [m.group(0) for m in _ERR_LINE_RE.finditer(err)]
    if not keys:
        return err
    head = "\n".join(dict.fromkeys(keys[:12]))
    return head + "\n---\n" + err
