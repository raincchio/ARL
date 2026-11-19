"""SWE-bench AgentEnv: persistent isolated repository, final 0/1 official reward."""
from __future__ import annotations

from dataclasses import dataclass
import logging

from .backend import BackendError
from .tools import TOOLS, parse_tool_calls, split_reasoning

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    max_turns: int = 64
    command_timeout: int = 120
    eval_timeout: int = 1800
    max_calls_per_turn: int = 4
    max_tool_result_chars: int = 16000
    keep_full_turns: int = 4

    def __post_init__(self):
        if min(self.max_turns, self.command_timeout, self.eval_timeout, self.max_calls_per_turn,
               self.max_tool_result_chars, self.keep_full_turns) < 1:
            raise ValueError("SWE turn, timeout, output and history limits must be positive")


@dataclass
class StepOutput:
    reward: float
    done: bool
    tool_results: list
    metrics: dict
    infra_error: bool = False


SYSTEM = """You are a software engineer fixing an issue in /testbed, already checked out at the task's base commit.
Use the bash tool to inspect the repository, edit source files and run your own tests.
Each bash call starts a fresh shell in /testbed; file edits persist. Internet access is disabled.
Fix the issue without modifying tests or committing changes. Do not access or change evaluation infrastructure.
Call submit when finished; a final response without tool calls also submits the current changes.
Only the final repository diff is graded, in a fresh environment with the official tests.
"""


class SweWorkspaceEnv:
    tools = TOOLS

    def __init__(self, task, workspace, cfg=None):
        self.task, self.workspace = task, workspace
        self.cfg = cfg or EnvConfig()
        self.done = False
        self._history = []
        self._turns = 0
        self._reward = 0.0
        self._infra = False
        self._patch = ""
        self._grade = {}
        self._closed = False

    async def reset(self):
        # Lazy container start: rollout creates all trajectories before sampling.
        if self._history or self.done or self._closed:
            raise RuntimeError("SWE workspace instances are single-use")

    def initial_messages(self):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (
            f"Repository: {self.task.repo}\nBase commit: {self.task.base_commit}\n"
            f"Fix this issue (up to {self.cfg.max_turns} turns):\n<issue>\n"
            f"{self.task.problem_statement}\n</issue>")}]

    workspace_dir = "/testbed"
    submit_message = "Submitting the final repository diff."

    def messages(self, aggressiveness=0):
        result = self.initial_messages()
        keep = max(1, self.cfg.keep_full_turns >> max(0, aggressiveness - 2))
        cut = max(0, len(self._history) - keep)
        if aggressiveness >= 3 and cut:
            result.append({"role": "user", "content": (
                f"[{cut} earlier turns omitted. All file edits persist in {self.workspace_dir}. "
                "Inspect the current files or git diff if needed.]")})
        for i, (response, feedback) in enumerate(self._history):
            if aggressiveness >= 3 and i < cut:
                continue
            reasoning, content = split_reasoning(response)
            msg = {"role": "assistant", "content": content}
            if aggressiveness == 0 or i >= cut:
                msg["reasoning_content"] = reasoning
            elif aggressiveness >= 2:
                msg["content"] = "[Earlier tool calls compressed; edits persist.]"
            result.append(msg)
            for name, text in feedback:
                if aggressiveness >= 2 and i < cut:
                    text = text[:200] + " [compressed]"
                result.append({"role": "tool", "name": name, "content": text})
        return result

    async def step(self, text, *, truncated=False):
        if self.done:
            raise RuntimeError("step after SWE episode ended")
        calls, errors = parse_tool_calls(text)
        # A truncated command may be incomplete: never execute its prefix.
        if truncated:
            calls = []
            errors = ["Response truncated. Use shorter reasoning and emit a complete tool call next turn."]
        if len(calls) > self.cfg.max_calls_per_turn:
            errors.append(f"At most {self.cfg.max_calls_per_turn} calls per turn; no commands executed.")
            calls = []
        feedback = [("bash", error) for error in errors]
        submitted = False
        infra = False
        try:
            for call in calls:
                if call["name"] == "submit":
                    submitted = True
                    feedback.append(("submit", self.submit_message))
                    break
                output = await self.workspace.execute(call["arguments"]["command"],
                    timeout=self.cfg.command_timeout, output_limit=self.cfg.max_tool_result_chars)
                feedback.append(("bash", f"exit_code={output['exit_code']}\n{output['output']}"))
        except BackendError as e:
            infra = True
            feedback.append(("bash", f"Workspace infrastructure failed: {e}"))
        self._turns += 1
        self._history.append((text, feedback))
        terminal = infra or submitted or (not truncated and not calls and not errors) or self._turns >= self.cfg.max_turns
        metrics = {"n_tool_calls": len(calls), "n_invalid_calls": len(errors),
                   "format_ok": float(not errors), "truncated": float(truncated)}
        if terminal:
            out = await self._finish(infra=infra)
            out.metrics.update(metrics)
            out.tool_results = feedback
            return out
        return StepOutput(0., False, feedback, metrics)

    async def finish(self, reason=""):
        if self.done:
            raise RuntimeError("finish after SWE episode ended")
        return await self._finish()

    async def finish_submission(self, patch):
        """Grade exactly the patch emitted by an external harness submission."""
        if self.done:
            raise RuntimeError("finish after SWE episode ended")
        return await self._finish(submission=patch)

    async def _finish(self, infra=False, submission=None):
        self.done = True
        self._infra = infra
        try:
            if not infra:
                result = await self.workspace.patch() if submission is None else {"patch": submission}
                self._patch = result.get("patch", "")
                if result.get("agent_error"):
                    self._grade = {"resolved": False, "error": result["agent_error"]}
                elif self._patch.strip():
                    self._grade = await self.workspace.grade(self._patch, self.cfg.eval_timeout)
                    self._infra = bool(self._grade.get("infra_error"))
                else:
                    self._grade = {"resolved": False, "error": "empty patch"}
                self._reward = float(bool(self._grade.get("resolved"))) if not self._infra else 0.
        except BackendError as e:
            self._infra = True
            self._grade = {"error": str(e)}
        finally:
            await self.close()
        return StepOutput(self._reward, True, [], {
            "correct": None if self._infra else self._reward,
            "resolved": self._reward, "episode/reward": self._reward,
            "episode/turns": self._turns, "patch_bytes": len(self._patch.encode())}, self._infra)

    def summary(self):
        # Do not return task.record: it contains trusted gold patches and tests.
        return {"instance_id": self.task.instance_id, "repo": self.task.repo,
                "workspace_id": getattr(self.workspace, "name", ""),
                "worker_id": getattr(self.workspace, "placement", {}).get("worker_id", ""),
                "image_id": getattr(self.workspace, "placement", {}).get("image_id", ""),
                "turns": self._turns, "resolved": bool(self._reward), "infra_error": self._infra,
                "model_patch": self._patch, "grade": self._grade,
                "final_files": {"model.patch": self._patch}}

    async def close(self):
        if not self._closed:
            try:
                await self.workspace.close()
                self._closed = True
            except BackendError as e:
                logger.warning("[sweenv] workspace cleanup failed: %s", e)
