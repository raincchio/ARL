"""Core RL interfaces and data containers.

Vendored from tinker_cookbook.rl.types with imports repointed to the local
backend. Backend-agnostic: no hosted-service calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import chz
from typing_extensions import TypedDict

from arl.models import types as tinker
from arl.algorithms.completers import StopCondition, TokensWithLogprobs
from arl.utils.misc_utils import safezip

Action: TypeAlias = list[int]
Observation: TypeAlias = "tinker.ModelInput"
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, float | int]
Logs: TypeAlias = dict[str, str | int | float]


@dataclass
class StepResult:
    """Result returned by :meth:`Env.step`."""

    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)


@dataclass
class Transition:
    """A single (observation, action, reward) tuple from a trajectory."""

    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)


class ActionExtra(TypedDict, total=False):
    """Extra metadata passed alongside an action to :meth:`Env.step`."""

    stop_reason: "tinker.StopReason"


class Env(ABC):
    """Stateful, single-use environment that one agent interacts with."""

    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        pass


@dataclass(frozen=True)
class Trajectory:
    """A complete episode: a sequence of transitions plus the final observation."""

    transitions: list[Transition]
    final_ob: Observation


@dataclass(frozen=True)
class RolloutError:
    """A captured error from a failed trajectory rollout."""

    error_type: str
    error_message: str


class EnvGroupBuilder(ABC):
    """Builds a group of environments (used by GRPO to center rewards)."""

    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        pass

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    async def cleanup(self) -> None:
        pass

    def logging_tags(self) -> list[str]:
        return []


@dataclass
class TrajectoryGroup:
    """A group of trajectories produced by one :class:`EnvGroupBuilder`."""

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: list[Metrics]
    rollout_errors: list[RolloutError] = field(default_factory=list)

    def get_total_rewards(self) -> list[float]:
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]


class RLDataset(ABC):
    """A dataset that produces batches of :class:`EnvGroupBuilder` instances."""

    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@chz.chz
class RLDatasetBuilder:
    """Abstract builder that constructs training and optional test RL datasets."""

    @abstractmethod
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        pass
