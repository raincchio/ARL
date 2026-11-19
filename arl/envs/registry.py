"""Lazy environment registrations shared by all task categories."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

EnvironmentCategory = Literal["coding", "instruction_following"]


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    category: EnvironmentCategory
    module: str
    config_class: str
    source_class: str
    aliases: tuple[str, ...] = ()
    interface: str = "agent_messages"
    default_harness: str = ""
    supported_harnesses: tuple[str, ...] = ()
    backend_interfaces: tuple[tuple[str, str], ...] = ()

    def make_task_source(self, **kwargs):
        module = import_module(self.module)
        return getattr(module, self.source_class)(getattr(module, self.config_class)(**kwargs))

    def from_training_config(self, cfg):
        return import_module(self.module).make_source_from_config(cfg)
