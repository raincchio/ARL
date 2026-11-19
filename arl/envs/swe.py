"""Compatibility imports; SWE-bench tasks now live in arl.envs.coding.swe."""
from .coding.swe import SweSourceConfig, SweTaskSource, _import_sweenv

__all__ = ["SweSourceConfig", "SweTaskSource"]
