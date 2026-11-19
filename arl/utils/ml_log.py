"""Minimal metrics logger, replacing tinker_cookbook.utils.ml_log.

loop.py uses only ``setup_logging(...) -> logger`` with ``.log_metrics(metrics,
step)`` and ``.close()``. We write scalar metrics to ``metrics.jsonl`` and, if a
wandb project is configured and wandb is importable, mirror them to wandb.
"""

from __future__ import annotations

import json
import logging
import numbers
import os
from pathlib import Path
import socket
import time
from typing import Any

logger = logging.getLogger(__name__)


def _host_ip() -> str | None:
    """The outbound IP of this host (see rlhf.ppo.rm_service._outbound_ip:
    a UDP "connect" sends nothing, it just resolves which source IP the
    default route uses, avoiding 127.0.1.1 / docker-bridge answers)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None
    finally:
        s.close()


# Switches the launch scripts pass as env vars rather than CLI args, so they never reach
# the cfg (e.g. KERNEL_RL_MICRO_BSZ, KERNEL_RL_EVAL_MODE, KERNEL_STATIC_FORBIDDEN).
_ENV_PREFIXES = ("KERNEL_RL_", "KERNEL_STATIC_", "KERNEL_EVAL_")
# May carry credentials (http://user:pass@host).
_ENV_EXCLUDE = {"KERNEL_RL_WANDB_PROXY"}


def env_snapshot() -> dict[str, str]:
    return dict(sorted(
        (k, v) for k, v in os.environ.items()
        if k.startswith(_ENV_PREFIXES) and k not in _ENV_EXCLUDE
    ))


def _record_env(log_dir: str, env: dict[str, str]) -> None:
    """One line per launch in <log_dir>/run_env.jsonl (same-dir resumes append)."""
    logger.info("run env switches: " + " ".join(f"{k}={v}" for k, v in env.items()))
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run_env.jsonl"), "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env}) + "\n")
    except OSError as e:
        logger.warning(f"could not write run_env.jsonl ({e})")


class MLLogger:
    def __init__(self, log_dir: str, wandb_run: Any | None = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "metrics.jsonl")
        self._fh = open(self._path, "a")
        self._wandb_run = wandb_run

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        # Keep only json-serializable scalars for the jsonl record.
        record: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, numbers.Real) or isinstance(v, (str, bool)):
                record[k] = v
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(
                    {k: v for k, v in metrics.items() if isinstance(v, numbers.Real)},
                    step=step,
                )
            except Exception as e:  # never let logging kill training
                logger.warning(f"wandb log failed: {e}")

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception:
                    pass


def setup_logging(
    log_dir: str,
    wandb_project: str | None = None,
    config: Any | None = None,
    wandb_name: str | None = None,
) -> MLLogger:
    env = env_snapshot()
    _record_env(log_dir, env)
    wandb_run = None
    if wandb_project:
        try:
            import os

            import wandb  # type: ignore

            cfg = None
            if config is not None:
                cfg = config if isinstance(config, dict) else getattr(config, "__dict__", None)
            # Record where this run actually executed in the wandb config.
            cfg = dict(cfg or {})
            cfg["host_ip"] = _host_ip()
            cfg["hostname"] = socket.gethostname()
            cfg["env"] = env
            # Proxy only for wandb: the run env strips the global http(s)_proxy (so
            # internal traffic -- eval endpoint, etc. -- stays direct), and the proxy
            # needed to reach api.wandb.ai is handed to wandb ALONE via Settings.
            # Default to the training network proxy; explicitly empty disables
            # this W&B-specific proxy (launch scripts clear global proxies).
            init_kwargs: dict[str, Any] = dict(project=wandb_project, name=wandb_name or Path(log_dir).resolve().name,
                                              dir=str(Path(log_dir).resolve()), config=cfg)
            proxy = os.environ.get("KERNEL_RL_WANDB_PROXY", "http://10.0.134.70:10809").strip()
            if proxy:
                init_kwargs["settings"] = wandb.Settings(http_proxy=proxy, https_proxy=proxy)
            wandb_run = wandb.init(**init_kwargs)
        except Exception as e:
            logger.warning(f"wandb init failed ({e}); continuing without wandb")
    return MLLogger(log_dir, wandb_run=wandb_run)
