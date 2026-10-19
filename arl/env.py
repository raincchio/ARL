"""Environment variable loading utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from arl.netenv import harden_network_env

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load environment variables from .env file.

    Searches for .env file in the following order:
    1. Current working directory
    2. kernel-rl package root directory

    Environment variables already set will NOT be overwritten.
    """
    # Try current working directory first
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)
        logger.debug(f"Loaded environment from {cwd_env}")
        return

    # Try package root directory
    package_root = Path(__file__).parent.parent
    root_env = package_root / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
        logger.debug(f"Loaded environment from {root_env}")
        return

    logger.debug("No .env file found")


def setup_environment() -> None:
    """Load .env file and set up environment variables.

    This function:
    1. Loads variables from .env file (if present)
    2. Auto-detects KERNELBENCH_ROOT if not set
    3. Warns if TINKER_API_KEY is not set
    4. Fixes up this machine's DNS / proxy environment (arl.netenv)
    """
    # Load .env file first
    load_env()

    # Set KERNELBENCH_ROOT if not already set
    if "KERNELBENCH_ROOT" not in os.environ:
        candidates = [
            "/workspace/KernelBench",
            "/workspace/kernel_dev/KernelBench",
            os.path.expanduser("~/KernelBench"),
            str(Path(__file__).parent.parent.parent / "KernelBench"),
        ]
        for path in candidates:
            if os.path.exists(path):
                os.environ["KERNELBENCH_ROOT"] = os.path.abspath(path)
                logger.info(f"Set KERNELBENCH_ROOT={os.environ['KERNELBENCH_ROOT']}")
                break

    # Training runs on the self-hosted vLLM + DeepSpeed stack (r3lm.models);
    # no hosted Tinker service / TINKER_API_KEY is required. DeepSpeed needs a
    # single-process torch.distributed rendezvous when not launched via `deepspeed`.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # 最后一步: 按需修正这台机器的 DNS / 代理环境 —— 主机名解析不了、代理 env 大小写
    # 指到两个地址、内网服务被代理吞掉, 这些都是"机器给的环境不对"而不是代码的问题,
    # 但代价是几十秒起步的超时, 而且换台机器才现形。环境正常时整段是 no-op, 细节和
    # 关掉的办法见 arl/netenv.py。放在 MASTER_ADDR 之后, 它才能一起进 no_proxy。
    harden_network_env()
