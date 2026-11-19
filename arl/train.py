#!/usr/bin/env python3
"""ARL 训练入口 (torchrun 拉起, 每个进程一个训练 rank; rank0 另起 vLLM 子进程):

    torchrun --nproc_per_node=8 -m arl.train model_name=/models/Qwen/Qwen3.8-27B \
        log_path=./runs/cuda_grpo_v1 train_set=ops6k batch_size=8 group_size=8
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import chz

from arl.grpo import GRPOConfig, run_grpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from chz.blueprint._entrypoint import EntrypointHelpException
    from arl.utils.quiet_deepspeed import import_deepspeed_quietly

    blueprint = chz.Blueprint(GRPOConfig)
    try:
        if len(sys.argv) > 1:
            blueprint.apply_from_argv(sys.argv[1:], layer_name="command line")
    except EntrypointHelpException as e:
        print(e)
        sys.exit(0)
    cfg = blueprint.make()
    from arl.grpo import _validate
    try:
        _validate(cfg, int(os.environ.get("WORLD_SIZE", "1")))
    except (ValueError, NotImplementedError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    import_deepspeed_quietly()
    logger.info(f"ARL agentic GRPO: policy={cfg.model_name} env={cfg.env}/{cfg.train_set} "
                f"log_path={cfg.log_path}")
    exit_code = 0
    try:
        asyncio.run(run_grpo(cfg))
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
