#!/usr/bin/env python3
"""Authenticated worker health/status/close from the training container."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arl.envs.coding.swe import _import_sweenv


async def main(args):
    ce = _import_sweenv()
    backend = ce.DockerBackend(ce.BackendConfig(endpoint=args.endpoint, token_file=args.token_file))
    try:
        payload = {"name": args.name} if args.action == "close" else {}
        print(json.dumps(await backend.rpc(args.action, **payload), indent=2))
    finally:
        await backend.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["health", "status", "close"], default="status", nargs="?")
    p.add_argument("--name", default="")
    p.add_argument("--endpoint", default="http://10.0.134.70:18770")
    p.add_argument("--token-file", default="/root/ai_data/swe/service/token")
    asyncio.run(main(p.parse_args()))
