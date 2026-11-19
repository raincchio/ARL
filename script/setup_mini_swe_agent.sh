#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the training runtime's pydantic>=2, jinja2, rich, dotenv, platformdirs.
# No LiteLLM/API client: model calls use ARL's local token sampler.
python3 -m pip install --index-url https://pypi.org/simple --no-deps \
  --target "$ROOT/.cache/benchmarks/mini-swe-agent" mini-swe-agent==2.4.6
cd "$ROOT"
python3 -c 'from arl.harnesses.mini_swe import check_dependencies; check_dependencies()'
