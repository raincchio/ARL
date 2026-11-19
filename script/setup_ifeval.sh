#!/usr/bin/env bash
# Fetch hash-verified official files and install graders outside the training env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/envs/instruction_following/IFEvalEnv/setup.py"
