#!/usr/bin/env bash
# Clone the pinned upstream snapshot and install static-rule dependencies in an isolated cache.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -c 'from arl.benchmarks.multicodeif import setup; print(setup())'
