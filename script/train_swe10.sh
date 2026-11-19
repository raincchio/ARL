#!/usr/bin/env bash
# Baseline -> SWE first-10 GRPO -> checkpoint evaluation -> comparison.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/script/swe10.py" "$@"
