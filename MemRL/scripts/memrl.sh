#!/usr/bin/env bash
# Use the existing verified Python, regardless of the caller's active shell environment.
set -euo pipefail
task_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
exec /home/ubuntu/coding/venvs/eur-alf-pilot-20261011/bin/python \
  "$task_source/start_memrl.py" "$@"
