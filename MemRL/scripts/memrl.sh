#!/usr/bin/env bash
# Use the existing verified Python, regardless of the caller's active shell environment.
set -euo pipefail
task_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
# WSL CUDA offload can hold more than SSH's default 1024 descriptors.
# Raise this process tree only; never change system or login configuration.
task_nofile_hard=$(ulimit -Hn)
task_nofile_soft=$(ulimit -Sn)
task_nofile_target=65536
if [[ "$task_nofile_hard" != unlimited ]] && (( task_nofile_hard < task_nofile_target )); then
  task_nofile_target=$task_nofile_hard
fi
if (( task_nofile_soft < task_nofile_target )); then
  ulimit -Sn "$task_nofile_target"
fi
echo "MemRL process open-file limit: $(ulimit -Sn)"
exec /home/ubuntu/coding/venvs/eur-alf-pilot-20261011/bin/python \
  "$task_source/start_memrl.py" "$@"
