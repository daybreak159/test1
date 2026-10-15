"""Verify deferred, digest-pinned SWE environments before their first use."""
import subprocess
import time
from pathlib import Path

from ..util import atomic_json, digest, read_json


def ensure_environments(tasks, environment, audit_path, *, wait_seconds=7200):
    """Wait for background image imports, then audit two independent resets.

    The immutable manifest declares all tasks and image digests up front. This
    append-only evidence file is operational readiness, not training data.
    """
    from ..services.backends import SWEEnvironment

    audit_path = Path(audit_path)
    audit = read_json(audit_path) if audit_path.exists() else {'records': {}}
    for task in tasks:
        image = environment['allowed_images'][task.id]
        expected = task.identity.history_or_environment_hash
        previous = audit['records'].get(task.id)
        if previous and previous['image'] == image and previous['initial_state_hash'] == expected:
            continue
        deadline = time.monotonic() + wait_seconds
        while subprocess.run(['docker', 'image', 'inspect', image], capture_output=True).returncode:
            atomic_json(audit_path.with_name('environment_wait.json'),
                        {'task_id': task.id, 'image': image, 'status': 'waiting_for_background_download'})
            if time.monotonic() >= deadline:
                raise RuntimeError('Background image import deadline exceeded: ' + task.id)
            time.sleep(10)
        pair = []
        for _ in range(2):
            instance = SWEEnvironment(task.public_payload, 11, environment)
            try:
                state = instance.state_hash
                observation_hash = digest(instance.observation)
                result, _ = instance.step({'command': 'git rev-parse HEAD && python -c "import sys; print(sys.version)"'}, 60)
                if state != expected or result['exit_code'] or instance.finalize('readiness audit')['patch'].strip():
                    raise ValueError('Deferred environment reset audit failed: ' + task.id)
                pair.append((state, observation_hash, result))
            finally:
                instance.close()
        if pair[0] != pair[1]:
            raise ValueError('Independent deferred resets differ: ' + task.id)
        audit['records'][task.id] = {'image': image, 'initial_state_hash': pair[0][0],
                                    'initial_observation_hash': pair[0][1], 'pair': pair, 'passed': True}
        atomic_json(audit_path, audit)
