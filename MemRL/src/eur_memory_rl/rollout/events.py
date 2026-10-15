from __future__ import annotations

import threading

from ..util import IdentityConflict, digest


class ExternalTurnRouter:
    """A retry may change request_id, but cannot change the external input."""

    def __init__(self):
        self._seen = {}
        self._active = {}
        self._lock = threading.RLock()

    def read(self, execution_id, turn, select, precomputed=None):
        if turn.event_origin != "external_request":
            return None, False
        key = (execution_id, turn.task_id, turn.external_turn_id)
        fingerprint = digest([turn.text, turn.public_history, turn.public_time])
        with self._lock:
            if key in self._seen:
                prior, slate = self._seen[key]
                if prior != fingerprint:
                    raise IdentityConflict(f"Different input under the same external turn: {key}")
                if precomputed is not None and precomputed != slate:
                    raise IdentityConflict("Retry attempted to replace an existing exposure")
                return slate, False
            slate = precomputed if precomputed is not None else select()
            self._seen[key] = (fingerprint, slate)
            self._active[(execution_id, turn.task_id)] = slate.memory_block
            return slate, True

    def active_block(self, execution_id, task_id):
        return self._active.get((execution_id, task_id), "")
