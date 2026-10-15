from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

from ..util import atomic_json, file_hash, read_json


def random_state():
    import torch

    result = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np

        result["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return result


def restore_random_state(state):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint cannot resume without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def adapter_state(shared):
    return {
        role: {name: p.detach().cpu().clone() for name, p in group}
        for role, group in shared.role_parameters.items()
    }


def restore_adapters(shared, state):
    import torch

    if set(state) != set(shared.role_parameters):
        raise ValueError("Checkpoint lacks a complete role set")
    with torch.no_grad():
        for role, group in shared.role_parameters.items():
            if set(state[role]) != {name for name, _ in group}:
                raise ValueError("Adapter architecture drift")
            for name, p in group:
                if p.shape != state[role][name].shape:
                    raise ValueError("Adapter shape drift")
                p.copy_(state[role][name].to(p.device, p.dtype))


class CheckpointStore:
    def __init__(self, root, domain, base_revision, protocol_hash):
        self.root = Path(root) / domain / base_revision / "E_U_R"
        self.domain, self.base_revision, self.protocol_hash = domain, base_revision, protocol_hash

    def publish(self, shared, optimizers, iteration, data_state, extra=None):
        import torch

        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"iteration-{iteration:06d}"
        if final.exists():
            raise FileExistsError(f"Checkpoint already exists: {final}")
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        bundle = {
            "format": "eur-checkpoint-v1",
            "iteration": iteration,
            "domain": self.domain,
            "base_revision": self.base_revision,
            "base_hash": shared.base_hash,
            "protocol_hash": self.protocol_hash,
            "adapters": adapter_state(shared),
            "optimizers": {role: optimizer.state_dict() for role, optimizer in optimizers.items()},
            "rng": random_state(),
            "data_state": data_state,
            "extra": extra or {},
        }
        path = stage / "state.pt"
        torch.save(bundle, path)
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
        atomic_json(
            stage / "manifest.json",
            {
                "iteration": iteration,
                "domain": self.domain,
                "protocol_hash": self.protocol_hash,
                "base_hash": shared.base_hash,
                "state_sha256": file_hash(path),
                "roles": list(shared.role_parameters),
                "model": shared.manifest(),
            },
        )
        # Directory rename is the commit. Incomplete staging directories are never resumed.
        os.replace(stage, final)
        try:
            atomic_json(self.root / "latest.json", {"checkpoint": final.name, "iteration": iteration})
        except OSError:
            # A complete directory is already committed. Do not roll back weights after commit.
            import warnings

            warnings.warn(
                "Checkpoint committed; latest pointer unavailable. Resume discovers complete directories."
            )
        return final

    def load(self, shared, optimizers=None, checkpoint=None, restore_rng=True, strict_protocol=True):
        import torch

        complete = sorted(p for p in self.root.glob("iteration-*") if (p / "manifest.json").is_file())
        if checkpoint is None and not complete:
            raise FileNotFoundError("No complete joint checkpoint")
        directory = Path(checkpoint) if checkpoint else complete[-1]
        manifest = read_json(directory / "manifest.json")
        if file_hash(directory / "state.pt") != manifest["state_sha256"]:
            raise ValueError("Checkpoint integrity failure")
        # Only load trusted local checkpoints created by this application; optimizer/RNG use pickle.
        bundle = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
        if not strict_protocol and optimizers is not None:
            raise ValueError("Training resumes cannot change protocol; only frozen evaluation may do so")
        keys = (
            ("domain", "base_revision", "base_hash", "protocol_hash")
            if strict_protocol
            else ("domain", "base_revision", "base_hash")
        )
        expected = (
            (self.domain, self.base_revision, shared.base_hash, self.protocol_hash)
            if strict_protocol
            else (self.domain, self.base_revision, shared.base_hash)
        )
        actual = tuple(bundle[k] for k in keys)
        if actual != expected or bundle["iteration"] != manifest["iteration"]:
            raise ValueError("Checkpoint domain/base/protocol/iteration mismatch")
        restore_adapters(shared, bundle["adapters"])
        if optimizers is not None:
            if set(optimizers) != set(bundle["optimizers"]):
                raise ValueError("Optimizer set mismatch")
            for role, optimizer in optimizers.items():
                optimizer.load_state_dict(bundle["optimizers"][role])
        if restore_rng:
            restore_random_state(bundle["rng"])
        shared.version = f"iteration-{bundle['iteration']}"
        return bundle
