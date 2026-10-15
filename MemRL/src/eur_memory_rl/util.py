from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value: Any) -> str:
    return json.dumps(
        plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seed_for(seed: int, *identity: Any) -> int:
    return int(digest([seed, *identity])[:15], 16)


def shuffled(items, seed: int):
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


class ConfigurationError(ValueError):
    pass


class InfrastructureError(RuntimeError):
    """Only environmental/model/transport/judge failures, never policy mistakes."""


class IdentityConflict(ValueError):
    pass


class GroupExcluded(RuntimeError):
    pass


def retry_infrastructure(call, attempts: int = 3):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return call()
        except InfrastructureError:
            if attempt + 1 == attempts:
                raise
