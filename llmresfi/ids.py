"""Unique experiment identifier generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def generate_experiment_id(prefix: str = "exp") -> str:
    """Return a UTC-timestamped identifier suitable for a new experiment run."""

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix.strip()}_{stamp}_{uuid.uuid4().hex[:8]}"

