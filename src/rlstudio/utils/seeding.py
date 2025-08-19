"""Seeding utilities for reproducibility (Milestone 1a).

Provides a single entry point `seed_all` that seeds Python, NumPy, and torch (if available)
and returns a dictionary capturing RNG states (sufficient for checkpointing later).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict

try:  # Optional torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - absence path
    torch = None  # type: ignore

try:  # Optional numpy
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def seed_all(seed: int, *, deterministic: bool = False) -> Dict[str, Any]:
    """Seed all available RNG sources.

    Args:
        seed: Base integer seed.
        deterministic: If True and torch present, enable deterministic algorithms (best-effort).
    Returns:
        Dict capturing seeds and raw RNG states where accessible.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np_state = None
    if np is not None:  # pragma: no branch - simple guard
        np.random.seed(seed)
        try:
            np_state = np.random.get_state()
        except Exception:  # pragma: no cover - fallback
            np_state = None

    torch_state = None
    if torch is not None:  # pragma: no branch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - may not run in CI
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass
        torch_state = torch.random.get_rng_state()

    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_random_example": random.random(),  # sample to show seeded randomness
        "np_state": np_state,
        "torch_state_len": int(torch_state.numel()) if torch_state is not None else None,
        "timestamp": time.time(),
    }
