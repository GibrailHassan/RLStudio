from __future__ import annotations

import os
import random
from typing import Any

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    np = None  # runtime fallback when numpy absent

try:
    import torch  # type: ignore[import-not-found]  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # runtime fallback when torch absent


def seed_all(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy, Torch (if available) and return metadata."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        # CUDA seeding if GPUs available
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
        if deterministic and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_rand": random.random(),
        "np_rand": float(np.random.rand()) if np is not None else None,
    }
