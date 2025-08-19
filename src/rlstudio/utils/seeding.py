from __future__ import annotations

import os
import random
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def seed_all(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy, Torch (if available) and return metadata."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)  # type: ignore[attr-defined]
    if torch is not None:
        torch.manual_seed(seed)  # type: ignore[attr-defined]
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_rand": random.random(),
        "np_rand": float(np.random.rand()) if np is not None else None,  # type: ignore[attr-defined]
    }
