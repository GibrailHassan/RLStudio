"""Minimal in-memory replay buffer (Milestone 1a).

Design goals (initial):
* Simplicity over performance.
* Store transitions column-wise for extensibility later.
* Accept either a `Batch` or individual components.
"""

from __future__ import annotations

from collections import deque

from ..core.batch import Batch


class ReplayBufferSimple:
    """Simple append-only replay buffer with naive recent sampling."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        # Internal storage: bounded deque of Batch objects
        self._data: deque[Batch] = deque(maxlen=capacity)

    def add(self, batch: Batch) -> None:
        self._data.append(batch)

    def sample(self, batch_size: int) -> Batch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._data:
            raise ValueError("buffer empty")
        if batch_size == 1:
            return self._data[-1]
        # Sampling policy (milestone 1a): deterministic, returns the most recent
        # N batches (no randomization). This keeps implementation trivial and
        # aids reproducibility. Future versions can introduce random sampling
        # or prioritized strategies without changing the public API.
        items = list(self._data)[-batch_size:]
        return Batch.merge(items)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._data)

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self),
            "batch_units": sum(b.size() for b in self._data),
        }
