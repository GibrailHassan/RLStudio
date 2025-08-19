"""Minimal in-memory replay buffer (Milestone 1a).

Design goals (initial):
* Simplicity over performance.
* Store transitions column-wise for extensibility later.
* Accept either a `Batch` or individual components.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from ..core.batch import Batch


class ReplayBufferSimple:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: Deque[Batch] = deque(maxlen=capacity)

    def add(self, batch: Batch) -> None:
        self._data.append(batch)

    def sample(self, batch_size: int) -> Batch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._data:
            raise ValueError("buffer empty")
        if batch_size == 1:
            return self._data[-1]
        # naive uniform without replacement by taking most recent batch_size entries
        items = list(self._data)[-batch_size:]
        return Batch.merge(items)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": len(self),
            "batch_units": sum(b.size() for b in self._data),
        }
