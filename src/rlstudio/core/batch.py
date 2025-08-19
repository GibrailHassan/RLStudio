from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class Batch:
    """Minimal batch container for experience tuples.

    Stores columns as sequences. Size is inferred from length of first column.
    """

    observations: list[Any]
    actions: list[Any]
    rewards: list[float]

    def size(self) -> int:
        return len(self.rewards)

    @staticmethod
    def merge(batches: Iterable[Batch]) -> Batch:
        obs: list[Any] = []
        acts: list[Any] = []
        rews: list[float] = []
        for b in batches:
            obs.extend(b.observations)
            acts.extend(b.actions)
            rews.extend(b.rewards)
        return Batch(obs, acts, rews)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
        }
