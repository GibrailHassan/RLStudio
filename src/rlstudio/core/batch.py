from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BatchData = Any  # Placeholder tensor/array type; refined after adding torch/np guards.


@dataclass
class Batch:
    """Minimal experience container.

    Fields kept intentionally generic for early milestone; algorithms may extend via composition.
    """

    observations: BatchData
    actions: BatchData
    rewards: BatchData
    dones: BatchData
    next_observations: Optional[BatchData]
    infos: Any = None  # Structure not enforced here.

    def size(self) -> int:
        """Best effort batch dimension inference.

        Tries observations first; user responsible for providing consistent shapes.
        """
        obs = self.observations
        try:
            return len(obs)  # type: ignore[arg-type]
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError("Cannot infer batch size from observations") from e

    __len__ = size

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "next_observations": self.next_observations,
            "infos": self.infos,
        }

    @staticmethod
    def merge(batches: List["Batch"]) -> "Batch":
        if not batches:
            raise ValueError("Cannot merge empty list of batches")
        if len(batches) == 1:
            return batches[0]

        def _cat(values):  # naive concatenation supporting list-like
            first = values[0]
            if isinstance(first, list):
                out = []
                for v in values:
                    out.extend(v)
                return out
            # Fallback: attempt + operator
            acc = first
            for v in values[1:]:
                acc = acc + v  # type: ignore[operator]
            return acc

        return Batch(
            observations=_cat([b.observations for b in batches]),
            actions=_cat([b.actions for b in batches]),
            rewards=_cat([b.rewards for b in batches]),
            dones=_cat([b.dones for b in batches]),
            next_observations=_cat([b.next_observations for b in batches]) if batches[0].next_observations is not None else None,
            infos=[b.infos for b in batches],
        )
