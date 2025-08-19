from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Batch:
    """Experience batch container.

    Column-wise storage of transitions. Kept intentionally small for milestone 1a.
    """

    observations: List[Any]
    actions: List[Any]
    rewards: List[float]
    terminals: List[bool]
    next_observations: List[Any]
    infos: List[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.observations)
        if not (
            len(self.actions)
            == len(self.rewards)
            == len(self.terminals)
            == len(self.next_observations)
            == n
        ):
            raise ValueError("All core columns must have equal length")
        if self.infos and len(self.infos) not in (0, n):
            raise ValueError("infos length must be 0 or match other columns")

    def size(self) -> int:
        return len(self.rewards)

    @staticmethod
    def merge(batches: Iterable["Batch"]) -> "Batch":
        batches = list(batches)
        if not batches:
            raise ValueError("no batches to merge")
        if len(batches) == 1:
            b = batches[0]
            return Batch(
                list(b.observations),
                list(b.actions),
                list(b.rewards),
                list(b.terminals),
                list(b.next_observations),
                infos=[dict(x) for x in b.infos],
            )
        obs: List[Any] = []
        acts: List[Any] = []
        rews: List[float] = []
        terms: List[bool] = []
        nxt: List[Any] = []
        infos: List[dict[str, Any]] = []
        for b in batches:
            obs.extend(b.observations)
            acts.extend(b.actions)
            rews.extend(b.rewards)
            terms.extend(b.terminals)
            nxt.extend(b.next_observations)
            infos.extend(b.infos)
        return Batch(obs, acts, rews, terms, nxt, infos=infos)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminals": self.terminals,
            "next_observations": self.next_observations,
            "infos": self.infos,
        }

    @staticmethod
    def from_transition(
        obs: Any,
        action: Any,
        reward: float,
        terminal: bool,
        next_obs: Any,
        info: dict[str, Any] | None = None,
    ) -> "Batch":
        return Batch(
            [obs],
            [action],
            [reward],
            [terminal],
            [next_obs],
            infos=[info] if info is not None else [],
        )
