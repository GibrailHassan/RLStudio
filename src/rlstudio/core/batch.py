"""Batch container for RL transitions.

Unified, conflict-free version with backward compatibility for earlier 'dones' naming.
Primary field retained as 'dones' (list[bool]); a read-only alias property 'terminals'
is provided for forward-looking terminology.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Batch:
    observations: list[Any]
    actions: list[Any]
    rewards: list[float]
    dones: list[bool]
    next_observations: list[Any]
    infos: list[dict[str, Any]] = field(default_factory=list)

    # --- compatibility alias -------------------------------------------------
    @property
    def terminals(self) -> list[bool]:  # pragma: no cover - simple alias
        return self.dones

    # --- lifecycle -----------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.observations)
        if not (
            len(self.actions)
            == len(self.rewards)
            == len(self.dones)
            == len(self.next_observations)
            == n
        ):
            raise ValueError("All core columns must have equal length")
        if self.infos and len(self.infos) not in (0, n):
            raise ValueError("infos length must be 0 or match other columns")

    # --- basic API -----------------------------------------------------------
    def size(self) -> int:
        return len(self.rewards)

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

    # --- constructors --------------------------------------------------------
    @staticmethod
    def from_transition(
        obs: Any,
        action: Any,
        reward: float,
        done: bool | None,
        next_obs: Any,
        info: dict[str, Any] | None = None,
        *,
        terminal: bool | None = None,  # optional alias
    ) -> Batch:
        flag = done if done is not None else bool(terminal)
        return Batch(
            [obs],
            [action],
            [reward],
            [flag],
            [next_obs],
            infos=[info] if info is not None else [],
        )

    # --- operations ----------------------------------------------------------
    @staticmethod
    def merge(batches: Iterable[Batch]) -> Batch:
        batches = list(batches)
        if not batches:
            raise ValueError("no batches to merge")
        if len(batches) == 1:
            b = batches[0]
            return Batch(
                list(b.observations),
                list(b.actions),
                list(b.rewards),
                list(b.dones),
                list(b.next_observations),
                infos=[dict(x) for x in b.infos],
            )
        obs: list[Any] = []
        acts: list[Any] = []
        rews: list[float] = []
        dones: list[bool] = []
        nxt: list[Any] = []
        infos: list[dict[str, Any]] = []
        for b in batches:
            obs.extend(b.observations)
            acts.extend(b.actions)
            rews.extend(b.rewards)
            dones.extend(b.dones)
            nxt.extend(b.next_observations)
            infos.extend(b.infos)
        return Batch(obs, acts, rews, dones, nxt, infos=infos)
