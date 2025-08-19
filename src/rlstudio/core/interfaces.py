from __future__ import annotations

from typing import Protocol, runtime_checkable

from .batch import Batch


@runtime_checkable
class RLModule(Protocol):
    def act(self, batch: Batch) -> Batch: ...


@runtime_checkable
class ReplayBuffer(Protocol):
    def add(self, batch: Batch) -> None: ...
    def sample(self, batch_size: int) -> Batch: ...
    def stats(self) -> dict[str, int]: ...


@runtime_checkable
class Trainer(Protocol):
    def train_step(self) -> dict[str, float]: ...
