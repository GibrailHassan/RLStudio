"""Core protocol interfaces for RLStudio (Milestone 1a).

These lightweight protocols define the expected surface area for key components.
They intentionally avoid prescribing frameworks (PyTorch/NumPy) at this stage.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Dict

from .batch import Batch


@runtime_checkable
class RLModule(Protocol):
    """Policy/value module interface.

    Methods may return a modified batch including action/value annotations.
    """

    def act(self, batch: Batch) -> Batch:  # pragma: no cover - protocol
        ...

    def value(self, batch: Batch) -> Batch:  # optional; may raise NotImplementedError
        ...  # pragma: no cover - protocol


@runtime_checkable
class ReplayBuffer(Protocol):
    """Experience storage."""

    def add(self, batch: Batch) -> None:  # pragma: no cover - protocol
        ...

    def sample(self, batch_size: int) -> Batch:  # pragma: no cover - protocol
        ...

    def __len__(self) -> int:  # pragma: no cover - protocol
        ...


@runtime_checkable
class Trainer(Protocol):
    """Training orchestrator."""

    def train_step(self) -> Dict[str, Any]:  # pragma: no cover - protocol
        ...

    def eval_step(self) -> Dict[str, Any]:  # pragma: no cover - protocol
        ...
