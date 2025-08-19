# RLStudio Architecture (Draft)

Status: Draft (Milestone 1a)

## Layered Overview

```
Environment(s) -> Experience -> ReplayBuffer -> Trainer <-> RLModule (policy/value) -> Optimizer
                                 |                \
                                 |                 -> Metrics / Logger
                                 -> Checkpointing / Seeding / Config
```

Core layers:
1. RLModule: Encapsulates policy (and optional value) forward passes. Pure compute; no env logic.
2. ReplayBuffer: Storage & sampling of experience `Batch` objects (or decomposed columns).
3. Trainer: Orchestrates data collection, sampling, optimization steps, evaluation, logging.
4. Utilities: Seeding, checkpoint IO, configuration, metrics logging, plugin/registry.

## Batch Specification

`Batch` is the minimal container passed between components. It standardizes field names ensuring reproducibility and interop.

Required fields (first milestone):
| Field | Type | Shape (example) | Description |
|-------|------|-----------------|-------------|
| observations | Any / Tensor | (B, *obs_shape) | Raw observations prior to action selection. |
| actions | Any / Tensor | (B, *action_shape) | Actions taken. |
| rewards | float / Tensor | (B,) | Scalar rewards associated with transitions. |
| dones | bool / Tensor | (B,) | Episode termination flags (True where transition ended an episode). |
| next_observations | Any / Tensor | (B, *obs_shape) | Observations after step (optional for final transitions). |
| infos | dict | (len=B) | Per-step auxiliary metadata (env-specific). |

Rules:
* All tensor-like columns must share leading batch dimension B.
* `next_observations` MAY be None for terminal entries; consumers must handle missing next state.
* `infos` is a list or dict-of-lists; structure intentionally opaque to core algorithms.
* Additional algorithm-specific columns (e.g. log_probs, values) are stored in derived batches or separate structures.

Utility operations (implemented incrementally):
* `merge(list[Batch]) -> Batch`: concatenates along batch dim.
* `select(indices) -> Batch`: indexing operation.
* `to(device) -> Batch`: best-effort device movement (if tensors present).

## Interfaces (Milestone 1a scope)

```
class RLModule(Protocol):
    def act(self, batch: Batch) -> Batch: ...  # adds actions/log_probs
    def value(self, batch: Batch) -> Batch: ...  # optional; adds value estimates

class ReplayBuffer(Protocol):
    def add(self, batch: Batch) -> None: ...
    def sample(self, batch_size: int) -> Batch: ...
    def __len__(self) -> int: ...

class Trainer(Protocol):
    def train_step(self) -> dict: ...
    def eval_step(self) -> dict: ...
```

These protocols target clarity over completeness; future expansions (distributed, multi-agent) will extend them.

## Seeding & Reproducibility (Preview)
Single entry-point `seed_all(seed: int)` will coordinate Python, NumPy, torch (if installed), and environment-specific seeds; returns a structured report of states captured for checkpointing.

## Checkpoint Schema (Preview)
Minimal dictionary:
```
{
  "version": 1,
  "module_state": {...},
  "trainer_state": {...},
  "replay_meta": {"size": N},
  "rng_state": {"python": ..., "numpy": ..., "torch": ...},
  "meta": {"created_at": iso8601, "run_id": str}
}
```

## Metrics & Logging (Preview)
Lightweight JSONL logger writing one record per event: `{ts, step, key, value, tags}`.

## Future Extensions
* Vectorized env runners
* Distributed rollout workers
* Registry-driven algorithm plug-ins
* Enhanced schema validation via pydantic or attrs (optional)

---
This document will be iterated as interfaces solidify. Link updates in the README will point here once stable.
