# RLStudio

Early-stage reinforcement learning framework.

## Status

Milestone 1a: Core Contracts & Reproducibility (in progress).

Implemented so far:

* Batch container with validation & helpers (field name `dones` with read‑only alias `terminals`)
* Minimal in‑memory `ReplayBufferSimple`
* Core Protocols: `RLModule`, `ReplayBuffer`, `Trainer`
* Deterministic seeding utility (`seed_all`)
* Tooling: Ruff (lint+format), mypy (strict), pytest

## Vision

Flexible, scalable, reproducible RL with layered APIs and strong testing discipline.

## Quick Start

Install (editable + dev extras):

```bash
pip install -e .[dev]
pytest -q  # run test suite (should be green)
```

Basic usage:

```python
from rlstudio import Batch, ReplayBufferSimple, seed_all

seed_all(42)
rb = ReplayBufferSimple(capacity=3)
rb.add(Batch.from_transition(obs=0, action=1, reward=1.0, terminal=False, next_obs=1))
rb.add(Batch.from_transition(obs=1, action=0, reward=0.5, terminal=False, next_obs=2))
sample = rb.sample(1)
print(sample.rewards)
```

### Dev Environment (uv optional)

You can use [uv](https://github.com/astral-sh/uv) for faster installs:

```bash
uv venv && .venv\\Scripts\\activate  # Windows
uv pip install -e .[dev]
```

Tooling stack: Ruff (lint + format + import sorting), mypy (strict types), pytest.

Ruff acts as both formatter and linter (Black/Flake8 removed). Run:

```bash
ruff check . && ruff format --check .
```

## Batch Notes

The canonical termination field is stored as `dones`; a compatibility alias property `terminals` is provided. Prefer `dones` when constructing batches and `batch.terminals` only for forward-facing terminology.

`Batch.merge` performs a shallow list concatenation and returns a new instance. `Batch.from_transition` builds a single‑step batch.

## Seeding

`seed_all(seed, deterministic=True)` seeds Python, NumPy, and Torch (if available) and returns metadata. Call it once at process start for reproducibility.

## Milestones

See `project plan.md` for the broader roadmap and milestone breakdown.

## License

License finalization pending (MIT vs Apache-2.0). Until finalized, contributions default to MIT.
