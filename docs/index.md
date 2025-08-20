# RLStudio Documentation

Welcome to the RLStudio docs.

## Overview

RLStudio is an early-stage reinforcement learning framework focused on:

- Clear, typed core interfaces
- Reproducibility (deterministic seeding)
- Minimal, test-driven incremental features

## Getting Started

Install in editable mode with dev extras:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

## Core API

```{autosummary}
:toctree: api
:recursive:

rlstudio.Batch
rlstudio.ReplayBufferSimple
rlstudio.seed_all
```

## Batch Notes

See README for detailed notes on the `Batch` container (`dones` vs `terminals`).

## Roadmap

High level milestones are tracked in `project plan.md`.

## License

Pending final selection (MIT vs Apache-2.0).
