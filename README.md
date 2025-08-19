# RLStudio

Early-stage reinforcement learning framework.

## Status

Milestone 1a: Core Contracts & Reproducibility (in planning).

## Vision

Flexible, scalable, reproducible RL with layered APIs and strong testing discipline.

## Quick Start (Placeholder)

```bash
pip install -e .
rlstudio run configs/ppo/cartpole.yaml
```

### Dev Environment (uv optional)

You can use [uv](https://github.com/astral-sh/uv) for faster installs:

```bash
uv venv && source .venv/bin/activate  # or on Windows: .venv\\Scripts\\activate
uv pip install -e .[dev]
```

Tooling stack: Ruff (lint + format + import sorting), mypy (types), pytest.

## Milestones

See `../RLStudio Project Plan.md` for full roadmap.

## License

To be finalized (MIT or Apache-2.0 planned).

