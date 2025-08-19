# RLStudio

Early-stage reinforcement learning framework (internal educational / experimental project).

## Status

Milestone 1a: Core Contracts & Reproducibility (in planning).

## Vision

Flexible, scalable, reproducible RL with layered APIs and strong testing discipline.

## Quick Start (Placeholder)

```bash
pip install -e .
rlstudio run configs/ppo/cartpole.yaml
```

## Milestones

Canonical sequence: 1a → 1b → 2 → 3 → 4 (GitHub internally numbers them 1..5 respectively). The table below shows both human-friendly names and the GitHub milestone links.

| Milestone | GitHub # | Target (tentative) | Summary |
|-----------|----------|--------------------|---------|
| 1a: Core Contracts & Reproducibility | [Milestone 1](https://github.com/GibrailHassan/RLStudio/milestone/1) | 2025-09-20 | Core abstractions (RLModule, Trainer, ReplayBuffer), spec & batch schema, seeding, checkpoint v1, metrics & logging baseline, CI + typing + pre-commit, governance + docs seeds, KPI definitions, performance & KPI placeholder scripts. |
| 1b: Expanded Evaluation & Docs Hardening | [Milestone 2](https://github.com/GibrailHassan/RLStudio/milestone/2) | 2025-10-05 | Flesh out PPO loop, richer evaluation protocol, coverage & timing surfaced in CI, checkpoint versioning ADR, learning regression baseline artifacts, seed drift detection, persistent JSONL metrics sink. |
| 2: Core Algorithm Implementations & Persistence | [Milestone 3](https://github.com/GibrailHassan/RLStudio/milestone/3) | 2025-11-01 | Full PPO + second algorithm skeleton, checkpoint version bump & backward test, JSONL rotation & plotting, reproducibility variance report, expanded property tests. |
| 3: Scaling & Experiment Management | [Milestone 4](https://github.com/GibrailHassan/RLStudio/milestone/4) | (TBD) | Parallel/vector envs, sweep runner, profiling + optimization, ReplayBuffer optimization, GPU path ADR, cross-run aggregation. |
| 4: Productionization & Extensibility | [Milestone 5](https://github.com/GibrailHassan/RLStudio/milestone/5) | 2025-12-01 | Plugin registry maturity, migration tooling, reproducibility audit, full API docs, tutorial, security/integrity ADR, release packaging (v0.1.0). |

Tracking issues (umbrella / narrative): #1 (1a), #30 (1b), #31 (2), #32 (3), #33 (4). Milestone internal IDs: 1→1a, 2→1b, 3→2, 4→3, 5→4.

Status tallies and detailed acceptance criteria live in those tracking issues. A future `MILESTONES.md` may consolidate snapshots.

## Usage / Rights

Internal educational prototype. Not open-source; all rights reserved. See `NOTICE.md` for permitted internal use and restrictions.
