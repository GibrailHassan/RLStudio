# RLStudio Documentation

Welcome to the RLStudio docs.

## Overview

Focus: typed interfaces, reproducibility, incremental expansion.

## Getting Started

```bash
pip install -e .[dev]
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
