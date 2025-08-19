# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Initial repository scaffolding: README, LICENSE placeholder, gitignore, CI workflow, issue & PR templates.
- Core protocols: RLModule, ReplayBuffer, Trainer.
- Batch container with merge, validation, from_transition helper.
- ReplayBufferSimple (deterministic recent sampling) with tests.
- Seeding utility covering Python, NumPy, Torch.
- Ruff-only tooling (replaces Black/Flake8/isort) + strict mypy.
