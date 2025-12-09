# RLStudio

**RLStudio** is a flexible, scalable, and reproducible Reinforcement Learning framework.

## Philosophies

1. **Python-First**: Integration with the PyData ecosystem.
2. **Layered API**: High-level for ease of use, Mid/Low-level for research flexibility.
3. **Composition over Inheritance**: Agents are composed of independent components.
4. **Reproducibility**: Configuration-driven experiments and MLflow tracking.

## Structure

- `conf/`: Configuration files (hydra params, kedro catalog).
- `src/rlstudio/`: Core library code.
- `notebooks/`: Jupyter notebooks.
- `data/`: Local data storage (ignored by git).
