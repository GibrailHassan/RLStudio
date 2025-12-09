# RLStudio 🤖

**RLStudio** is a production-ready, scalable, and educational Reinforcement Learning framework designed for both novices and researchers. It bridges the gap between simple baselines and complex distributed training systems.

![Dashboard Preview](https://via.placeholder.com/800x400?text=RLStudio+Dashboard+Preview)

## 🌟 Key Features

### 1. **Layered API Design**

* **High-Level API**: `Agent` class for scikit-learn style `fit()` / `predict()` usage. Ideal for quick experiments.
* **Core API**: Composable `Trainer`, `RLModule`, `Pipeline`, and `Node` abstractions for deep research flexibility.

### 2. **Scalability & Distribution**

* **Vectorized Environments**: Built-in `SubprocVecEnv` for single-machine multiprocessing.
* **Distributed Training**: Seamless integration with **Ray** for scaling data collection across clusters (Actor-Learner architecture).

### 3. **Interactive Dashboard**

* **Experiment Launcher**: Configure and launch training jobs directly from the UI.
* **Pipeline Visualizer**: Interactive DAG visualization of your training/inference pipelines (powered by Dash Cytoscape).
* **Experiment Browser**: View MLflow runs and metrics.

### 4. **Production MLOps**

* **MLflow Integration**: Automatic logging of hyperparameters, metrics, and models.
* **ONNX Export**: Standardized model export for optimized inference deployment.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GibrailHassan/RLStudio.git
cd RLStudio

# Install dependencies (including dev tools)
pip install -e ".[dev,ray]"
```

### Training an Agent (High-Level API)

```python
from rlstudio import Agent
from rlstudio.envs import RLDataModule

# 1. Setup Data
datamodule = RLDataModule(env_id="CartPole-v1", num_envs=4, seed=42)

# 2. Define Agent
agent = Agent(algorithm="PPO", env_id="CartPole-v1")

# 3. Train
agent.fit(datamodule, max_epochs=10)

# 4. Predict
obs = datamodule.train_env.reset()[0]
action = agent.predict(obs)
print(f"Action: {action}")
```

### Launching the Dashboard

```bash
python -m rlstudio.dashboard.index
```

Open `http://127.0.0.1:8050` in your browser to access the Experiment Launcher and Visualizers.

---

## 🏗️ Architecture

RLStudio follows a modular design inspired by Kedro and PyTorch Lightning.

```mermaid
graph TD
    User[User / Dashboard] -->|Config| Agent
    Agent -->|Wraps| Trainer
    Trainer -->|Orchestrates| RayActors[Ray Actors (Distributed)]
    Trainer -->|Updates| RLModule[RLModule (Policy/Value)]
    RayActors -->|Collect| Env[Environments]
    RLModule -->|Export| ONNX[ONNX Artifact]
    Trainer -->|Log| MLflow[MLflow Tracking]
```

* **RLModule**: Encapsulates the neural networks (Actor/Critic).
* **Trainer**: Handles the training loop, utilizing `SubprocVecEnv` or `Ray` for data collection.
* **DataModule**: Manages environment creation and preprocessing.

---

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

- `src/rlstudio/`: Main package.
  * `agent.py`: High-level API.
  * `core/`: Core components (`Trainer`, `Experiment`).
  * `envs/`: Environment wrappers (`VecEnv`, `DataModule`).
  * `algorithms/`: Algorithm implementations (`PPO`).
  * `distributed/`: Ray actor implementations.
  * `dashboard/`: Dash application.

---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

**License**: MIT
