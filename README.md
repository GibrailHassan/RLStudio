# 🧠 RLStudio: Enterprise MLOps for Reinforcement Learning

> **A professional, distributed, and production-ready Reinforcement Learning ecosystem.**

RLStudio is a comprehensive MLOps platform designed to bridge the gap between academic RL research and enterprise production. It provides a unified interface for designing, training, and deploying RL agents, backed by a powerful distributed computing backend (Ray) and enterprise-grade observability (MLflow + Dash).

---

## 🚀 Key Features

### 1. **Enterprise Dashboard**

A stunning, dark-themed Glassmorphism UI for managing the full lifecycle of RL experiments.

- **Glass Card UI**: Modern, responsive, and visually immersive.
- **Live Observability**: Real-time training metrics (Loss, Reward, Entropy) with interactive Plotly charts.
- **Run Management**: Sort, filter, and compare experiments with ease.
- **Experiment Launcher**: Launch new training jobs directly from the UI with custom hyperparameters.

### 2. **Distributed Training Engine**

Scalable architecture built on **Ray** for high-throughput experience collection.

- **SubprocVecEnv**: Parallelize environment steps across CPU cores.
- **Distributed Sampling**: Decoupled "Actors" for collecting data and "Learners" for updating policies (PPO, DQN).
- **LazyMemmapStorage**: Efficiently handle massive replay buffers that exceed RAM.

### 3. **Modular RL Pipeline**

A strictly typed, component-based architecture inspired by modern MLOps standards.

- **RLModule**: PyTorch-based policy/value networks.
- **DataModule**: Standardized interface for environments (`Gymnasium` compatible).
- **Core Abstractions**: `Trainer`, `Agent`, `Experiment` for reproducible workflows.

### 4. **Production Ready**

- **MLflow Integration**: Automatic logging of hyperparameters, metrics, and models.
- **ONNX Export**: Seamless export of trained agents for high-performance inference.
- **Strict Typing**: Full Python type hinting for reliability.

---

## 🛠️ Architecture

```mermaid
graph TD
    Dashboard[🖥️ Enterprise Dashboard] -->|Launch| Experiment
    Experiment -->|Config| Trainer
    Trainer -->|Update| Agent[🤖 Agent (PPO/DQN)]
    Trainer -->|Sync| RayActors[⚡ Ray Actors]
    RayActors -->|Sample| Envs[Gymnasium Envs]
    RayActors -->|Push Data| Buffer[ReplayBuffer]
    Trainer -->|Pull Data| Buffer
    Trainer -->|Log| MLflow[(MLflow Database)]
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- Windows / Linux / MacOS

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rlstudio.git
cd rlstudio

# Install dependencies (using uv is recommended for speed)
pip install -e .
```

### 1. **Launch the Dashboard**

The command center for your RL operations.

```bash
python -m rlstudio.dashboard.index
```

> Access at `http://127.0.0.1:8050`

### 2. **Run an Experiment via CLI**

```bash
python run_experiment.py --algo PPO --env CartPole-v1 --epochs 50 --workers 4
```

### 3. **Use the Python API**

```python
from rlstudio.agent import Agent

# Train an agent
agent = Agent(algorithm="PPO", env_id="LunarLander-v3")
agent.fit(epochs=100)

# Evaluate
avg_reward = agent.evaluate(episodes=10)
print(f"Average Reward: {avg_reward}")
```

---

## 📚 Project Structure

```
rlstudio/
├── conf/               # Hydra configuration files
├── notebooks/          # Jupyter notebooks for analysis
├── src/
│   └── rlstudio/
│       ├── algorithms/ # PPO, DQN implementations
│       ├── core/       # Trainer, Experiment, Callbacks
│       ├── dashboard/  # Dash Enterprise UI
│       ├── data/       # Replay Buffers, Storage
│       ├── distributed/# Ray Actor implementations
│       ├── envs/       # Environment wrappers
│       ├── modules/    # Neural Network definitions
│       └── pipeline/   # Pipeline abstractions
├── tests/              # Unit and integration tests
└── run_experiment.py   # CLI Entry point
```

---

## 🤝 Contributing

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch.
5. Open a Pull Request.

---

**Built with ❤️ by RLStudio Team**
