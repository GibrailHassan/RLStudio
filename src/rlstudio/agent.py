from typing import Optional, Dict, Any, Union
import gymnasium as gym
from rlstudio.algorithms import PPO
from rlstudio.envs import RLDataModule
from rlstudio.core import Trainer
from rlstudio.data import ReplayBuffer
import torch.nn as nn
import torch


class Agent:
    """
    High-Level API for RLStudio.
    Provides a scikit-learn style interface (fit, predict) for RL algorithms.
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        env_id: str = "CartPole-v1",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.algorithm_name = algorithm
        self.env_id = env_id
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.trainer = None
        self.datamodule = None

    def fit(self, total_timesteps: int = 10000, max_epochs: Optional[int] = None):
        """
        Trains the agent.
        """
        # 1. Setup Environment
        self.datamodule = RLDataModule(env_id=self.env_id)
        self.datamodule.setup()

        # 2. Setup Model (Auto-configure based on Env)
        if self.model is None:
            self.model = self._build_model(self.datamodule.train_env)

        # 3. Setup Trainer
        epochs = (
            max_epochs if max_epochs else 10
        )  # heuristic mapping timesteps -> epochs for MVP
        self.trainer = Trainer(max_epochs=epochs)

        # 4. Train
        # Create a default buffer if needed
        buffer = ReplayBuffer(capacity=1000)
        self.trainer.fit(self.model, self.datamodule, buffer)

    def predict(self, obs: Union[torch.Tensor, Any], deterministic: bool = True) -> Any:
        """
        Predicts action for a given observation.
        """
        if self.model is None:
            raise RuntimeError("Agent must be fit before calling predict.")

        # Ensure tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        if deterministic:
            return self.model.forward_inference(obs)
        else:
            action, _ = self.model.explore(obs)
            return action

    def _build_model(self, env) -> nn.Module:
        """
        Factory to build the RLModule based on algorithm name and env specs.
        """
        if self.algorithm_name == "PPO":
            # Heuristic network creation
            # Real implementation would use properly sized heads based on observation_space
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n  # Assuming discrete for MVP CartPole

            actor = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, action_dim)
            )
            critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))
            return PPO(env.observation_space, env.action_space, actor, critic)
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm_name} not supported.")
