import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Tuple, Union
from rlstudio.modules import RLModule


class DQN(RLModule):
    """
    Deep Q-Network (DQN) implementation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        q_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 1000,
        target_update_freq: int = 1000,
    ):
        super().__init__(observation_space, action_space)
        self.q_net = q_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target net is frozen

        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq

        self.steps = 0

        # This flag tells Trainer to use Off-Policy loop
        self.is_off_policy = True

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.q_net(obs), {}

    def explore(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            action = torch.tensor(self.action_space.sample())
        else:
            with torch.no_grad():
                q_values = self.q_net(obs)
                action = torch.argmax(q_values, dim=-1)

        # Update epsilon
        self.steps += 1
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

        return action, {"epsilon": self.epsilon}

    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        q_values = self.q_net(obs)
        return torch.argmax(q_values, dim=-1)

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        actions = batch["action"].long()
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        dones = batch["done"]

        # Compute Q(s, a)
        q_values = self.q_net(obs)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute V(s') from Target Net
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_max_q = next_q_values.max(dim=1)[0]
            expected_q = rewards + self.gamma * next_max_q * (1.0 - dones)

        # Loss
        loss = F.mse_loss(q_action, expected_q)

        # Optimization
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Soft update target net? Or hard update?
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss, "epsilon": self.epsilon, "mean_q": q_values.mean()}
