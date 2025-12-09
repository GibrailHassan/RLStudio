import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class RLModule(nn.Module, ABC):
    """
    Base class for Reinforcement Learning modules (Agents).
    Encapsulates the policy, value function, and other learnable components.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Default forward pass (usually for training).
        Returns: (output, extra_info)
        """
        pass

    @abstractmethod
    def explore(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Action selection for exploration (e.g., sampling from distribution).
        Returns: (action, log_prob_or_info)
        """
        pass

    @abstractmethod
    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection for inference/testing.
        Returns: action
        """
        pass
