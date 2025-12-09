import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union


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

    @abstractmethod
    def training_step(self, batch: Any) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Performs a single training step on a batch of data.
        Returns a dictionary containing losses and metrics.
        """
        pass

    def to_onnx(self, file_path: str, input_sample: torch.Tensor):
        """
        Exports the actor network to ONNX format.
        """
        if self.actor is None:
            raise ValueError("Actor network is not defined.")

        # Ensure model is in eval mode
        self.actor.eval()

        torch.onnx.export(
            self.actor,
            input_sample,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
