import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from rlstudio.modules import RLModule


class PPO(RLModule):
    """
    Proximal Policy Optimization (PPO) implementation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        actor_net: nn.Module,
        critic_net: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__(observation_space, action_space)
        self.actor = actor_net
        self.critic = critic_net
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.optimizer = optimizer

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # For training, we might need logits/distribution
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, {"value": value}

    def explore(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, {"log_prob": log_prob, "value": self.critic(obs)}

    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        return torch.argmax(logits, dim=-1)

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Batch unpacking (assumes TensorDict-like structure or dict of tensors)
        obs = batch["obs"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]  # PPO needs old log probs
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        dones = batch["done"]

        # NOTE: In a real PPO, we compute Advantages/Returns BEFORE mini-batch updates.
        # This training_step assumes it receives a batch with advantages already, OR
        # it computes them on the fly if the batch is a full rollout.
        # For MVP simplicity, let's assume 'batch' contains 'advantage' and 'return'.
        # If not, we would need a 'process_rollout' method.
        # To keep 'training_step' simple (optimization only), we assume external processing.

        advantages = batch.get("advantage", torch.zeros_like(rewards))
        returns = batch.get("return", torch.zeros_like(rewards))

        # Current Policy evaluation
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Value evaluation
        values = self.critic(obs).squeeze()

        # Ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Surrogate Losses
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Loss
        value_loss = F.mse_loss(values, returns)

        # Total Loss
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

        # Optimization
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
