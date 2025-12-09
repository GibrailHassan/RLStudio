import ray
import gymnasium as gym
import torch
import numpy as np
from typing import Dict, Any, List


@ray.remote
class RayActor:
    """
    Ray Actor for collecting experience in parallel.
    """

    def __init__(self, env_id: str, seed: int, rank: int):
        self.env = gym.make(env_id)
        self.seed = seed
        self.rank = rank
        # Reset immediately
        self.env.reset(seed=seed + rank)
        self.local_model = None

    def set_model(self, model_class, obs_space, action_space, actor_arch, critic_arch):
        """
        Initialize the local copy of the model.
        """
        # This is a bit hacky for MVP. Ideally we pass a model factory.
        # Assuming PPO logic for now or generic RLModule if we can serialize creation logic.
        # For simplicity, we might just pass the class and args.
        # But we need to imports inside if they aren't available globally?
        # Assuming model_class is picklable (it is).
        self.local_model = model_class(obs_space, action_space, actor_arch, critic_arch)

    def update_weights(self, weights: Dict[str, Any]):
        """
        Update local model weights from Learner.
        """
        if self.local_model:
            self.local_model.load_state_dict(weights)

    def collect_rollouts(self, steps: int) -> List[Dict[str, Any]]:
        """
        Run the environment and collect data.
        """
        rollouts = []
        obs, _ = (
            self.env.reset()
        )  # Start fresh or enable continue? MVP: Start fresh or keep state?
        # Standard RL keeps state.
        # But for simpler MVP, let's just reset if needed or keep self.obs
        if not hasattr(self, "obs"):
            self.obs, _ = self.env.reset(seed=self.seed + self.rank)

        for _ in range(steps):
            # Convert obs to tensor
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32)

            # Select action
            if self.local_model:
                with torch.no_grad():
                    action, log_prob = self.local_model.explore(obs_tensor)
                    action_val = action.item()  # Assuming scalar action for CartPole
            else:
                action_val = self.env.action_space.sample()

            next_obs, reward, terminated, truncated, info = self.env.step(action_val)

            # Store data
            rollouts.append(
                {
                    "obs": self.obs,
                    "action": action_val,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": terminated or truncated,
                }
            )

            data_obs = next_obs

            if terminated or truncated:
                next_obs, _ = self.env.reset()

            self.obs = next_obs

        return rollouts
