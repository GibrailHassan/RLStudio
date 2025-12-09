import functools
import gymnasium as gym
from typing import Optional, Callable, List
from .vec_env import DummyVecEnv


def _create_env(env_id: str, seed: int, rank: int):
    """
    Top-level helper to create environment, ensuring pickalability.
    """
    env = gym.make(env_id)
    env.reset(seed=seed + rank)
    return env


class RLDataModule:
    """
    Manages the environment setup (train/val/test) and data processing.
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        seed: int = 42,
        val_env_id: Optional[str] = None,
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.seed = seed
        self.val_env_id = val_env_id or env_id
        self.train_env = None
        self.val_env = None

    def setup(self):
        """
        Initializes the environments.
        """
        self.train_env = self._make_vec_env(self.env_id, self.num_envs, "train")
        self.val_env = self._make_vec_env(
            self.val_env_id, 1, "val"
        )  # Val usually 1 env for deterministic eval

    def _make_vec_env(self, env_id: str, num_envs: int, mode: str):
        # Use functools.partial to create a no-arg callable that is picklable
        env_fns = [
            functools.partial(_create_env, env_id, self.seed, i)
            for i in range(num_envs)
        ]

        if mode == "train" and num_envs > 1:
            from .vec_env import SubprocVecEnv

            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)

    def train_dataloader(self):
        return self.train_env

    def val_dataloader(self):
        return self.val_env
