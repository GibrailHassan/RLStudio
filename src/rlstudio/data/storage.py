import numpy as np
import os
from typing import Dict, Any


class LazyMemmapStorage:
    """
    Disk-based storage for large ReplayBuffers using memory mapping.
    Avoids loading the entire dataset into RAM.
    """

    def __init__(
        self,
        max_size: int,
        obs_shape: tuple,
        obs_dtype=np.float32,
        root_dir: str = "./buffer_data",
    ):
        self.max_size = max_size
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

        self.obs_shape = obs_shape
        self.ptr = 0
        self.size = 0

        # Initialize memmap files
        self.obs = np.memmap(
            os.path.join(root_dir, "obs.dat"),
            dtype=obs_dtype,
            mode="w+",
            shape=(max_size,) + obs_shape,
        )
        self.next_obs = np.memmap(
            os.path.join(root_dir, "next_obs.dat"),
            dtype=obs_dtype,
            mode="w+",
            shape=(max_size,) + obs_shape,
        )
        self.actions = np.memmap(
            os.path.join(root_dir, "actions.dat"),
            dtype=np.float32,  # Assuming continuous or simple
            mode="w+",
            shape=(max_size, 1),  # Support simple shape for now
        )
        self.rewards = np.memmap(
            os.path.join(root_dir, "rewards.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(max_size, 1),
        )
        self.dones = np.memmap(
            os.path.join(root_dir, "dones.dat"),
            dtype=np.float32,
            mode="w+",
            shape=(max_size, 1),
        )

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # Flush occasionally? Memmap handles OS paging.

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": self.obs[idxs],
            "action": self.actions[idxs],
            "reward": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "done": self.dones[idxs],
        }

    def close(self):
        # Delete files if temporary? Or keep for persistence?
        # For now, we leave them.
        pass
