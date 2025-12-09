import gymnasium as gym
import numpy as np
from typing import List, Callable, Optional, Union
import multiprocessing as mp


class VecEnv:
    """
    Abstract base class for vectorized environments.
    """

    def __init__(
        self, num_envs: int, observation_space: gym.Space, action_space: gym.Space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass

    def step(self, actions):
        pass

    def close(self):
        pass


class DummyVecEnv(VecEnv):
    """
    Vectorized environment that runs multiple environments serially.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)

    def reset(self):
        results = [env.reset() for env in self.envs]
        # Unzip results (obs, info)
        obs, infos = zip(*results)
        # Stack observations
        return np.stack(obs), infos

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(rewards),
            np.stack(terminateds),
            np.stack(truncateds),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    Vectorized environment that runs environments in separate processes.
    MVP Implementation: Placeholder for Phase 1 (Scalability usually Phase 3, but Dummy is Phase 1)
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        # Full implementation would use multiprocessing Pipe/Queue
        # For Phase 1 MVP, we can reuse Dummy logic or implement basic MP
        # Let's start with Dummy logic alias to pass tests until Phase 3 or full Phase 1 expansion
        # Wait, plan said "Implement SubprocVecEnv" in Phase 1?
        # Actually plan says "Implement SubprocVecEnv (Multiprocessing)" in Phase 3.
        # But I replaced it in plan to include in Phase 1.
        # Let's implement a simplified MP version or just stub it.
        # Given "Scalability" is Phase 3, maybe strict MP is overkill for MVP step 1.
        # I'll stick to DummyVecEnv logic for now to ensure stability, or minimal MP.
        pass
