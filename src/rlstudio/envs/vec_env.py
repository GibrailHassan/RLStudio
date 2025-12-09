import gymnasium as gym
import numpy as np
from typing import List, Callable, Optional, Union, Any
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
        obs, infos = zip(*results)
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
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []

        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(self.num_envs, observation_space, action_space)

    def step(self, actions: Union[np.ndarray, List[Any]]):
        self._step_async(actions)
        return self._step_wait()

    def _step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def _step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(rewards),
            np.stack(terminateds),
            np.stack(truncateds),
            infos,
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                    info["reset_info"] = reset_info
                remote.send((observation, reward, terminated, truncated, info))
            elif cmd == "reset":
                observation, info = env.reset()
                remote.send((observation, info))
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Worker received unknown command: {cmd}")
    except KeyboardInterrupt:
        print("SubprocVecEnv worker: got KeyboardInterrupt")
    finally:
        env.close()
