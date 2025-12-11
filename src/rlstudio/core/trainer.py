import torch
import mlflow
from typing import Optional, Any
from rlstudio.modules import RLModule
from rlstudio.envs import RLDataModule
from rlstudio.data import ReplayBuffer
import os
import numpy as np


class Trainer:
    """
    Orchestrates the training loop (Collection -> Optimization -> Logging).
    """

    def __init__(
        self,
        max_epochs: int = 100,
        train_batch_size: int = 2048,
        minibatch_size: int = 64,
        num_workers: int = 0,
        distributed: bool = False,
        logger: Optional[Any] = None,
    ):
        self.max_epochs = max_epochs
        self.train_batch_size = train_batch_size
        self.minibatch_size = minibatch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.logger = logger
        self.actors = []

    def fit(
        self, model: RLModule, datamodule: Optional[RLDataModule], buffer: ReplayBuffer
    ):
        """
        Main training loop.
        """
        # Initialize Ray if needed
        if self.distributed:
            import ray
            from rlstudio.distributed.actors import RayActor

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            env_id = datamodule.env_id
            seed = datamodule.seed

            self.actors = [
                RayActor.remote(env_id, seed, i) for i in range(self.num_workers)
            ]

            # Init models on actors
            ray.get(
                [
                    actor.set_model.remote(
                        model.__class__,
                        model.observation_space,
                        model.action_space,
                        model.actor,
                        model.critic,
                    )
                    for actor in self.actors
                ]
            )

        if self.logger:
            self.logger.log_params(
                {"max_epochs": self.max_epochs, "batch_size": self.train_batch_size}
            )

        for epoch in range(self.max_epochs):
            # 1. Collect Data
            if self.distributed:
                # Sync weights
                weights = model.state_dict()
                weights_cpu = {k: v.cpu() for k, v in weights.items()}

                [actor.update_weights.remote(weights_cpu) for actor in self.actors]

                # Collect
                steps_per_worker = self.train_batch_size // self.num_workers
                futures = [
                    actor.collect_rollouts.remote(steps_per_worker)
                    for actor in self.actors
                ]
                results = ray.get(futures)

                # Add to buffer
                # TODO: Implement batch add
            else:
                # LOCAL COLLECTION

                if getattr(model, "is_off_policy", False):
                    # --- OFF-POLICY LOOP (DQN, SAC) ---
                    # 1. Collect Steps & Store in Buffer
                    # We continue from where we left off ideally, but for MVP we reset or keep simple state
                    obs, _ = datamodule.train_env.reset()
                    obs = obs[0]  # Unpack VecEnv result for single-env logic

                    for _ in range(self.train_batch_size):
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                        with torch.no_grad():
                            action, info = model.explore(obs_t)

                        # Ensure action is a list for VecEnv
                        # Support single env locally for now
                        act_val = (
                            action.item()
                            if isinstance(action, torch.Tensor)
                            else action
                        )

                        next_obs_arr, reward_arr, terminated_arr, truncated_arr, _ = (
                            datamodule.train_env.step([act_val])
                        )
                        # Unpack single env results
                        next_obs = next_obs_arr[0]
                        reward = reward_arr[0]
                        terminated = terminated_arr[0]
                        truncated = truncated_arr[0]
                        done = terminated or truncated

                        # Add to ReplayBuffer
                        # Assuming action is tensor, we might need item()
                        act_val = (
                            action.item()
                            if isinstance(action, torch.Tensor)
                            else action
                        )
                        buffer.add(
                            {
                                "obs": torch.tensor(obs, dtype=torch.float32),
                                "action": torch.tensor(act_val, dtype=torch.float32),
                                "reward": torch.tensor(reward, dtype=torch.float32),
                                "next_obs": torch.tensor(next_obs, dtype=torch.float32),
                                "done": torch.tensor(done, dtype=torch.float32),
                            }
                        )

                        obs = next_obs
                        if done:
                            obs, _ = datamodule.train_env.reset()
                            obs = obs[0]  # Unpack

                    # 2. Train on Buffer
                    # Verify we have enough data
                    if len(buffer) >= self.minibatch_size:
                        num_updates = self.train_batch_size // self.minibatch_size
                        for _ in range(num_updates):
                            # Sample and convert to Tensor
                            s_batch = buffer.sample(self.minibatch_size)
                            t_batch = {
                                "obs": torch.tensor(
                                    s_batch["obs"], dtype=torch.float32
                                ),
                                "action": torch.tensor(
                                    s_batch["action"], dtype=torch.float32
                                ),
                                "reward": torch.tensor(
                                    s_batch["reward"], dtype=torch.float32
                                ),
                                "next_obs": torch.tensor(
                                    s_batch["next_obs"], dtype=torch.float32
                                ),
                                "done": torch.tensor(
                                    s_batch["done"], dtype=torch.float32
                                ),
                            }

                            metrics = model.training_step(t_batch)

                            if self.logger:
                                self.logger.log_metrics(
                                    {
                                        k: (
                                            v.item()
                                            if isinstance(v, torch.Tensor)
                                            else v
                                        )
                                        for k, v in metrics.items()
                                    },
                                    step=epoch,
                                )

                else:
                    # --- ON-POLICY LOOP (PPO) ---
                    # 1. Collect Rollout
                    observations = []
                    actions = []
                    rewards = []
                    log_probs = []
                    dones = []
                    values = []

                    obs, _ = datamodule.train_env.reset()
                    obs = obs[0]  # Unpack VecEnv result

                    for _ in range(self.train_batch_size):
                        # Convert to tensor
                        obs_tensor = torch.tensor(obs, dtype=torch.float32)

                        # Explore
                        with torch.no_grad():
                            action, info = model.explore(obs_tensor)
                            value = info.get("value", torch.tensor(0.0))
                            log_prob = info.get("log_prob", torch.tensor(0.0))

                        # Step
                        # Ensure action is a list for VecEnv
                        act_val = (
                            action.item()
                            if isinstance(action, torch.Tensor)
                            else action
                        )

                        next_obs_arr, reward_arr, terminated_arr, truncated_arr, _ = (
                            datamodule.train_env.step([act_val])
                        )
                        # Unpack
                        next_obs = next_obs_arr[0]
                        reward = reward_arr[0]
                        terminated = terminated_arr[0]
                        truncated = truncated_arr[0]
                        done = terminated or truncated

                        observations.append(obs_tensor)
                        actions.append(action)
                        rewards.append(torch.tensor(reward, dtype=torch.float32))
                        log_probs.append(log_prob)
                        dones.append(torch.tensor(done, dtype=torch.float32))
                        values.append(value)

                        obs = next_obs
                        if done:
                            obs, _ = datamodule.train_env.reset()
                            obs = obs[0]  # Unpack

                    # Stack
                    batch = {
                        "obs": torch.stack(observations),
                        "action": torch.stack(actions),
                        "reward": torch.stack(rewards),
                        "log_prob": torch.stack(log_probs),
                        "done": torch.stack(dones),
                        "value": torch.stack(values),  # Needed for GAE
                    }

                    # Compute GAE (On-Policy only)
                    # Ideally this should be modular, but putting here for MVP
                    next_val = 0  # Approximation for last step
                    batch = self._compute_gae(
                        batch, next_val, model.gamma, model.gae_lambda
                    )

                    # 2. Train on Batch
                    dataset_size = batch["obs"].shape[0]
                    indices = np.arange(dataset_size)

                    for _ in range(
                        self.num_workers if self.distributed else 4
                    ):  # Epochs per rollout
                        np.random.shuffle(indices)
                        for start in range(0, dataset_size, self.minibatch_size):
                            end = start + self.minibatch_size
                            mb_idx = indices[start:end]

                            mini_batch = {k: v[mb_idx] for k, v in batch.items()}

                            # Optimization Step
                            metrics = model.training_step(mini_batch)

                            # Log metrics
                            if self.logger:
                                self.logger.log_metrics(
                                    {
                                        k: (
                                            v.item()
                                            if isinstance(v, torch.Tensor)
                                            else v
                                        )
                                        for k, v in metrics.items()
                                    },
                                    step=epoch,
                                )

        # End of training MLOps
        if self.logger:
            # 1. Log PyTorch Model
            try:
                mlflow.pytorch.log_model(model, "model")
            except Exception as e:
                print(f"MLflow model logging failed (normal if no run active): {e}")

            # 2. Log ONNX Artifact
            # Construct dummy input based on observation space
            # Assuming Box space for simplicity
            try:
                if hasattr(model.observation_space, "shape"):
                    # Create dummy input of correct shape
                    shape = (1,) + model.observation_space.shape
                    dummy_input = torch.randn(*shape)
                else:
                    dummy_input = torch.randn(1, 4)  # Fallback

                onnx_path = "model.onnx"
                model.to_onnx(onnx_path, dummy_input)
                mlflow.log_artifact(onnx_path)
                print(f"Logged ONNX model to MLflow.")
                os.remove(onnx_path)
            except Exception as e:
                print(f"Failed to export/log ONNX model: {e}")

    def _compute_gae(self, batch, next_value, gamma, gae_lambda):
        rewards = batch["reward"].view(-1)
        values = batch["value"].view(-1)
        dones = batch["done"].view(-1)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        # Iterate backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        batch["advantage"] = advantages
        batch["return"] = returns
        return batch
