import torch
import mlflow
from typing import Optional, Any
from rlstudio.modules import RLModule
from rlstudio.envs import RLDataModule
from rlstudio.data import ReplayBuffer


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
                pass  # Local collection

            # 2. Train (Placeholder)
            pass
