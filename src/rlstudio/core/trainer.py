import torch
import mlflow
from typing import Optional
from rlstudio.modules import RLModule
from rlstudio.envs import RLDataModule
from rlstudio.data import ReplayBuffer


class Trainer:
    """
    Orchestrates the training loop (Collection -> Optimization -> Logging).
    """

    def __init__(
        self,
        max_epochs: int = 10,
        limit_train_batches: int = 100,
        logger: bool = True,
        default_root_dir: str = None,
    ):
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.logger = logger
        self.default_root_dir = default_root_dir

    def fit(
        self,
        model: RLModule,
        datamodule: RLDataModule,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        """
        Main training loop.
        """
        print("Setting up datamodule...")
        datamodule.setup()
        train_env = datamodule.train_dataloader()

        # Initial Logging
        if self.logger:
            mlflow.log_param("max_epochs", self.max_epochs)
            # Log model architecture?

        print(f"Starting training for {self.max_epochs} epochs...")

        for epoch in range(self.max_epochs):
            # 1. Collection Phase (Rollouts)
            # In a real loop, we'd interact with the env using the model's distinct 'explore' policy
            # For pure MVP reference, let's assume we do some steps

            # obs = train_env.reset()
            # ... interaction loop ...
            # buffer.add(transitions)

            # 2. Optimization Phase
            # batch = replay_buffer.sample()
            # loss = model.training_step(batch)
            # optimizer.step()

            # 3. Logging
            if self.logger:
                mlflow.log_metric("epoch_reward", 0.0, step=epoch)  # Placeholder

            print(f"Epoch {epoch} complete.")

        print("Training complete.")

    def test(self, model: RLModule, datamodule: RLDataModule):
        """
        Evaluation loop.
        """
        pass
