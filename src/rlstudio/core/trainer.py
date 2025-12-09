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
        """
        Evaluation loop.
        """
        pass
