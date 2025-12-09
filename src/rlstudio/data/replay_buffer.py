from torchrl.data import ReplayBuffer as TRLReplayBuffer
from torchrl.data import LazyTensorStorage, ListStorage
from torchrl.data import RandomSampler, PrioritizedSampler
import torch


class ReplayBuffer:
    """
    Wrapper around torchrl.data.ReplayBuffer with simplified API for RLStudio.
    """

    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 32,
        device: str = "cpu",
        prioritized: bool = False,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # Storage Backend
        self.storage = LazyTensorStorage(max_size=capacity, device=torch.device(device))

        # Sampler
        if prioritized:
            self.sampler = PrioritizedSampler(
                max_capacity=capacity, alpha=0.7, beta=0.5
            )
        else:
            self.sampler = RandomSampler()

        self.buffer = TRLReplayBuffer(
            storage=self.storage, sampler=self.sampler, batch_size=batch_size
        )

    def add(self, data: dict):
        """
        Adds a transition or batch of transitions to the buffer.
        """
        self.buffer.extend(data)

    def sample(self) -> dict:
        """
        Samples a batch of transitions.
        """
        return self.buffer.sample()

    def __len__(self):
        return len(self.buffer)
