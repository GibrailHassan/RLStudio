from rlstudio.core.interfaces import RLModule, ReplayBuffer, Trainer
from rlstudio.core.batch import Batch


class DummyModule:
    def act(self, batch: Batch) -> Batch:
        return batch

    def value(self, batch: Batch) -> Batch:
        return batch


class DummyReplay:
    def __init__(self):
        self._batches = []

    def add(self, batch: Batch) -> None:
        self._batches.append(batch)

    def sample(self, batch_size: int) -> Batch:
        return self._batches[0]

    def __len__(self) -> int:
        return len(self._batches)


class DummyTrainer:
    def train_step(self):
        return {"loss": 0.0}

    def eval_step(self):
        return {"reward": 1.0}


def test_protocol_runtime_checks():
    dm = DummyModule()
    dr = DummyReplay()
    dt = DummyTrainer()
    assert isinstance(dm, RLModule)
    assert isinstance(dr, ReplayBuffer)
    assert isinstance(dt, Trainer)
