from rlstudio.core.batch import Batch
from rlstudio.data.replay_buffer import ReplayBufferSimple


def make_batch(x: int) -> Batch:
    return Batch([x], [x % 2], [float(x)], [False], [x + 1], infos=[{"i": x}])


def test_add_and_len():
    rb = ReplayBufferSimple(capacity=3)
    rb.add(make_batch(1))
    rb.add(make_batch(2))
    assert len(rb) == 2
    rb.add(make_batch(3))
    rb.add(make_batch(4))  # should evict oldest (1)
    assert len(rb) == 3


def test_sample_merge():
    rb = ReplayBufferSimple(capacity=5)
    for i in range(5):
        rb.add(make_batch(i))
    out = rb.sample(3)
    assert out.size() == 3
    assert out.rewards[-1] == float(4)


def test_sample_single():
    rb = ReplayBufferSimple(capacity=2)
    rb.add(make_batch(10))
    last = rb.sample(1)
    assert last.rewards[0] == 10.0
