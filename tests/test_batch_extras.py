import pytest

from rlstudio.core.batch import Batch


def test_merge_single():
    b = Batch([1], [0], [1.0], [False], [2])
    out = Batch.merge([b])
    assert out.observations == [1]
    assert out is not b  # copy


def test_merge_empty_raises():
    with pytest.raises(ValueError):
        Batch.merge([])


def test_from_transition():
    b = Batch.from_transition(1, 2, 3.0, False, 4, info={"x": 5})
    assert b.rewards == [3.0]
    assert b.infos[0]["x"] == 5
