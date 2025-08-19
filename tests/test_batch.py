from rlstudio.core.batch import Batch


def test_batch_len_and_as_dict():
    b = Batch(
        observations=[1, 2, 3],
        actions=[0, 0, 1],
        rewards=[1.0, 0.0, -1.0],
        dones=[False, False, True],
        next_observations=[2, 3, None],
        infos=[{"ep": 0}, {"ep": 0}, {"ep": 0}],
    )
    assert len(b) == 3
    d = b.as_dict()
    assert d["actions"][2] == 1


def test_batch_merge_lists():
    b1 = Batch([1], [0], [1.0], [False], [2], infos=[{"i": 1}])
    b2 = Batch([3], [1], [0.5], [True], [None], infos=[{"i": 2}])
    merged = Batch.merge([b1, b2])
    assert len(merged.observations) == 2
    assert merged.rewards == [1.0, 0.5]
    assert isinstance(merged.infos, list) and len(merged.infos) == 2
