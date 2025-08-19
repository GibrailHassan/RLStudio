from rlstudio.utils.seeding import seed_all


def test_seed_all_returns_dict_and_deterministic_example():
    r1 = seed_all(123)
    r2 = seed_all(123)
    assert r1["seed"] == 123 and r2["seed"] == 123
    assert r1["python_random_example"] == r2["python_random_example"]


def test_seed_all_negative_seed_error():
    import pytest

    with pytest.raises(ValueError):
        seed_all(-5)
