import pytest
import torch
import torch.nn as nn
import gymnasium as gym
from rlstudio.algorithms import PPO
from rlstudio.envs import RLDataModule
from rlstudio.pipeline import Node, Pipeline


def test_imports():
    from rlstudio.core import Trainer, Experiment
    from rlstudio.io import DataCatalog

    assert True


def test_node_pipeline():
    def add_one(x):
        return x + 1

    node1 = Node(add_one, inputs="x", outputs="intermediate", name="add_one")
    pipeline = Pipeline([node1])

    res = node1.run({"x": 1})
    assert res["intermediate"] == 2


def test_ppo_instantiation():
    obs_space = gym.spaces.Box(low=0, high=1, shape=(4,))
    action_space = gym.spaces.Discrete(2)

    actor = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)

    ppo = PPO(obs_space, action_space, actor, critic)
    assert isinstance(ppo, nn.Module)
