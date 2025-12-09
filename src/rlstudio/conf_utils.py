import yaml
import os
from typing import Dict, Any


def load_defaults(env_id: str) -> Dict[str, Any]:
    """
    Loads default hyperparameters for a given environment ID.
    For MVP, returns hardcoded dictionary or loads from simple YAML.
    """
    # Placeholder implementation
    defaults = {
        "CartPole-v1": {
            "algorithm": "PPO",
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "batch_size": 64,
        }
    }
    return defaults.get(env_id, {})
