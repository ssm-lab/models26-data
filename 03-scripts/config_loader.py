"""
Load environments.json and learning.json, build the curriculum artifact.
The curriculum is a list of (EnvSpec, steps) ordered by complexity.
"""
import json
from dataclasses import dataclass


@dataclass
class EnvSpec:
    """Specification for a single environment."""
    env_id: str
    desc: list[str]
    reward_schedule: tuple[float, ...]
    complexity: float


def load_configs(env_config_path: str, learn_config_path: str) -> tuple[dict, dict]:
    """Load and return raw JSON dicts."""
    with open(env_config_path) as f:
        env_cfg = json.load(f)
    with open(learn_config_path) as f:
        learn_cfg = json.load(f)
    return env_cfg, learn_cfg