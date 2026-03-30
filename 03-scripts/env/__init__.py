from gymnasium.envs.registration import register
from .fire import BurningForest

register(
    id="BurningForest-v0",
    entry_point = BurningForest,
    max_episode_steps=200
)