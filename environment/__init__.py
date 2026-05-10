"""
Registro del entorno BasketballLineup-v0 en Gymnasium.
"""

import gymnasium as gym
from gymnasium.envs.registration import register


ENV_ID = "BasketballLineup-v0"


def register_basketball_env():
    if ENV_ID not in gym.envs.registry:
        register(
            id=ENV_ID,
            entry_point="environment.basketball_env:BasketballEnv",
            kwargs={
                "obs_mode": "continuous",
                "force_synthetic": False,
            },
        )