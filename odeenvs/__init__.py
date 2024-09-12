"""
ODEEnvs - ODE-based reinforcement learning environments.
"""

__version__ = "0.0.1"

from .env import ODEEnv
from .acc import ACCEnv


import gymnasium as gym

gym.register(
    "odeenvs:ACC-v0",
    entry_point="odeenvs:ACCEnv",
)
