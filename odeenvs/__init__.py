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
gym.register(
    "odeenvs:ACCSingleInitialState-v0",
    entry_point="odeenvs:ACCEnv",
    kwargs=dict(
        time_steps=500,
        mu=0.0,
        switch_frequency=500,
        x0_lead=60.0,
        v0_lead=30.0,
        x0_ego=0.0,
        v0_ego=20.0,
    ),
)
