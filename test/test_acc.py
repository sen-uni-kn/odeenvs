#  Copyright (c) 2024.
#  Licensed under the MIT license
from time import sleep

import numpy as np
import pytest

from odeenvs import ACCEnv


@pytest.mark.parametrize(
    "render_mode", [None, pytest.param("human", marks=pytest.mark.render)]
)
def test_simulate(render_mode):
    env = ACCEnv(time_steps=100, render_mode=render_mode)
    # rng = np.random.default_rng(32051360)
    observation, _ = env.reset(seed=7739521)

    for i in range(200):
        action = 2
        observation, reward, d_cost, v_cost, terminated, truncated, _ = env.step(
            np.array([action], dtype=np.float32)
        )

        if render_mode is None:
            print(f"{i=}, {action=}, {observation=}, {reward=}, {d_cost=}, {v_cost=}")

        if terminated or truncated:
            observation, _ = env.reset()

    env.close()


@pytest.mark.render
def test_render():
    env = ACCEnv(time_steps=100, render_mode="human")
    env.reset()
    env.render()
    sleep(10)
