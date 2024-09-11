#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from typing import Callable, TypeVar

import gymnasium as gym


def map_space(f: Callable, space: gym.Space, *values):
    flat = (gym.spaces.flatten(space, v) for v in values)
    mapped = f(*flat)
    return gym.spaces.unflatten(space, mapped)
