#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from typing import Callable

import gymnasium as gym


def map_space[T: gym.spaces.Space](f: Callable, space: T, *values):
    flat = (gym.spaces.flatten(space, v) for v in values)
    mapped = f(*flat)
    return gym.spaces.unflatten(space, mapped)
