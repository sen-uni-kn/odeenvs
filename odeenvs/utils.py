#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from typing import Callable

import gymnasium as gym


def map_space[T: gym.spaces.Space](f: Callable, space: T, *values):
    if isinstance(space, gym.spaces.Box):
        return f(*values)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(
            map_space(f, sub, *(v[i] for v in values))
            for i, sub in enumerate(space.spaces)
        )
    elif isinstance(space, gym.spaces.Dict):
        return {
            key: map_space(f, sub, *(v[key] for v in values))
            for key, sub in space.spaces.items()
        }
    else:
        flat = (gym.spaces.flatten(space, v) for v in values)
        mapped = f(*flat)
        return gym.spaces.unflatten(space, mapped)
