#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from abc import ABC, abstractmethod
from typing import Any, Final, Literal, override

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.core import RenderFrame
import pygame

from .utils import map_space


class ODEEnv[S, O, A, IO](gym.Env[O, A], ABC):
    """The ODE Environment base class.

    Solves a ODE of the form
    ```
    x'(t) = f(x(t), u(t), t)
    ```
    where `x(t)` is the *state*, `u(t)` is the *action* from the agent, the
    input to the ODE and `t` is the *time*.

    The step method of this environment returns a safety cost
    as an additional value after the reward.

    To implement, you need to provide
     - action space, observation space, state space,
       and initial state options space for `ODEEnvs` initializer.
       These spaces should not contain batch dimensions.
     - an `_initial_state` method to provides `x(0)`.
     - an `_derivative` method that computes `x'(t)`.
     - an `_reward` method that computes the reward for a state, action pair.

    Additionally, you can implement
     - `_cost()` to provide a safety specification cost (robustness value).
     - `_obs()` to construct observations from the state.
     - `_info()` to provide additional information that is returned by `step`.
     - `_draw()` for rendering the current state.

    Generics:
     - `S`: The state space type.
     - `O`: The observation space type.
     - `A`: The action space type.
     - `IO`: The initial state options type.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        state_space: gym.Space[S],
        observation_space: gym.Space[O],
        action_space: gym.Space[A],
        initial_state_options_space: gym.Space[IO],
        time_steps: int,
        step_size: float,
        n_costs: int = 1,
        batch_size: int = 1,
        engine: Literal["RK4", "Euler"] = "RK4",
        render_mode=None,
    ):
        """
        Initialize an `ODEEnv`.

        Args:
            state_space: The space of states.
            observation_space: The space of observations.
            action_space: The space of actions.
            time_steps: How many simulation steps to perform.
            step_size: The time step size to use for simulation.
            render_mode: See `gymnasium.Env`.
            batch_size: How many simulations to perform in one batch.
                If `None`, a single simulation is performed.
            n_costs: The number of cost terms (safety constraints).
            engine: The integration method to use for solving the ODE.
                Options: `RK4` (the Runge-Kutta 4 method), `Euler` (the Euler method).
        """
        self.state_space: Final = state_space
        self.observation_space: Final = observation_space
        self.action_space: Final = action_space
        self.initial_state_options_space: Final = initial_state_options_space

        self.time_steps: Final = time_steps
        self.step_size: Final = step_size
        self.n_costs: Final = n_costs
        self.batch_size: Final = batch_size
        self.engine: Final = engine

        self.__state: S | None = None
        self.__action: A | None = None
        self.__t: float | None = None
        self.__time_step: int | None = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = (512, 512)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.figure = None
        self._plot_data = None
        self.__plt_was_interactive = True

    @abstractmethod
    def _initial_state(self, options: IO) -> S:
        """The initial state of the ODE.

        When this method is called, `self.np_random` is seeded, so that you can use
        it to randomly initialize the initial state.

        Args:
            options: User arguments for resetting the environment.

        Returns:
            The initial state
        """
        raise NotImplementedError()

    @abstractmethod
    def _derivative(self, state: S, action: A, t: float) -> S:
        """The derivative `x'(t) = f(x(t), u(t), t)` of the ODE.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched.
            t: The time `t`.

        Returns:
            The (batched) derivative `x'(t)` of the ODE.
        """
        raise NotImplementedError()

    @abstractmethod
    def _reward(self, state: S, action: A, t: float) -> NDArray[np.float32]:
        """Returns the reward for a state and action.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched.
            t: The time `t`.

        Returns:
            The (batched) reward.
        """
        raise NotImplementedError()

    @abstractmethod
    def _costs(self, state: S, action: A, t: float) -> tuple[NDArray[np.float32], ...]:
        """Returns a cost for satisfying/violating the safety specification.

        If the cost is positive, the specification is violated.
        If it is negative, it is violated.

        The default implementation always return -1 (safe).

        Args:
            state: The state `x(t)´. Is batched.
            action: The actions `u(t)`. Is batched.
            t: The time `t`.

        Returns:
            The (batched) safety costs.
        """
        raise NotImplementedError()

    def _obs(self, state: S, action: A | None, t: float) -> O:
        """Construct observations from state `x(t)`, action `u(t)` and time step `t`.

        Returns the state by default.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched. None at initialization.
            t: The time `t`.

        Returns:
            The (batched) observations for the given state.
        """
        return state

    def _info(self, state: S, action: A | None, t: float) -> dict[str, Any]:
        """Get additional information to be returned by `step`.

        Returns an empty dictionary by default.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched. None at initialization.
            t: The time `t`.
        """
        return {}

    @property
    def t(self):
        """The current time of the simulation."""
        return self.__t

    @property
    def time_step(self):
        """The current time step of the simulation."""
        return self.__time_step

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: IO | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.__t = 0.0
        self.__time_step = 0

        x0 = self._initial_state(options)
        obs = self._obs(x0, None, self.__t)
        info = self._info(x0, None, self.__t)
        self.__state = x0
        self.__action = None

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(
        self, action: A
    ) -> tuple[O, NDArray[np.float32], ..., bool, bool, dict[str, Any]]:
        """Run one simulation step.

        Args:
            action: The action `u(t)`. Batched.

        Returns:
            The observations, the reward, one or several (safety) costs,
            whether the simulation, was terminated, whether it was truncated,
            and a dictionary containing auxiliary information.
            Observations, reward, and costs are batched. The auxiliary information
            may also be batched.
        """
        next_state = self._engine(self.__state, action, self.__t)
        self.__t += self.step_size
        self.__time_step += 1
        self.__state = next_state
        self.__action = action

        reward = self._reward(next_state, action, self.__t)
        costs = self._costs(next_state, action, self.__t)
        obs = self._obs(next_state, action, self.__t)
        info = self._info(next_state, action, self.__t)

        terminated = self.__time_step >= self.time_steps

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, *costs, terminated, False, info

    def _engine(self, state: S, action: A, t: float) -> S:
        """Runs one simulation step using the simulation engine."""
        match self.engine:
            case "Euler":
                return self._euler(state, action, t)
            case "RK4":
                return self._rk4(state, action, t)
            case _:
                raise NotImplementedError()

    def _euler(self, state: S, action: A, t: float) -> S:
        h = self.step_size
        k1 = self._derivative(state, action, t + h)
        return map_space(lambda s, k1: s + h * k1, self.state_space, state, k1)

    def _rk4(self, state: S, action: A, t: float) -> S:
        def map_(f, *args):
            return map_space(f, self.state_space, *args)

        h = self.step_size
        k1 = self._derivative(state, action, t)
        k2 = self._derivative(
            map_(lambda s, k1: s + h * k1 / 2, state, k1), action, t + h / 2
        )
        k3 = self._derivative(
            map_(lambda s, k2: s + h * k2 / 2, state, k2), action, t + h / 2
        )
        k4 = self._derivative(map_(lambda s, k3: s + h * k3, state, k3), action, t + h)

        def update(s, k1, k2, k3, k4):
            return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return map_(update, state, k1, k2, k3, k4)

    # ==================================================================================
    # MARK: rendering
    # ==================================================================================

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.figure is None and self.render_mode == "human":
            self.__plt_was_interactive = plt.isinteractive()
            plt.ion()
            self.figure, self._plot_data = self._prepare_figure()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        self._draw(self.__state, self.__action, canvas)

        match self.render_mode:
            case "human":
                self._plot(self.__state, self.__action, self.figure, self._plot_data)

                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                self.clock.tick(self.metadata["render_fps"])
            case "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    @abstractmethod
    def _draw(self, state: S, action: A | None, canvas: pygame.Surface):
        """Draw the current state."""
        raise NotImplementedError()

    def _prepare_figure(self) -> tuple[plt.Figure | None, Any | None]:
        """Create Matplotlib plots for plotting."""
        return None, None

    def _plot(self, state: S, action: A | None, fig: plt.Figure, plot_data: Any):
        """Update the state plot."""
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        if self.figure is not None:
            plt.close(self.figure)
            if not self.__plt_was_interactive:
                plt.ioff()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
