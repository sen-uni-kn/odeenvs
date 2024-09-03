from abc import ABC, abstractmethod
from typing import Any, Literal, override

import numpy as np
import gymnasium as gym
from gymnasium.core import RenderFrame
import pygame


class ODEEnv(gym.Env[np.ndarray, np.ndarray], ABC):
    """The ODE Environment base class.

    Solves a ODE of the form
    ```
    x_dot(t) = f(x(t), u(t), t)
    ```
    where `x(t)` is the *state*, `u(t)` is the *action* from the agent, the
    input to the ODE and `t` is the *time*.

    The step method of this environment returns a safety cost
    as an additional value after the reward.

    To implement, you need to provide
     - an `action_space` attribute (`gymnasium.Env`)
     - an `observation_space` attribute (`gymnasium.Env`)
     - an `_initial_state` method to provides `x(0)`.
     - an `_derivative` method that computes `x_dot(t)`.
     - an `_reward` method that computes the reward for a state, action pair.

    Additionally, you can implement
     - `_cost()` to provide a safety specification cost (robustness value).
     - `_obs()` to construct observations from the state.
     - `_info()` to provide additional information that is returned by `step`.
     - `_draw()` for rendering the current state.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(
        self,
        time_horizon: float,
        step_size: float,
        batch_size: int = 1,
        engine: Literal["RK4", "Euler"] = "RK4",
        render_mode=None,
    ):
        """
        Initialize an `ODEEnv`.

        Args:
            time_horizon: For how many time steps to run the simulation.
            step_size: The step size to use for simulation.
            render_mode: See `gymnasium.Env`.
            batch_size: How many simulations to perform in one batch.
                If `None`, a single simulation is performed.
            engine: The integration method to use for solving the ODE.
                Options: `RK4` (the Runge-Kutta 4 method), `Euler` (the Euler method).
        """
        self.time_horizon = time_horizon
        self.step_size = step_size
        self.batch_size = batch_size
        self.engine = engine

        self.__state = None
        self.__t = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    @abstractmethod
    def _initial_state(self, options) -> np.ndarray:
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
    def _derivative(self, state, action, t: float) -> np.ndarray:
        """The derivative `x_dot(t) = f(x(t), u(t), t)` of the ODE.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched.
            t: The time `t`.

        Returns:
            The (batched) derivative `x_dot(t)` of the ODE.
        """
        raise NotImplementedError()

    @abstractmethod
    def _reward(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        """Returns the reward for a state and action.

        Args:
            state: The states `x(t)`. Is batched.
            action: The actions `u(t)`. Is batched. None at initialization.
            t: The time `t`.

        Returns:
            The (batched) reward.
        """
        raise NotImplementedError()

    def _cost(self, state: np.ndarray, t: float) -> np.ndarray:
        """Returns a cost for satisfying/violating the safety specification.

        If the cost is positive, the specification is violated.
        If it is negative, it is violated.

        The default implementation always return -1 (safe).

        Args:
            state: The state `x(t)´. Is batched.
            t: The time `t`.

        Returns:
            The (batched) safety cost.
        """
        return -np.ones((state.shape[0],), state.dtype)

    def _obs(
        self, state: np.ndarray, action: np.ndarray | None, t: float
    ) -> np.ndarray:
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

    def _info(
        self, state: np.ndarray, action: np.ndarray | None, t: float
    ) -> dict[str, Any]:
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

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        x0 = self._initial_state(options)
        obs = self._obs(x0, None, 0.0)
        info = self._info(x0, None, 0.0)

        self.__t = 0.0
        self.__state = x0

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """Run one simulation step.

        Args:
            action: The action `u(t)`. Batched.

        Returns:
            The observations, the reward, the cost (safety), whether the simulation
            was terminated, whether it was truncated, and a dictionary containing
            auxiliary information.
            Observations, reward, and cost are batched. The auxiliary information
            may also be batched.
        """
        next_state = self._engine(self.__state, action, self.__t)
        self.__t += self.step_size
        self.__state = next_state

        reward = self._reward(next_state, action, self.__t)
        cost = self._cost(next_state, self.__t)
        obs = self._obs(next_state, action, self.__t)
        info = self._info(next_state, action, self.__t)

        terminated = self.__t >= self.time_horizon

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, cost, terminated, False, info

    def _engine(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        """Runs one simulation step using the simulation engine."""
        match self.engine:
            case "Euler":
                return self._euler(state, action, t)
            case "RK4":
                return self._rk4(state, action, t)
            case _:
                raise NotImplementedError()

    def _euler(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        h = self.step_size
        k1 = self._derivative(state, action, t + h)
        return state + h * k1

    def _rk4(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        h = self.step_size
        k1 = self._derivative(state, action, t)
        k2 = self._derivative(state + h * k1 / 2, action, t + h / 2)
        k3 = self._derivative(state + h * k2 / 2, action, t + h / 2)
        k4 = self._derivative(state + h * k3, action, t + h)
        return state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

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
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self._draw(self.__state, canvas)

        match self.render_mode:
            case "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                self.clock.tick(self.metadata["render_fps"])
            case "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    @abstractmethod
    def _draw(self, state: np.ndarray, canvas: pygame.Surface):
        """Draw the current state."""
        raise NotImplementedError()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
