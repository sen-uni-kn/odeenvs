#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from typing import Final, override, Literal, Any

import gymnasium as gym
from gymnasium import spaces
import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pygame

from .env import ODEEnv


__all__ = ("ACCEnv",)


# _S = dict[
#     Literal["x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego"],
#     NDArray[np.float32],
# ]
# _O = dict[Literal["v_ego", "d_rel", "v_rel"], NDArray[np.float32]]
# _A = dict[Literal["in_ego"], NDArray[np.float32]]
# _IO = dict[Literal["in_lead"], NDArray[np.float32]]
_S = _O = _A = _IO = NDArray[np.float32]


class ACCEnv(ODEEnv[_S, _O, _A, _IO]):
    """Adaptive Cruise Control (ACC) Environment.

    The actor accelerates a car (the *ego* car) to follow the *lead* car
    or maintain a certain set velocity.
    The goal is to follow the lead car while maintaining a safe distance
    or maintaining the set velocity if the lead car is far ahead.

    This environment is vectorized: Every run performs a batch of simulations.

    Parameters:
     - `x0_lead`, `v0_lead`, ...: Fixed initial positions and velocities or a range of
        initial positions and velocities as a (min, max) tuple.
     - `a_lead_range`: Range of the throttle input of the lead car as a (min, max) tuple.
     - `a_ego_range`: Range of the throttle input of the ego car as a (min, max) tuple.
     - `mu`: A friction parameter.
     - `in_lead_switches`:
     - `switch_period`: Determines how frequently the throttle input of the
       lead car switches.
       The switch period is the duration (number of time steps)
       for which the throttle input of the lead car stays constant.
     - `safe_distance_absolute` (`D`): a fixed distance that the ego car needs to
        maintain to the lead car.
     - `safe_distance_time` (`t_gap`): the distance relative to the ego car
        velocity that the ego car needs to maintain to the lead car.
        The overall distance that the ego car needs to maintain is
        `D + t_gap * v_ego`.
     - `set_velocity` (`v_set`): The set velocity of the ego car.
     - `speed_limit` (`v_limit`): The maximum velocity the car may
        The ego car may not this velocity.

    State variables (index):
     - `x_lead` (0): The position of the lead car.
     - `v_lead` (1): The velocity of the lead car.
     - `a_lead` (2): The acceleration of the lead car.
     - `x_ego` (3): The position of the ego car.
     - `v_ego` (4): The velocity of the ego car.
     - `a_ego` (5): The acceleration of the ego car.

    Observations (index):
     - `v_ego` (0): The velocity of the ego car.
     - `d_rel = x_lead - x_ego` (1): The relative distance between the lead car and the ego car.
     - `v_rel = v_lead - v_ego` (2): The relative velocity of the ego car relative to the lead car.

    Actions:
     - `in_ego`: Ego car acceleration input.

    Initial state options:
     - An array of shape `(batch_size, 4 + in_lead_switches)`.
        The first for entries of the second dimension specify the initial
        positions and velocities `x0_lead, v0_lead, x0_ego, v0_ego` in this order.
        The initial acceleration is always zero.
        The remaining dimensions specify the throttle input of the lead car at the time
        points where this input switches.

    Dynamics:

        a_lead' = 2 * (in_lead - a_lead) - mu * v_lead^2
        v_lead' = a_lead
        x_lead' = v_lead

        a_ego' = 2 * (in_ego - a_ego) - mu * v_ego^2
        v_ego' = a_ego
        x_ego' = v_ego

    Safety constraint:
        `Always (d_rel >= D + t_gap * v_ego AND v_ego <= v_limit)`
        Costs:
          - `d_cost = (D + t_gap * v_ego - d_rel) / 10000`
          - `v_cost = (v_ego - v_limit) / 100`

    Reward:
        ```
        -where(d_rel >= 2 * d_save, |v_ego - v_set|, |v_ego - min(v_lead, v_set)|) / 100
        -0.001 * |a_ego|/a_scale - 0.01 * |a_ego - in_ego|/a_scale
        ```
        where `d_safe = (D + t_gap * v_ego)` and `a_scale = max(a_max, -a_min)`.

        The second term incentives efficiency (accelerate less).
        The last term relates the current acceleration to the current acceleration
        input for incentivizing driving comfort.

    """

    state_var = {
        "x_lead": 0,
        "v_lead": 1,
        "a_lead": 2,
        "x_ego": 3,
        "v_ego": 4,
        "a_ego": 5,
    }
    """Maps state variables to indices in the state array."""

    observation_var = {"v_ego": 0, "d_rel": 1, "v_rel": 2}
    """Maps observation variables to indices in the observation array."""

    def __init__(
        self,
        time_steps: int = 1000,
        step_size: float = 0.1,
        safe_distance_absolute: float = 10.0,
        safe_distance_time: float = 1.4,
        set_velocity: float = 30.0,
        speed_limit: float = 33.0,
        x0_lead: float | tuple[float, float] = (60.0, 100.0),
        v0_lead: float | tuple[float, float] = (10.0, 40.0),
        x0_ego: float | tuple[float, float] = (0.0, 0.0),
        v0_ego: float | tuple[float, float] = (10.0, 30.0),
        a_lead_range: tuple[float, float] = (-3.0, 3.0),
        a_ego_range: tuple[float, float] = (-3.0, 2.0),
        mu: float = 0.0001,
        switch_period: int = 250,
        batch_size: int = 1,
        engine: Literal["RK4", "Euler"] = "RK4",
        render_mode=None,
    ):
        def pair(val):
            return val if isinstance(val, tuple) else (val, val)

        x0_lead, v0_lead = pair(x0_lead), pair(v0_lead)
        x0_ego, v0_ego = pair(x0_ego), pair(v0_ego)

        def box(low, high, shape=None):
            low_high = (np.array(val) for val in (low, high))
            if shape is not None:
                low_high = (np.resize(a, shape) for a in low_high)
            return spaces.Box(*low_high, dtype=np.float32)

        a_lead_min, a_lead_max = a_lead_range
        a_ego_min, a_ego_max = a_ego_range
        state_space = box(
            # x_lead, v_lead, a_lead, x_ego, v_ego, a_ego
            low=[0, -np.inf, a_lead_min, 0, -np.inf, a_ego_min],
            high=[np.inf, np.inf, a_lead_max, np.inf, np.inf, a_ego_max],
        )
        obs_space = box(
            # v_ego, d_rel, v_rel
            low=[-np.inf, 0, -np.inf],
            high=[np.inf] * 3,
        )
        act_space = box(low=a_ego_min, high=a_ego_max)
        in_lead_switches = time_steps // switch_period
        init_options_space = box(
            low=[x0_lead[0], v0_lead[0], x0_ego[0], v0_ego[0]]
            + [a_lead_min] * in_lead_switches,
            high=[x0_lead[1], v0_lead[1], x0_ego[1], v0_ego[1]]
            + [a_lead_max] * in_lead_switches,
            shape=(batch_size, 4 + in_lead_switches),
        )

        super().__init__(
            state_space,
            obs_space,
            act_space,
            init_options_space,
            time_steps,
            step_size,
            n_costs=2,
            batch_size=batch_size,
            engine=engine,
            render_mode=render_mode,
        )
        self.safe_distance_absolute: Final[float] = safe_distance_absolute
        self.safe_distance_time: Final[float] = safe_distance_time
        self.set_velocity: Final[float] = set_velocity
        self.speed_limit: Final[float] = speed_limit
        self.x0_lead: Final[tuple[float, float]] = x0_lead
        self.v0_lead: Final[tuple[float, float]] = v0_lead
        self.x0_ego: Final[tuple[float, float]] = x0_ego
        self.v0_ego: Final[tuple[float, float]] = v0_ego
        self.a_lead_range: Final[tuple[float, float]] = a_lead_range
        self.a_ego_range: Final[tuple[float, float]] = a_ego_range
        self.switch_period: Final[int] = switch_period
        self.in_lead_switches: Final[int] = in_lead_switches
        self.in_lead_min: Final[float] = a_lead_min
        self.in_lead_max: Final[float] = a_lead_max
        self.mu: Final[float] = mu

        self.__in_lead = None

        self.window_size = (750, 320)

    def step(self, action: _A) -> tuple[
        _O,
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        bool,
        bool,
        dict[str, Any],
    ]:
        return super().step(action)  # type: ignore

    @override
    def _initial_state(self, options: _IO | None) -> _S:
        if options:
            init = options.reshape((-1, 4 + self.in_lead_switches))
        else:
            sub_seed = int(self.np_random.integers(0, 2 * 32 - 1))
            self.initial_state_options_space.seed(sub_seed)
            init = self.initial_state_options_space.sample()
        init_lead = init[:, :2]
        init_ego = init[:, 2:4]
        in_lead = init[:, 4:]

        # stretch in_lead to the number of time steps
        self.__in_lead = np.empty((self.batch_size, self.time_steps))
        size_repeated = self.switch_period * self.in_lead_switches
        self.__in_lead[:, :size_repeated] = np.repeat(
            in_lead, self.switch_period, axis=-1
        )
        if size_repeated < self.time_steps:
            self.__in_lead[:, size_repeated:] = in_lead[:, (-1,)]

        initial_state = np.empty((self.batch_size, 6), dtype=np.float32)
        initial_state[:, :2] = init_lead
        initial_state[:, 2] = 0.0  # lead acceleration
        initial_state[:, 3:5] = init_ego
        initial_state[:, 5] = 0.0  # ego acceleration
        return initial_state

    def _car_model(self, v, a, in_):
        d_a = 2 * (in_ - a) - self.mu * np.square(v)
        d_v = a
        d_x = v
        return d_x, d_v, d_a

    @override
    def _derivative(self, state: _S, action: _A, t: float) -> _S:
        x_lead, v_lead, a_lead, x_ego, v_ego, a_ego = state.T
        in_ego = action.reshape((-1, *self.action_space.shape))

        in_lead = self.__in_lead[:, self.time_step]

        d_x_lead, d_v_lead, d_a_lead = self._car_model(v_lead, a_lead, in_lead)
        d_x_ego, d_v_ego, d_a_ego = self._car_model(v_ego, a_ego, in_ego)
        return np.stack(
            [d_x_lead, d_v_lead, d_a_lead, d_x_ego, d_v_ego, d_a_ego], axis=-1
        )

    def _d_rel(self, state: _S) -> NDArray[np.float32]:
        x_lead, x_ego = self.get_state(state, "x_lead", "x_ego")
        return x_lead - x_ego

    def _safe_distance(self, state: _S) -> NDArray[np.float32]:
        (v_ego,) = self.get_state(state, "v_ego")
        return self.safe_distance_absolute + self.safe_distance_time * v_ego

    @override
    def _reward(self, state: _S, action: _A, t: float) -> NDArray[np.float32]:
        # return -np.maximum(self._d_rel(state) - self._safe_distance(state), 0.0)
        d_rel = self._d_rel(state)
        d_safe = self._safe_distance(state)
        v_lead, v_ego, a_ego = self.get_state(state, "v_lead", "v_ego", "a_ego")
        in_ego = action.reshape((-1, *self.action_space.shape))

        r_v_far = -np.abs(v_ego - self.set_velocity)
        r_v_near = -np.abs(v_ego - np.minimum(v_lead, self.set_velocity))
        r_v = np.where(d_rel >= 2 * d_safe, r_v_far, r_v_near)

        # normalize accelerations
        a_min, a_max = self.a_ego_range
        a_scale = max(a_max, -a_min)
        a_ego = a_ego / a_scale
        in_ego = in_ego / a_scale
        r_a = -np.abs(a_ego) - np.abs(a_ego - in_ego)

        return r_v / 100 + 0.001 * r_a

    @override
    def _costs(
        self, state: _S, action: _A, t: float
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        (v_ego,) = self.get_state(state, "v_ego")
        d_rel = self._d_rel(state)
        safe_d = self._safe_distance(state)

        d_cost = (safe_d - d_rel) / 1e4
        v_cost = (v_ego - self.speed_limit) / 100
        return d_cost, v_cost

    @override
    def _obs(
        self,
        state: _S,
        action: _A | None,
        t: float,
    ) -> _O:
        v_lead, v_ego = self.get_state(state, "v_lead", "v_ego")
        v_rel = v_lead - v_ego
        d_rel = self._d_rel(state)
        return np.stack([v_ego, d_rel, v_rel], axis=-1)

    def get_state(self, state, *state_vars) -> tuple[NDArray[np.float32], ...]:
        return tuple(state[:, self.state_var[var]] for var in state_vars)

    def get_obs(self, obs, *obs_vars) -> tuple[NDArray[np.float32], ...]:
        obs = obs.reshape((-1, *self.observation_space.shape))
        return tuple(obs[:, self.observation_var[var]] for var in obs_vars)

    @override
    def _draw(self, state: _S, action: _A | None, canvas: pygame.Surface):
        if action is not None:
            action = action.reshape((-1, *self.action_space.shape))
        # visual reference: https://commons.wikimedia.org/wiki/File:Schema_ICC.svg
        w, h = self.window_size

        street = (172, 172, 172)
        border = (100, 100, 100)
        white = (255, 255, 255)
        black = (0, 0, 0)
        ego_color = matplotlib.colors.TABLEAU_COLORS["tab:orange"]
        lead_color = matplotlib.colors.TABLEAU_COLORS["tab:blue"]
        good = matplotlib.colors.TABLEAU_COLORS["tab:green"]
        bad = matplotlib.colors.TABLEAU_COLORS["tab:red"]

        # street background
        border_h = 0.05 * h
        blank_bottom = 0.1 * h
        line_h = 9
        line_up_y = 0.12 * h
        line_down_y = h - blank_bottom - line_h
        canvas.fill(street)
        # upper border
        pygame.draw.rect(canvas, border, (0, 0, w, border_h))
        # upper line
        pygame.draw.rect(canvas, white, (0, line_up_y, w, line_h))
        pygame.draw.line(canvas, black, (0, line_up_y), (w, line_up_y), width=2)
        pygame.draw.line(
            canvas, black, (0, line_up_y + line_h), (w, line_up_y + line_h), width=2
        )
        # lower line
        pygame.draw.rect(canvas, white, (0, line_down_y, w, line_h))
        pygame.draw.line(canvas, black, (0, line_down_y), (w, line_down_y), width=2)
        pygame.draw.line(
            canvas, black, (0, line_down_y + line_h), (w, line_down_y + line_h), width=2
        )

        # center strip (moves with position)
        strip_h = 7
        lane_h = (line_down_y - line_up_y - line_h - strip_h) / 2
        strip_y = line_up_y + line_h + lane_h
        strip_w = 50
        strip_distance = 1.5 * strip_w

        x_time = (0.02 * float(state[0, self.state_var["x_lead"]])) % 1
        strip_x = -x_time * (strip_w + strip_distance) - 10
        while strip_x < w:
            pygame.draw.rect(canvas, white, (strip_x, strip_y, strip_w, strip_h))
            pygame.draw.rect(
                canvas, black, (strip_x, strip_y, strip_w, strip_h), width=2
            )
            strip_x += strip_w + strip_distance

        # cars
        car_h = 50
        car_w = 80
        roof_w = 45
        roof_h = 42
        roof_space_x = 13
        roof_space_y = 4
        car_y = strip_y + 0.2 * lane_h
        lead_x = 0.8 * w
        unit_car_distance = 0.75 * car_w / self.safe_distance_absolute

        def draw_car(x, color):
            pygame.draw.rect(canvas, color, (x, car_y, car_w, car_h))
            pygame.draw.rect(canvas, black, (x, car_y, car_w, car_h), width=2)
            pygame.draw.rect(
                canvas,
                black,
                (x + roof_space_x, car_y + roof_space_y, roof_w, roof_h),
                width=2,
            )

        draw_car(lead_x, lead_color)
        d_rel = self._d_rel(state)
        ego_x = lead_x - float(d_rel[0]) * unit_car_distance
        draw_car(ego_x, ego_color)

        # distance indicator
        d_cost, v_cost = self._costs(state, action, self.t)
        cost = max(float(d_cost[0]), float(v_cost[0]))
        line_color = bad if cost >= 0 else good
        line_y = car_y + car_h / 2
        line_sep = 10
        line_start = ego_x + car_w + line_sep
        line_end = lead_x - line_sep
        whisker_head_h = 20
        pygame.draw.line(
            canvas,
            line_color,
            (line_start, line_y),
            (line_end, line_y),
            width=4,
        )
        pygame.draw.line(
            canvas,
            line_color,
            (line_start, line_y - whisker_head_h / 2),
            (line_start, line_y + whisker_head_h / 2),
            width=4,
        )
        pygame.draw.line(
            canvas,
            line_color,
            (line_end, line_y - whisker_head_h / 2),
            (line_end, line_y + whisker_head_h / 2),
            width=4,
        )

    @override
    def _prepare_figure(
        self,
    ) -> tuple[plt.Figure, tuple[dict[str, plt.Axes], dict[str, plt.Line2D]]]:
        fig, axes = plt.subplot_mosaic(
            [
                ["in_ego", "reward", "cost"],
                ["a_ego", "v_ego", "x_ego"],
                ["a_lead", "v_lead", "x_lead"],
            ],
            figsize=(10, 6),
        )
        axes["d_cost"] = axes["v_cost"] = axes["cost"]

        colors = {
            "in_ego": "tab:orange",
            "reward": "tab:green",
            "d_cost": "tab:red",
            "v_cost": "tab:blue",
            "a_ego": "tab:orange",
            "v_ego": "tab:orange",
            "x_ego": "tab:orange",
            "a_lead": "tab:blue",
            "v_lead": "tab:blue",
            "x_lead": "tab:blue",
        }
        lines = {
            var: ax.plot([], [], color=colors[var], linestyle="solid")[0]
            for var, ax in axes.items()
            if not var.endswith("cost")
        }
        lines["d_cost"] = axes["d_cost"].plot(
            [], [], color=colors["d_cost"], linestyle="solid"
        )[0]
        lines["v_cost"] = axes["v_cost"].plot(
            [], [], color=colors["v_cost"], linestyle="solid"
        )[0]

        for var, ax in axes.items():
            ax.set_xlim(0.0, self.time_steps * self.step_size)
            ax.set_title(var)
            ax.set_xlabel("t")
            ax.set_ylabel(var)

        axes["cost"].set_title("cost")
        axes["cost"].set_ylabel("cost")
        fig.tight_layout(pad=0.1)
        return fig, (axes, lines)

    @override
    def _plot(
        self,
        state: _S,
        action: _A,
        fig: plt.Figure,
        plot_data: tuple[dict[str, plt.Axes], dict[str, plt.Line2D]],
    ):
        axes, lines = plot_data

        x = np.arange(0, self.time_step + 1) * self.step_size

        def update_view(var, value):
            line = lines[var]
            ax = axes[var]

            x_ = x
            old_y = line.get_ydata(orig=True)
            if len(old_y) > len(x_):
                old_y = old_y[: len(x_) - 1]
            elif len(x) - 1 > len(old_y):
                x_ = x_[-len(old_y) - 1 :]

            y = np.concatenate((old_y, [value]))
            lines[var].set_data(x_, y)

            ax.relim()
            ax.autoscale_view()

        if action is not None:
            update_view("in_ego", action[0])

            reward = self._reward(state, action, self.t)
            update_view("reward", reward[0])
            d_cost, v_cost = self._costs(state, action, self.t)
            update_view("d_cost", d_cost[0])
            update_view("v_cost", v_cost[0])

        for var, idx in self.state_var.items():
            update_view(var, state[0, idx])

        fig.canvas.draw()
        fig.canvas.flush_events()
