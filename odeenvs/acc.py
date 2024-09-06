#  Copyright (c) 2024 David Boetius.
#  Licensed under the MIT license
from typing import Final, override, Literal, Any

import gymnasium as gym
import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pygame

from .env import ODEEnv


__all__ = ("ACCEnv",)


_S = dict[
    Literal["x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego"],
    NDArray[np.float32],
]
_O = dict[Literal["v_ego", "d_rel", "v_rel"], NDArray[np.float32]]
_A = dict[Literal["in_ego"], NDArray[np.float32]]
_IO = dict[Literal["in_lead"], NDArray[np.float32]]


class ACCEnv(ODEEnv[_S, _O, _A, _IO]):
    """Adaptive Cruise Control (ACC) Environment.

    The actor accelerates a car (the *ego* car) to follow the *lead* car.
    The goal is to follow the lead car while maintaining a safe distance.

    State variables:
     - `x_lead`: The position of the lead car.
     - `v_lead`: The velocity of the lead car.
     - `a_lead`: The acceleration of the lead car.
     - `x_ego`: The position of the ego car.
     - `v_ego`: The velocity of the ego car.
     - `a_ego`: The acceleration of the ego car.

    Observations:
     - `v_ego`: The velocity of the ego car.
     - `d_rel = x_lead - x_ego`: The relative distance between the lead car and the ego car.
     - `v_rel = v_lead - v_ego`: The relative velocity of the ego car relative to the lead car.

    Actions:
     - `in_ego`: Ego car acceleration input.

     Initial state options:
      - `in_lead`: The throttle input of the lead car. Accepts an array of size
        `in_lead_switches` (initializer argument).

     Dynamics:

        a_lead' = 2 * (in_lead - a_lead)
        v_lead' = a_lead
        x_lead' = v_lead

        a_ego' = 2 * (in_ego - a_ego)
        v_ego' = a_ego
        x_ego' = v_ego

    Parameters:
     - `safe_distance_absolute` (`D`): a fixed distance that the ego car needs to
        maintain to the lead car.
     - `safe_distance_relative` (`t_gap`): the distance relative to the ego car
        velocity that the ego car needs to maintain to the lead car.
        The overall distance that the ego car needs to maintain is
        `safe_distance_absolute + safe_distance_relative * v_ego`.
     - `set_velocity` (`v_set`): The set velocity of the ego car.
        The ego car may not exceed `v_set + 0.1` in velocity.

    Reward:
        `-d_rel`

    Safety constraint:
        `Always (d_rel >= D + t_gap * v_ego AND v_ego <= v_set + 0.1)`
        Resulting cost: max(D + t_gap * v_ego - d_rel, v_ego - (v_set + 0.1))
    """

    def __init__(
        self,
        time_steps: int = 500,
        step_size: float = 0.1,
        safe_distance_absolute: float = 10.0,
        safe_distance_relative: float = 1.4,
        set_velocity: float = 30.0,
        x0_lead: float = 70.0,
        v0_lead: float = 28.0,
        x0_ego: float = 10.0,
        v0_ego: float = 22.0,
        a_lead_min: float = -1.0,
        a_lead_max: float = 1.0,
        a_ego_min: float = -3.0,
        a_ego_max: float = 2.0,
        in_lead_switches: int = 5,
        batch_size: int = 1,
        engine: Literal["RK4", "Euler"] = "RK4",
        render_mode=None,
    ):
        assert time_steps % in_lead_switches == 0, (
            f"The number of time steps must be a multiple of in_lead_switches. "
            f"Got {time_steps=} and {in_lead_switches=}."
        )

        def batched_box(low, high):
            return gym.spaces.Box(low, high, shape=(batch_size,), dtype=np.float32)

        state_space = gym.spaces.Dict(
            {
                "x_lead": batched_box(low=0, high=np.inf),
                "v_lead": batched_box(low=-np.inf, high=np.inf),
                "a_lead": batched_box(low=a_lead_min, high=a_lead_max),
                "x_ego": batched_box(low=0, high=np.inf),
                "v_ego": batched_box(low=-np.inf, high=np.inf),
                "a_ego": batched_box(low=a_lead_min, high=a_lead_max),
            }
        )
        obs_space = gym.spaces.Dict(
            {
                "v_ego": batched_box(low=-np.inf, high=np.inf),
                "d_rel": batched_box(low=0, high=np.inf),
                "v_rel": batched_box(low=-np.inf, high=np.inf),
            }
        )
        act_space = gym.spaces.Dict(
            {"in_ego": batched_box(low=a_ego_min, high=a_ego_max)}
        )
        init_options_space = gym.spaces.Dict(
            {
                "in_lead": gym.spaces.Box(
                    low=a_lead_min,
                    high=a_lead_max,
                    shape=(
                        batch_size,
                        in_lead_switches,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        super().__init__(
            state_space,
            obs_space,
            act_space,
            init_options_space,
            time_steps,
            step_size,
            batch_size,
            engine,
            render_mode,
        )
        self.safe_distance_absolute: Final = safe_distance_absolute
        self.safe_distance_relative: Final = safe_distance_relative
        self.set_velocity: Final = set_velocity
        self.x0_lead: Final = x0_lead
        self.v0_lead: Final = v0_lead
        self.x0_ego: Final = x0_ego
        self.v0_ego: Final = v0_ego
        self.in_lead_switches: Final = in_lead_switches

        self.__initial_state = {
            "x_lead": self.x0_lead,
            "v_lead": self.v0_lead,
            "a_lead": 0.0,
            "x_ego": self.x0_ego,
            "v_ego": self.v0_ego,
            "a_ego": 0.0,
        }
        self.__in_lead = None

        self.window_size = (750, 320)

    @override
    def _initial_state(self, options: _IO | None) -> _S:
        if options is not None:
            in_lead = options["in_lead"]
        else:
            sub_seed = int(self.np_random.integers(2**32 - 1))
            self.initial_state_options_space.seed(sub_seed)
            in_lead = self.initial_state_options_space.sample()["in_lead"]

        # stretch in_lead to the number of time steps
        repetitions = self.time_steps // self.in_lead_switches
        self.__in_lead = np.repeat(in_lead, repetitions, axis=-1)

        return {
            key: np.full((self.batch_size,), val, dtype=np.float32)
            for key, val in self.__initial_state.items()
        }

    @staticmethod
    def _car_model(x, v, a, in_):
        d_a = 2 * (in_ - a)
        d_v = a
        d_x = v
        return d_x, d_v, d_a

    @override
    def _derivative(self, state: _S, action: _A, t: float) -> _S:
        x_lead, v_lead, a_lead = state["x_lead"], state["v_lead"], state["a_lead"]
        x_ego, v_ego, a_ego = state["x_ego"], state["v_ego"], state["a_ego"]
        in_ego = action["in_ego"]

        in_lead = self.__in_lead[:, self.time_step]

        d_x_lead, d_v_lead, d_a_lead = self._car_model(x_lead, v_lead, a_lead, in_lead)
        d_x_ego, d_v_ego, d_a_ego = self._car_model(x_ego, v_ego, a_ego, in_ego)
        return {
            "x_lead": d_x_lead,
            "v_lead": d_v_lead,
            "a_lead": d_a_lead,
            "x_ego": d_x_ego,
            "v_ego": d_v_ego,
            "a_ego": d_a_ego,
        }

    @staticmethod
    def _d_rel(state: _S) -> np.ndarray:
        x_lead, x_ego = state["x_lead"], state["x_ego"]
        return x_lead - x_ego

    def _safe_distance(self, state: _S) -> np.ndarray:
        v_ego = state["v_ego"]
        return self.safe_distance_absolute + self.safe_distance_relative * v_ego

    @property
    def _max_velocity(self) -> float:
        return self.set_velocity + 0.1

    @override
    def _reward(self, state: _S, action: _A, t: float) -> np.ndarray:
        return -self._d_rel(state)

    @override
    def _cost(self, state: _S, action: _A, t: float) -> np.ndarray:
        v_ego = state["v_ego"]
        d_rel = self._d_rel(state)
        safe_d = self._safe_distance(state)
        d_cost = safe_d - d_rel
        v_cost = v_ego - self._max_velocity
        return np.maximum(d_cost, v_cost)

    @override
    def _obs(
        self,
        state: _S,
        action: _A | None,
        t: float,
    ) -> dict[str, np.ndarray]:
        v_lead, v_ego = state["v_lead"], state["v_ego"]
        v_rel = v_lead - v_ego
        d_rel = self._d_rel(state)
        return {"v_ego": v_ego, "d_rel": d_rel, "v_rel": v_rel}

    @override
    def _draw(self, state: _S, action: _A, canvas: pygame.Surface):
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

        x_time = (0.05 * float(state["x_lead"])) % 1
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
        car_y = strip_y + 0.375 * lane_h
        lead_x = 0.8 * w
        unit_car_distance = 1.1 * car_w / self.safe_distance_absolute

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
        cost = float(self._cost(state, action, self.t)[0])
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

        colors = {
            "in_ego": "tab:orange",
            "reward": "tab:green",
            "cost": "tab:red",
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
        }

        for var, ax in axes.items():
            ax.set_xlim(0.0, self.time_steps * self.step_size)
            ax.set_title(var)
            ax.set_xlabel("t")
            ax.set_ylabel(var)

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

            y = np.concat((old_y, [value[0]]))
            lines[var].set_data(x_, y)

            ax.relim()
            ax.autoscale_view()

        if action is not None:
            update_view("in_ego", action["in_ego"])

        reward = self._reward(state, action, self.t)
        update_view("reward", reward)
        cost = self._cost(state, action, self.t)
        update_view("cost", cost)

        for var, val in state.items():
            update_view(var, val)

        fig.canvas.draw()
        fig.canvas.flush_events()
