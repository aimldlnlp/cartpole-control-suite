from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import matplotlib.patches as patches
import numpy as np


MODE_LABELS = {
    "energy_pump": "swing-up",
    "capture_assist": "capture",
    "balance": "balance",
}


@dataclass
class CartPoleArtist:
    ax: object
    track_limit: float
    pendulum_length: float
    title: str
    color: str
    text_color: str
    panel_color: str
    background_color: str
    trail_length: int = 18
    trail_points: deque[tuple[float, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.trail_points = deque(maxlen=self.trail_length)
        self.ax.set_xlim(-self.track_limit - 0.6, self.track_limit + 0.6)
        self.ax.set_ylim(-0.18, self.pendulum_length + 0.42)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.panel_color)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axhspan(-0.18, 0.0, color=self.background_color, zorder=0)
        self.ax.plot(
            [-self.track_limit - 0.4, self.track_limit + 0.4],
            [0.0, 0.0],
            color=self.text_color,
            linewidth=1.0,
            alpha=0.75,
        )
        self.ax.axvline(-self.track_limit, color=self.text_color, linewidth=0.8, linestyle="--", alpha=0.35)
        self.ax.axvline(self.track_limit, color=self.text_color, linewidth=0.8, linestyle="--", alpha=0.35)
        self.cart = patches.FancyBboxPatch(
            (-0.2, 0.02),
            0.4,
            0.18,
            boxstyle="round,pad=0.015,rounding_size=0.04",
            edgecolor=self.color,
            facecolor=self.panel_color,
            linewidth=1.35,
        )
        self.ax.add_patch(self.cart)
        (self.pendulum,) = self.ax.plot([], [], color=self.color, linewidth=1.9)
        self.bob = patches.Circle((0.0, 0.0), 0.045, edgecolor=self.color, facecolor=self.panel_color, linewidth=1.35)
        self.ax.add_patch(self.bob)
        (self.trail,) = self.ax.plot([], [], color=self.color, linewidth=1.0, alpha=0.18)
        self.label = self.ax.text(
            0.03,
            0.965,
            self.title,
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color=self.text_color,
        )
        self.status = self.ax.text(
            0.03,
            0.89,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            color=self.text_color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": self.panel_color, "edgecolor": self.color, "linewidth": 0.9},
        )

    def update(self, state: np.ndarray, t: float, mode: str) -> None:
        x, _, theta, _ = state
        cart_y = 0.02
        self.cart.set_bounds(x - 0.2, cart_y, 0.4, 0.18)
        pivot_x = x
        pivot_y = cart_y + 0.18
        end_x = pivot_x + self.pendulum_length * np.sin(theta)
        end_y = pivot_y + self.pendulum_length * np.cos(theta)
        self.pendulum.set_data([pivot_x, end_x], [pivot_y, end_y])
        self.bob.center = (end_x, end_y)
        self.trail_points.append((end_x, end_y))
        if len(self.trail_points) >= 2:
            xs, ys = zip(*self.trail_points, strict=True)
            self.trail.set_data(xs, ys)
        mode_label = MODE_LABELS.get(mode, mode.replace("_", " "))
        self.status.set_text(f"{mode_label} · t={t:4.2f}s")
