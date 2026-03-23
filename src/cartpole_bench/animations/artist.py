from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as patches
import numpy as np


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

    def __post_init__(self) -> None:
        self.ax.set_xlim(-self.track_limit - 0.6, self.track_limit + 0.6)
        self.ax.set_ylim(-0.14, self.pendulum_length + 0.34)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(self.panel_color)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axhspan(-0.14, 0.0, color=self.background_color, zorder=0)
        self.ax.plot(
            [-self.track_limit - 0.4, self.track_limit + 0.4],
            [0.0, 0.0],
            color=self.text_color,
            linewidth=0.95,
            alpha=0.75,
        )
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
        self.label = self.ax.text(0.03, 0.965, self.title, transform=self.ax.transAxes, ha="left", va="top", fontsize=8.4, color=self.text_color)
        self.status = self.ax.text(0.03, 0.89, "", transform=self.ax.transAxes, ha="left", va="top", fontsize=6.9, color=self.text_color)

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
        mode_label = {
            "energy_pump": "swing-up",
            "capture_assist": "capture",
            "balance": "balance",
        }.get(mode, mode.replace("_", " "))
        self.status.set_text(mode_label)
