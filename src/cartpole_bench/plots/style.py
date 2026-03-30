from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from cartpole_bench.config import load_theme_config
from cartpole_bench.types import RenderThemeConfig


MODE_COLORS = {
    "initial": "#D7D7D7",
    "energy_pump": "#9AA3AF",
    "capture_assist": "#C7A34C",
    "balance": "#6E966E",
}


def apply_theme(theme: str = "paper_white") -> RenderThemeConfig:
    cfg = load_theme_config(theme)
    plt.rcParams.update(
        {
            "figure.dpi": cfg.dpi,
            "savefig.dpi": cfg.dpi,
            "font.family": cfg.font_family,
            "font.weight": "normal",
            "text.color": cfg.text_color,
            "axes.edgecolor": cfg.spine_color,
            "axes.labelcolor": cfg.text_color,
            "axes.linewidth": cfg.axes_line_width,
            "axes.facecolor": cfg.panel_color,
            "axes.titleweight": "normal",
            "figure.facecolor": cfg.background_color,
            "figure.titleweight": "normal",
            "savefig.facecolor": cfg.background_color,
            "savefig.edgecolor": cfg.background_color,
            "xtick.color": cfg.text_color,
            "ytick.color": cfg.text_color,
            "grid.color": cfg.grid_color,
            "grid.linestyle": "-",
            "lines.linewidth": cfg.line_width,
            "legend.frameon": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    return cfg


def controller_color(theme_cfg: RenderThemeConfig, controller: str) -> str:
    return theme_cfg.controller_colors.get(controller, theme_cfg.text_color)


def soften(color: str, amount: float = 0.82) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(color)
    white = (1.0, 1.0, 1.0)
    return tuple((1.0 - amount) * channel + amount * white_channel for channel, white_channel in zip(rgb, white, strict=True))


def style_axis(ax, theme_cfg: RenderThemeConfig) -> None:
    ax.set_facecolor(theme_cfg.panel_color)
    ax.grid(True, alpha=0.55, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme_cfg.spine_color)
    ax.spines["bottom"].set_color(theme_cfg.spine_color)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, width=0.8)


def add_panel_title(ax, title: str, subtitle: str | None = None, theme_cfg: RenderThemeConfig | None = None) -> None:
    if theme_cfg is None:
        raise ValueError("theme_cfg is required")
    ax.set_title(title, fontsize=9.2, fontweight="normal", color=theme_cfg.text_color, pad=6)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=theme_cfg.muted_color,
        )


def add_panel_tag(ax, tag: str, theme_cfg: RenderThemeConfig) -> None:
    ax.text(
        -0.09,
        1.02,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=theme_cfg.text_color,
    )


def add_badge(ax, text: str, theme_cfg: RenderThemeConfig) -> None:
    ax.text(
        0.98,
        1.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=theme_cfg.accent_color,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": soften(theme_cfg.accent_color, 0.88), "edgecolor": theme_cfg.accent_color},
    )


def add_event_band(ax, start: float, end: float, theme_cfg: RenderThemeConfig, alpha: float = 0.08) -> None:
    ax.axvspan(start, end, color=theme_cfg.accent_color, alpha=alpha, linewidth=0.0)


def plot_percentile_band(
    ax,
    x: np.ndarray,
    samples: np.ndarray,
    color: str,
    *,
    median_width: float = 1.9,
    band_alpha: float = 0.16,
    label: str | None = None,
) -> None:
    lower = np.nanpercentile(samples, 25.0, axis=0)
    median = np.nanpercentile(samples, 50.0, axis=0)
    upper = np.nanpercentile(samples, 75.0, axis=0)
    ax.fill_between(x, lower, upper, color=color, alpha=band_alpha, linewidth=0.0)
    ax.plot(x, median, color=color, linewidth=median_width, label=label)


def controller_badge(ax, text: str, color: str, theme_cfg: RenderThemeConfig, *, y: float = 0.965) -> None:
    ax.text(
        0.03,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": soften(color, 0.85),
            "edgecolor": color,
        },
    )


def save_figure(fig, path: Path, theme_cfg: RenderThemeConfig) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=theme_cfg.background_color)
    plt.close(fig)
