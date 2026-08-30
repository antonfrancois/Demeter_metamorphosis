"""Construction of the persistent three-panel spline workspace."""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import torch

from .styles import CANVAS_COLOR, INK_COLOR, PANEL_COLOR


@dataclass
class Workspace:
    fig: Any
    axes: tuple[Any, Any, Any]
    images: tuple[Any, Any, Any]
    colorbar_axes: tuple[Any, Any, Any]
    footers: tuple[Any, Any, Any]
    buttons: dict[str, Button]
    time_slider: Slider
    status_text: Any


def build_workspace(
    source: torch.Tensor,
    target: torch.Tensor,
    n_steps: int,
    model: str,
    dynamic_artists: dict[Any, list[Any]],
) -> Workspace:
    fig = plt.figure(figsize=(17, 9.2), facecolor=CANVAS_COLOR)
    grid = fig.add_gridspec(
        1,
        3,
        left=0.04,
        right=0.77,
        bottom=0.24,
        top=0.79,
        wspace=0.13,
    )
    axes = tuple(fig.add_subplot(grid[0, index]) for index in range(3))
    height, width = source.shape[-2:]
    for axis in axes:
        axis.set_facecolor("#11191d")
        axis.set_aspect("equal")
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(-0.5, height - 0.5)
        axis.set_axis_off()
        dynamic_artists[axis] = []

    source_ax, current_ax, target_ax = axes
    images = (
        source_ax.imshow(source[0, 0], cmap="gray", origin="lower", vmin=0, vmax=1),
        current_ax.imshow(source[0, 0], cmap="gray", origin="lower", vmin=0, vmax=1),
        target_ax.imshow(target[0, 0], cmap="gray", origin="lower", vmin=0, vmax=1),
    )
    colorbar_axes = tuple(
        axis.inset_axes((0.12, -0.085, 0.76, 0.025))
        for axis in axes
    )
    for axis in colorbar_axes:
        axis.set_visible(False)

    def panel_footer(axis):
        return axis.text(
            0.5,
            -0.14,
            "",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            color=INK_COLOR,
            clip_on=False,
        )

    footers = tuple(panel_footer(axis) for axis in axes)
    fig.suptitle(
        "Metamorphosis Lab",
        x=0.405,
        y=0.982,
        fontsize=18,
        fontweight="bold",
        color=INK_COLOR,
    )
    fig.text(
        0.89,
        0.86,
        "CONTROLS",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=INK_COLOR,
    )

    def sidebar_button(
        y,
        label,
        *,
        height=0.05,
        color=PANEL_COLOR,
        hovercolor="#d7e4e2",
        emphasized=False,
    ):
        button = Button(
            fig.add_axes([0.805, y, 0.17, height]),
            label,
            color=color,
            hovercolor=hovercolor,
        )
        if emphasized:
            button.label.set_color("white")
            button.label.set_fontweight("bold")
        else:
            button.label.set_fontsize(9)
        return button

    parameter_button = sidebar_button(0.78, "PARAMETER MENU  [P]")
    menu_button = sidebar_button(0.715, "VIEW MENU  [V]")
    image_button = sidebar_button(0.65, "IMAGES  [I]")
    file_button = sidebar_button(0.585, "LOAD / SAVE  [L]")
    model_label = "SPLINE" if model == "splines" else model.upper()
    register_button = sidebar_button(
        0.48,
        f"REGISTER {model_label}",
        height=0.07,
        color="#4267ac",
        hovercolor="#557bc2",
        emphasized=True,
    )
    run_button = sidebar_button(
        0.405,
        f"RUN {model_label}",
        height=0.055,
        color="#168a8a",
        hovercolor="#20a3a3",
        emphasized=True,
    )
    clear_button = sidebar_button(
        0.32,
        "CLEAR DISPLAYED FIELD",
        height=0.055,
        hovercolor="#f1c4b5",
    )
    clear_all_button = sidebar_button(
        0.24,
        "CLEAR ALL FIELDS",
        height=0.055,
        hovercolor="#efad99",
    )

    time_slider = Slider(
        fig.add_axes([0.11, 0.105, 0.59, 0.028], facecolor=PANEL_COLOR),
        "t",
        0,
        n_steps,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )
    time_slider.valtext.set_fontsize(9)
    status_text = fig.text(
        0.405,
        0.064,
        "",
        ha="center",
        fontsize=11.5,
        color=INK_COLOR,
    )
    fig.text(
        0.012,
        0.975,
        "P  parameters\nV  view menu\nI  images\nL  files\nR  run\nG  register\n"
        "Arrow keys  move in time\n[ / ]  knots\n"
        "Mouse L/R  paint +/-\nShift-drag  erase",
        ha="left",
        va="top",
        fontsize=8.2,
        color="#63747a",
        linespacing=1.25,
    )
    return Workspace(
        fig,
        axes,
        images,
        colorbar_axes,
        footers,
        {
            "parameters": parameter_button,
            "view": menu_button,
            "images": image_button,
            "files": file_button,
            "run": run_button,
            "register": register_button,
            "clear": clear_button,
            "clear_all": clear_all_button,
        },
        time_slider,
        status_text,
    )
