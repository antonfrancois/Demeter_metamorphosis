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
    controls_heading: Any
    parameter_button: Button
    menu_button: Button
    image_button: Button
    file_button: Button
    run_button: Button
    register_button: Button
    clear_button: Button
    clear_all_button: Button
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
    controls_heading = fig.text(
        0.89,
        0.86,
        "CONTROLS",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=INK_COLOR,
    )

    parameter_button = Button(
        fig.add_axes([0.805, 0.78, 0.17, 0.05]),
        "PARAMETER MENU  [P]",
        color=PANEL_COLOR,
        hovercolor="#d7e4e2",
    )
    menu_button = Button(
        fig.add_axes([0.805, 0.715, 0.17, 0.05]),
        "VIEW MENU  [V]",
        color=PANEL_COLOR,
        hovercolor="#d7e4e2",
    )
    image_button = Button(
        fig.add_axes([0.805, 0.65, 0.17, 0.05]),
        "IMAGES  [I]",
        color=PANEL_COLOR,
        hovercolor="#d7e4e2",
    )
    file_button = Button(
        fig.add_axes([0.805, 0.585, 0.17, 0.05]),
        "LOAD / SAVE  [L]",
        color=PANEL_COLOR,
        hovercolor="#d7e4e2",
    )
    model_label = "SPLINE" if model == "splines" else model.upper()
    register_button = Button(
        fig.add_axes([0.805, 0.48, 0.17, 0.07]),
        f"REGISTER {model_label}",
        color="#4267ac",
        hovercolor="#557bc2",
    )
    register_button.label.set_color("white")
    register_button.label.set_fontweight("bold")
    run_button = Button(
        fig.add_axes([0.805, 0.405, 0.17, 0.055]),
        f"RUN {model_label}",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    run_button.label.set_color("white")
    run_button.label.set_fontweight("bold")
    clear_button = Button(
        fig.add_axes([0.805, 0.32, 0.17, 0.055]),
        "CLEAR DISPLAYED FIELD",
        color=PANEL_COLOR,
        hovercolor="#f1c4b5",
    )
    clear_all_button = Button(
        fig.add_axes([0.805, 0.24, 0.17, 0.055]),
        "CLEAR ALL FIELDS",
        color=PANEL_COLOR,
        hovercolor="#efad99",
    )
    for button in (
        parameter_button,
        menu_button,
        image_button,
        file_button,
        clear_button,
        clear_all_button,
    ):
        button.label.set_fontsize(9)

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
        controls_heading,
        parameter_button,
        menu_button,
        image_button,
        file_button,
        run_button,
        register_button,
        clear_button,
        clear_all_button,
        time_slider,
        status_text,
    )
