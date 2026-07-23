"""Categorized model, drawing, and numerical parameter menu."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
import torch

from ..core import SplineParameters
from ..styles import INK_COLOR, PANEL_COLOR
from .common import build_modal_backdrop, build_panel, set_radio_visible
from .control_times import ControlTimeEditor


MAX_STEPS = 40


@dataclass
class ParameterMenu:
    backdrop_ax: Any
    panel_axes: dict[str, Any]
    rho_slider: Slider
    alpha_slider: Slider
    beta_slider: Slider
    gamma_slider: Slider
    brush_slider: Slider
    amplitude_slider: Slider
    steps_slider: Slider
    device_radio: RadioButtons
    control_time_editor: ControlTimeEditor
    close_button: Button

    @property
    def sliders(self) -> list[Slider]:
        return [
            self.rho_slider,
            self.alpha_slider,
            self.beta_slider,
            self.gamma_slider,
            self.brush_slider,
            self.amplitude_slider,
            self.steps_slider,
        ]

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            *self.panel_axes.values(),
            *(slider.ax for slider in self.sliders),
            self.device_radio.ax,
            self.control_time_editor.axis,
            self.close_button.ax,
        ]

    @property
    def widgets(self) -> list[Any]:
        return [*self.sliders, self.device_radio, self.close_button]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)
        for widget in self.widgets:
            widget.active = visible
        set_radio_visible(self.device_radio, visible)
        self.control_time_editor.set_visible(visible)


def build_parameter_menu(
    fig,
    parameters: SplineParameters,
    *,
    device: str,
    on_control_add,
    on_control_move,
    on_control_remove,
    on_control_select,
    on_message,
) -> ParameterMenu:
    backdrop = build_modal_backdrop(
        fig,
        "PARAMETER MENU",
        "Model, drawing, and numerical controls.  Press P or Esc to return.",
    )
    panels = {
        "model": build_panel(fig, [0.06, 0.14, 0.50, 0.70], "MODEL"),
        "draw": build_panel(fig, [0.60, 0.51, 0.34, 0.33], "DRAW"),
        "numerical": build_panel(fig, [0.60, 0.20, 0.34, 0.25], "NUMERICAL"),
    }
    panels["model"].text(
        0.5,
        0.68,
        r"OPERATOR  $L:\quad Lv=-\alpha\Delta v-\beta\nabla(\nabla\!\cdot v)"
        r"+\gamma v,\quad K=L^{-1}$",
        transform=panels["model"].transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    panels["model"].text(
        0.5,
        0.19,
        r"CONTROL TIMES  $\tau_c$",
        transform=panels["model"].transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    panels["model"].text(
        0.5,
        0.035,
        "Left-click add/select, drag move, right-click remove",
        transform=panels["model"].transAxes,
        ha="center",
        fontsize=8.5,
        color="#63747a",
    )

    def slider(position, label, low, high, value, **kwargs):
        widget = Slider(
            fig.add_axes(position, facecolor=PANEL_COLOR, zorder=102),
            label,
            low,
            high,
            valinit=value,
            **kwargs,
        )
        widget.label.set_x(-0.10)
        widget.label.set_fontsize(10)
        widget.valtext.set_fontsize(9.5)
        return widget

    rho = slider([0.17, 0.69, 0.32, 0.028], r"$\rho$", 0, max(0.95, parameters.rho), parameters.rho)
    alpha = slider([0.17, 0.53, 0.32, 0.028], r"$\alpha$", 0, max(2, 1.5 * parameters.alpha), parameters.alpha)
    beta = slider([0.17, 0.43, 0.32, 0.028], r"$\beta$", 0, max(2, 1.5 * parameters.beta), parameters.beta)
    log_gamma = float(np.log10(parameters.gamma))
    gamma = slider(
        [0.17, 0.33, 0.32, 0.028],
        r"$\gamma$",
        min(-5, np.floor(log_gamma) - 1),
        max(1, np.ceil(log_gamma) + 1),
        log_gamma,
    )
    gamma.valtext.set_text(f"{parameters.gamma:.3g}")
    brush = slider([0.70, 0.68, 0.17, 0.028], "Brush", 1, 40, 3)
    amplitude = slider([0.70, 0.57, 0.17, 0.028], "Amplitude", 0.01, 4, 0.5)
    steps = slider(
        [0.70, 0.31, 0.17, 0.028],
        "steps",
        1,
        MAX_STEPS,
        parameters.n_steps,
        valstep=1,
        valfmt="%0.0f",
    )
    panels["numerical"].text(
        0.5,
        0.34,
        "COMPUTE DEVICE",
        transform=panels["numerical"].transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=INK_COLOR,
    )
    device_names = (
        ["CUDA"]
        if torch.cuda.is_available() or device.startswith("cuda")
        else []
    )
    device_names.append("CPU")
    device_radio = RadioButtons(
        fig.add_axes([0.70, 0.21, 0.17, 0.06], facecolor=PANEL_COLOR, zorder=102),
        device_names,
        active=0 if device.startswith("cuda") else len(device_names) - 1,
        activecolor="#168a8a",
    )
    for label in device_radio.labels:
        label.set_fontsize(9)
    control_axis = fig.add_axes(
        [0.16, 0.205, 0.34, 0.065],
        facecolor=PANEL_COLOR,
        zorder=102,
    )
    control_editor = ControlTimeEditor(
        control_axis,
        n_steps=parameters.n_steps,
        control_steps=parameters.control_steps,
        on_add=on_control_add,
        on_move=on_control_move,
        on_remove=on_control_remove,
        on_select=on_control_select,
        on_message=on_message,
    )
    close = Button(
        fig.add_axes([0.42, 0.06, 0.16, 0.055], zorder=102),
        "RETURN TO IMAGES  [P]",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = ParameterMenu(
        backdrop,
        panels,
        rho,
        alpha,
        beta,
        gamma,
        brush,
        amplitude,
        steps,
        device_radio,
        control_editor,
        close,
    )
    menu.set_visible(False)
    return menu
