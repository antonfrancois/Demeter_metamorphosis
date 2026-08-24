"""Categorized model, drawing, and numerical parameter menu."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import torch

from ...field_playground_core import DEFAULT_VECTOR_DISPLAY_SPACING
from ..core import OPTIMIZABLE_INITIAL_FIELDS, SplineParameters
from ..styles import INK_COLOR, PANEL_COLOR
from .common import build_modal_backdrop, build_panel, set_radio_visible
from .control_times import ControlTimeEditor


MAX_STEPS = 60
MAX_ITERATIONS = 50


def format_lbfgs_learning_rate(value: float) -> str:
    if value >= 1e-1:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"


@dataclass
class ParameterMenu:
    backdrop_ax: Any
    panel_axes: dict[str, Any]
    model_radio: RadioButtons
    rho_slider: Slider
    operator_radio: RadioButtons
    alpha_slider: Slider
    beta_slider: Slider
    gamma_slider: Slider
    sigma_slider: Slider
    optimized_fields_check: CheckButtons
    brush_slider: Slider
    amplitude_slider: Slider
    spacing_slider: Slider
    steps_slider: Slider
    iterations_slider: Slider
    cost_slider: Slider
    lbfgs_lr_slider: Slider
    regression_steps_slider: Slider
    regression_iterations_slider: Slider
    regression_cost_slider: Slider
    regression_lbfgs_lr_slider: Slider
    spline_numerical_heading: Any
    regression_numerical_heading: Any
    numerical_row_labels: tuple[Any, ...]
    numerical_split_line: Any
    spline_initialization_radio: RadioButtons
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
            self.sigma_slider,
            self.brush_slider,
            self.amplitude_slider,
            self.spacing_slider,
            self.steps_slider,
            self.iterations_slider,
            self.cost_slider,
            self.lbfgs_lr_slider,
            self.regression_steps_slider,
            self.regression_iterations_slider,
            self.regression_cost_slider,
            self.regression_lbfgs_lr_slider,
        ]

    @property
    def spline_numerical_sliders(self) -> tuple[Slider, ...]:
        return (
            self.cost_slider,
            self.steps_slider,
            self.iterations_slider,
            self.lbfgs_lr_slider,
        )

    @property
    def regression_numerical_sliders(self) -> tuple[Slider, ...]:
        return (
            self.regression_cost_slider,
            self.regression_steps_slider,
            self.regression_iterations_slider,
            self.regression_lbfgs_lr_slider,
        )

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            *self.panel_axes.values(),
            *(slider.ax for slider in self.sliders),
            self.operator_radio.ax,
            self.model_radio.ax,
            self.optimized_fields_check.ax,
            self.spline_initialization_radio.ax,
            self.device_radio.ax,
            self.control_time_editor.axis,
            self.close_button.ax,
        ]

    @property
    def widgets(self) -> list[Any]:
        return [
            *self.sliders,
            self.operator_radio,
            self.model_radio,
            self.optimized_fields_check,
            self.spline_initialization_radio,
            self.device_radio,
            self.close_button,
        ]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)
        for widget in self.widgets:
            widget.active = visible
        set_radio_visible(self.operator_radio, visible)
        set_radio_visible(self.model_radio, visible)
        set_radio_visible(self.spline_initialization_radio, visible)
        set_radio_visible(self.device_radio, visible)
        self.control_time_editor.set_visible(visible)
        self.set_warm_layout(
            self.spline_initialization_radio.value_selected == "Warm",
            visible=visible,
        )

    def set_warm_layout(self, warm: bool, *, visible: bool | None = None) -> None:
        if visible is None:
            visible = self.panel_axes["numerical"].get_visible()
        y_positions = (0.35, 0.295, 0.24, 0.185)
        if warm:
            spline_x, regression_x, width = 0.67, 0.82, 0.09
        else:
            spline_x, regression_x, width = 0.70, 0.80, 0.17
        for slider, y in zip(self.spline_numerical_sliders, y_positions):
            slider.ax.set_position([spline_x, y, width, 0.025])
            slider.ax.set_visible(bool(visible))
            slider.active = bool(visible)
            slider.label.set_visible(bool(visible and not warm))
            slider.label.set_fontsize(8.5 if warm else 10)
            slider.valtext.set_fontsize(8 if warm else 9.5)
        for slider, y in zip(self.regression_numerical_sliders, y_positions):
            slider.ax.set_position([regression_x, y, 0.09, 0.025])
            slider.ax.set_visible(bool(visible and warm))
            slider.active = bool(visible and warm)
            slider.label.set_visible(False)
            slider.valtext.set_fontsize(8)
        self.spline_numerical_heading.set_visible(bool(visible and warm))
        self.regression_numerical_heading.set_visible(bool(visible and warm))
        for label in self.numerical_row_labels:
            label.set_visible(bool(visible and warm))
        self.numerical_split_line.set_visible(bool(visible and warm))


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
        "PARAMETER",
        "Model, drawing, and numerical controls.  Press P or Esc to close.",
    )
    panels = {
        "model": build_panel(fig, [0.06, 0.14, 0.50, 0.70], "MODEL"),
        "draw": build_panel(fig, [0.60, 0.52, 0.34, 0.32], "DRAW"),
        "numerical": build_panel(fig, [0.60, 0.07, 0.34, 0.43], "NUMERICAL"),
    }
    panels["model"].text(
        0.25,
        0.58,
        r"$L v=-\alpha\Delta v-\beta\nabla(\nabla\!\cdot v)+\gamma v$"
        "\n" r"$K=L^{-1}$",
        transform=panels["model"].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK_COLOR,
    )
    panels["model"].text(
        0.75,
        0.58,
        r"$K=G_\sigma$" "\n" "Classic only",
        transform=panels["model"].transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=INK_COLOR,
    )
    panels["model"].plot(
        [0.5, 0.5],
        [0.35, 0.64],
        color="#c2cccf",
        linewidth=1,
        transform=panels["model"].transAxes,
    )
    panels["model"].text(
        0.5,
        0.14,
        r"CONTROL TIMES  $\tau_c$",
        transform=panels["model"].transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    panels["model"].text(
        0.5,
        0.01,
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

    model_radio = RadioButtons(
        fig.add_axes([0.12, 0.755, 0.40, 0.05], facecolor=PANEL_COLOR, zorder=102),
        ("Classic", "Spline"),
        active=0 if parameters.model == "classic" else 1,
        activecolor="#168a8a",
        layout=(1, 2),
    )
    for label in model_radio.labels:
        label.set_fontsize(11)
        label.set_fontweight("bold")
    model_radio.labels[0].set_position((0.13, 0.5))
    model_radio.labels[1].set_position((0.69, 0.5))
    model_buttons = getattr(model_radio, "_buttons", None)
    if model_buttons is not None:
        model_buttons.set_offsets(((0.09, 0.5), (0.65, 0.5)))
    else:
        model_radio.circles[0].center = (0.09, 0.5)
        model_radio.circles[1].center = (0.65, 0.5)
    rho = slider([0.17, 0.69, 0.32, 0.028], r"$\rho$", 0, max(0.95, parameters.rho), parameters.rho)
    operator_radio = RadioButtons(
        fig.add_axes([0.12, 0.565, 0.40, 0.045], facecolor=PANEL_COLOR, zorder=102),
        ("Sobolev", "Gaussian"),
        active=0 if parameters.kernel == "sobolev" else 1,
        activecolor="#168a8a",
        layout=(1, 2),
    )
    for label in operator_radio.labels:
        label.set_fontsize(9.5)
        label.set_fontweight("bold")
    operator_radio.labels[0].set_position((0.105, 0.5))
    operator_radio.labels[1].set_position((0.725, 0.5))
    buttons = getattr(operator_radio, "_buttons", None)
    if buttons is not None:
        buttons.set_offsets(((0.08, 0.5), (0.70, 0.5)))
    else:
        operator_radio.circles[0].center = (0.08, 0.5)
        operator_radio.circles[1].center = (0.70, 0.5)
    alpha = slider([0.13, 0.49, 0.14, 0.025], r"$\alpha$", 0, max(2, 1.5 * parameters.alpha), parameters.alpha)
    beta = slider([0.13, 0.43, 0.14, 0.025], r"$\beta$", 0, max(2, 1.5 * parameters.beta), parameters.beta)
    log_gamma = float(np.log10(parameters.gamma))
    gamma = slider(
        [0.13, 0.37, 0.14, 0.025],
        r"$\gamma$",
        min(-5, np.floor(log_gamma) - 1),
        max(1, np.ceil(log_gamma) + 1),
        log_gamma,
    )
    gamma.valtext.set_text(f"{parameters.gamma:.3g}")
    sigma = slider(
        [0.38, 0.49, 0.14, 0.025],
        r"$\sigma$",
        0.1,
        max(10, 1.5 * parameters.sigma),
        parameters.sigma,
    )
    panels["model"].text(
        0.5,
        0.305,
        "OPTIMIZED INITIAL FIELDS\n(unchecked fields stay fixed)",
        transform=panels["model"].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    optimized_fields = CheckButtons(
        fig.add_axes([0.21, 0.285, 0.20, 0.04], facecolor=PANEL_COLOR, zorder=102),
        ("Momentum", "Acceleration", "Jerk"),
        tuple(
            name in parameters.optimized_fields
            for name in OPTIMIZABLE_INITIAL_FIELDS
        ),
        layout=(1, 3),
    )
    for label in optimized_fields.labels:
        label.set_fontsize(8.5)
    brush = slider([0.70, 0.72, 0.17, 0.025], "Brush", 1, 40, 3)
    amplitude = slider([0.70, 0.64, 0.17, 0.025], "Amplitude", 0.01, 4, 0.5)
    spacing = slider(
        [0.70, 0.56, 0.17, 0.025],
        "Spacing",
        1,
        24,
        DEFAULT_VECTOR_DISPLAY_SPACING,
        valstep=1,
        valfmt="%0.0f",
    )
    steps = slider(
        [0.70, 0.31, 0.17, 0.025],
        "steps",
        1,
        MAX_STEPS,
        parameters.n_steps,
        valstep=1,
        valfmt="%0.0f",
    )
    iterations = slider(
        [0.70, 0.25, 0.17, 0.025],
        "iterations",
        1,
        max(MAX_ITERATIONS, parameters.iterations),
        parameters.iterations,
        valstep=1,
        valfmt="%0.0f",
    )
    log_cost = float(np.log10(max(parameters.cost_cst, 1e-12)))
    cost = slider(
        [0.70, 0.37, 0.17, 0.025],
        r"$\log_{10}$ cost",
        min(-8, np.floor(log_cost) - 1),
        max(2, np.ceil(log_cost) + 1),
        log_cost,
    )
    cost.valtext.set_text(f"{parameters.cost_cst:.3g}")
    assert parameters.lbfgs_lr is not None
    log_lbfgs_lr = float(np.log10(parameters.lbfgs_lr))
    lbfgs_lr = slider(
        [0.70, 0.19, 0.17, 0.025],
        r"$\log_{10}$ LBFGS lr",
        min(-5, np.floor(log_lbfgs_lr) - 1),
        max(1, np.ceil(log_lbfgs_lr) + 1),
        log_lbfgs_lr,
    )
    lbfgs_lr.valtext.set_text(format_lbfgs_learning_rate(parameters.lbfgs_lr))
    assert parameters.regression_cost_cst is not None
    log_regression_cost = float(np.log10(parameters.regression_cost_cst))
    regression_cost = slider(
        [0.80, 0.37, 0.11, 0.025],
        r"$\log_{10}$ cost",
        min(-8, np.floor(log_regression_cost) - 1),
        max(2, np.ceil(log_regression_cost) + 1),
        log_regression_cost,
    )
    regression_cost.valtext.set_text(f"{parameters.regression_cost_cst:.3g}")
    assert parameters.regression_n_steps is not None
    regression_steps = slider(
        [0.80, 0.31, 0.11, 0.025],
        "steps",
        1,
        MAX_STEPS,
        parameters.regression_n_steps,
        valstep=1,
        valfmt="%0.0f",
    )
    assert parameters.regression_iterations is not None
    regression_iterations = slider(
        [0.80, 0.25, 0.11, 0.025],
        "iterations",
        1,
        max(MAX_ITERATIONS, parameters.regression_iterations),
        parameters.regression_iterations,
        valstep=1,
        valfmt="%0.0f",
    )
    assert parameters.regression_lbfgs_lr is not None
    log_regression_lbfgs_lr = float(np.log10(parameters.regression_lbfgs_lr))
    regression_lbfgs_lr = slider(
        [0.80, 0.19, 0.11, 0.025],
        r"$\log_{10}$ LBFGS lr",
        min(-5, np.floor(log_regression_lbfgs_lr) - 1),
        max(1, np.ceil(log_regression_lbfgs_lr) + 1),
        log_regression_lbfgs_lr,
    )
    regression_lbfgs_lr.valtext.set_text(
        format_lbfgs_learning_rate(parameters.regression_lbfgs_lr)
    )
    spline_numerical_heading = panels["numerical"].text(
        0.34,
        0.745,
        "SPLINE",
        transform=panels["numerical"].transAxes,
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    regression_numerical_heading = panels["numerical"].text(
        0.78,
        0.745,
        "GEODESIC REGRESSION",
        transform=panels["numerical"].transAxes,
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=INK_COLOR,
    )
    numerical_row_labels = tuple(
        panels["numerical"].text(
            0.025,
            y,
            label,
            transform=panels["numerical"].transAxes,
            ha="left",
            va="center",
            fontsize=8,
            color=INK_COLOR,
        )
        for y, label in zip(
            (0.680, 0.552, 0.424, 0.297),
            (r"$\log_{10}$ cost", "steps", "iterations", r"$\log_{10}$ lr"),
        )
    )
    numerical_split_line = panels["numerical"].plot(
        [0.56, 0.56],
        [0.27, 0.72],
        transform=panels["numerical"].transAxes,
        color="#c2cccf",
        linewidth=1,
    )[0]

    def configure_binary_radio(widget, title):
        widget.ax.text(
            0.5,
            0.82,
            title,
            transform=widget.ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=INK_COLOR,
        )
        for index, label in enumerate(widget.labels):
            label.set_fontsize(8.5)
            label.set_position(((0.16, 0.70)[index], 0.31))
        buttons = getattr(widget, "_buttons", None)
        if buttons is not None:
            buttons.set_offsets(((0.10, 0.31), (0.64, 0.31)))
        else:
            widget.circles[0].center = (0.10, 0.31)
            widget.circles[1].center = (0.64, 0.31)

    spline_initialization_radio = RadioButtons(
        fig.add_axes([0.63, 0.405, 0.135, 0.052], facecolor=PANEL_COLOR, zorder=102),
        ("Cold", "Warm"),
        active=0 if parameters.spline_initialization == "cold" else 1,
        activecolor="#168a8a",
        layout=(1, 2),
    )
    configure_binary_radio(spline_initialization_radio, "INITIALIZATION")
    panels["numerical"].text(
        0.5,
        0.17,
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
        fig.add_axes([0.70, 0.08, 0.17, 0.055], facecolor=PANEL_COLOR, zorder=102),
        device_names,
        active=0 if device.startswith("cuda") else len(device_names) - 1,
        activecolor="#168a8a",
    )
    for label in device_radio.labels:
        label.set_fontsize(9)
    control_axis = fig.add_axes(
        [0.16, 0.17, 0.34, 0.055],
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
        "CLOSE  [P]",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = ParameterMenu(
        backdrop,
        panels,
        model_radio,
        rho,
        operator_radio,
        alpha,
        beta,
        gamma,
        sigma,
        optimized_fields,
        brush,
        amplitude,
        spacing,
        steps,
        iterations,
        cost,
        lbfgs_lr,
        regression_steps,
        regression_iterations,
        regression_cost,
        regression_lbfgs_lr,
        spline_numerical_heading,
        regression_numerical_heading,
        numerical_row_labels,
        numerical_split_line,
        spline_initialization_radio,
        device_radio,
        control_editor,
        close,
    )
    menu.set_visible(False)
    return menu
