"""Model, drawing, and solver controls."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import torch

from ...field_playground_core import DEFAULT_VECTOR_DISPLAY_SPACING
from ..core import OPTIMIZABLE_INITIAL_FIELDS, SplineParameters
from ..styles import INK_COLOR, PANEL_COLOR
from .common import (
    build_close_button,
    build_modal_backdrop,
    build_panel,
    set_radio_visible,
    set_widgets_visible,
)
from .control_times import ControlTimeEditor


MAX_STEPS = 60
MAX_ITERATIONS = 50
LOG_SLIDERS = {
    "gamma",
    "spline_cost",
    "spline_learning_rate",
    "regression_cost",
    "regression_learning_rate",
}


def format_learning_rate(value: float) -> str:
    if value >= 1e-1:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"


@dataclass
class ParameterMenu:
    backdrop_ax: Any
    panels: dict[str, Any]
    sliders: dict[str, Slider]
    radios: dict[str, RadioButtons]
    optimized_fields: CheckButtons
    control_times: ControlTimeEditor
    close_button: Button

    @property
    def axes(self) -> tuple[Any, ...]:
        return (
            self.backdrop_ax,
            *self.panels.values(),
            *(slider.ax for slider in self.sliders.values()),
            *(radio.ax for radio in self.radios.values()),
            self.optimized_fields.ax,
            self.control_times.axis,
            self.close_button.ax,
        )

    @property
    def widgets(self) -> tuple[Any, ...]:
        return (
            *self.sliders.values(),
            *self.radios.values(),
            self.optimized_fields,
            self.close_button,
        )

    def set_visible(self, visible: bool) -> None:
        set_widgets_visible(self.axes, self.widgets, visible)
        for radio in self.radios.values():
            set_radio_visible(radio, visible)
        self.control_times.set_visible(visible)

    def read(self, base: SplineParameters) -> SplineParameters:
        value = lambda name: float(self.sliders[name].val)
        integer = lambda name: round(value(name))
        logarithmic = lambda name: 10 ** value(name)
        return replace(
            base,
            alpha=value("alpha"),
            beta=value("beta"),
            gamma=logarithmic("gamma"),
            rho=value("rho"),
            sigma=value("sigma"),
            kernel=self.radios["kernel"].value_selected.lower(),
            model=(
                "splines"
                if self.radios["model"].value_selected == "Spline"
                else "classic"
            ),
            optimized_fields=tuple(
                name
                for name, active in zip(
                    OPTIMIZABLE_INITIAL_FIELDS,
                    self.optimized_fields.get_status(),
                )
                if active
            ),
            initialization=self.radios["initialization"].value_selected.lower(),
            spline=replace(
                base.spline,
                cost=logarithmic("spline_cost"),
                steps=integer("spline_steps"),
                iterations=integer("spline_iterations"),
                learning_rate=logarithmic("spline_learning_rate"),
            ),
            regression=replace(
                base.regression,
                cost=logarithmic("regression_cost"),
                steps=integer("regression_steps"),
                iterations=integer("regression_iterations"),
                learning_rate=logarithmic("regression_learning_rate"),
            ),
        )

    def sync(self, parameters: SplineParameters) -> None:
        values = {
            "alpha": parameters.alpha,
            "beta": parameters.beta,
            "gamma": parameters.gamma,
            "rho": parameters.rho,
            "sigma": parameters.sigma,
            "spline_cost": parameters.spline.cost,
            "spline_steps": parameters.spline.steps,
            "spline_iterations": parameters.spline.iterations,
            "spline_learning_rate": parameters.spline.learning_rate,
            "regression_cost": parameters.regression.cost,
            "regression_steps": parameters.regression.steps,
            "regression_iterations": parameters.regression.iterations,
            "regression_learning_rate": parameters.regression.learning_rate,
        }
        for name, value in values.items():
            slider = self.sliders[name]
            displayed = np.log10(value) if name in LOG_SLIDERS else value
            padding = 1 if name in LOG_SLIDERS else max(1, abs(displayed) * 0.5)
            slider.valmin = min(slider.valmin, displayed - padding)
            slider.valmax = max(slider.valmax, displayed + padding)
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            slider.set_val(displayed)
            self._format(name)
        self.radios["model"].set_active(0 if parameters.model == "classic" else 1)
        self.radios["kernel"].set_active(0 if parameters.kernel == "sobolev" else 1)
        self.radios["initialization"].set_active(
            0 if parameters.initialization == "cold" else 1
        )
        selected = set(parameters.optimized_fields)
        for index, (name, active) in enumerate(
            zip(OPTIMIZABLE_INITIAL_FIELDS, self.optimized_fields.get_status())
        ):
            if active != (name in selected):
                self.optimized_fields.set_active(index)

    def bind(
        self,
        *,
        on_parameter,
        on_model,
        on_kernel,
        on_initialization,
        on_amplitude,
        on_spacing,
        on_device,
    ) -> None:
        for name, slider in self.sliders.items():
            callback = {
                "amplitude": on_amplitude,
                "spacing": on_spacing,
                "brush": lambda _value: None,
            }.get(name, on_parameter)
            slider.on_changed(partial(self._changed, name, callback))
        self.radios["model"].on_clicked(on_model)
        self.radios["kernel"].on_clicked(on_kernel)
        self.radios["initialization"].on_clicked(on_initialization)
        self.radios["device"].on_clicked(on_device)
        self.optimized_fields.on_clicked(lambda _label: on_parameter(0.0))

    def _changed(self, name: str, callback, value: float) -> None:
        self._format(name)
        callback(value)

    def _format(self, name: str) -> None:
        if name not in LOG_SLIDERS:
            return
        value = 10 ** float(self.sliders[name].val)
        formatter = (
            format_learning_rate
            if name.endswith("learning_rate")
            else lambda x: f"{x:.3g}"
        )
        self.sliders[name].valtext.set_text(formatter(value))


def _slider(fig, position, label, low, high, value, **kwargs) -> Slider:
    slider = Slider(
        fig.add_axes(position, facecolor=PANEL_COLOR, zorder=102),
        label,
        low,
        high,
        valinit=value,
        **kwargs,
    )
    slider.label.set_x(-0.10)
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(8.5)
    return slider


def _log_slider(fig, position, label, value, low=-8, high=2) -> Slider:
    logarithm = float(np.log10(value))
    return _slider(
        fig,
        position,
        label,
        min(low, np.floor(logarithm) - 1),
        max(high, np.ceil(logarithm) + 1),
        logarithm,
    )


def _radio(fig, position, labels, active=0) -> RadioButtons:
    radio = RadioButtons(
        fig.add_axes(position, facecolor=PANEL_COLOR, zorder=102),
        labels,
        active=active,
        activecolor="#168a8a",
        layout=(1, len(labels)),
    )
    for label in radio.labels:
        label.set_fontsize(9)
    return radio


def _panel_text(axis, x: float, y: float, text: str, **style) -> None:
    style.setdefault("ha", "center")
    style.setdefault("va", "center")
    style.setdefault("fontsize", 8.5)
    axis.text(x, y, text, transform=axis.transAxes, color=INK_COLOR, **style)


def _annotate_panels(panels: dict[str, Any]) -> None:
    model = panels["model"]
    _panel_text(model, 0.25, 0.58, r"$L v=-\alpha\Delta v-\beta\nabla(\nabla\!\cdot v)+\gamma v$\n$K=L^{-1}$")
    _panel_text(model, 0.75, 0.58, r"$K=G_\sigma$" "\nClassic only", fontsize=9)
    _panel_text(model, 0.5, 0.305, "OPTIMIZED INITIAL FIELDS", fontweight="bold")
    _panel_text(model, 0.5, 0.14, r"CONTROL TIMES  $\tau_c$", fontsize=9.5, fontweight="bold")
    _panel_text(model, 0.5, 0.01, "Left-click add/select, drag move, right-click remove")

    solver = panels["solver"]
    for x, label in ((0.36, "SPLINE"), (0.80, "WARM START")):
        _panel_text(solver, x, 0.78, label, fontweight="bold")
    for y, label in zip((0.69, 0.56, 0.43, 0.30), ("cost", "steps", "iterations", "learning rate")):
        _panel_text(solver, 0.03, y, label, ha="left", fontsize=8)
    _panel_text(solver, 0.5, 0.17, "COMPUTE DEVICE", fontsize=9, fontweight="bold")


def _build_radios(fig, parameters: SplineParameters, device: str) -> dict[str, RadioButtons]:
    device_names = ("CUDA", "CPU") if torch.cuda.is_available() or device.startswith("cuda") else ("CPU",)
    specs = {
        "model": ([0.12, 0.755, 0.40, 0.05], ("Classic", "Spline"), int(parameters.model != "classic")),
        "kernel": ([0.12, 0.565, 0.40, 0.045], ("Sobolev", "Gaussian"), int(parameters.kernel != "sobolev")),
        "initialization": ([0.63, 0.405, 0.135, 0.052], ("Cold", "Warm"), int(parameters.initialization != "cold")),
        "device": ([0.70, 0.08, 0.17, 0.055], device_names, device_names.index("CUDA" if device.startswith("cuda") else "CPU")),
    }
    return {name: _radio(fig, *spec) for name, spec in specs.items()}


def _build_sliders(fig, parameters: SplineParameters) -> dict[str, Slider]:
    specs = {
        "rho": (_slider, [0.17, 0.69, 0.32, 0.028], r"$\rho$", 0, max(0.95, parameters.rho), parameters.rho),
        "alpha": (_slider, [0.13, 0.49, 0.14, 0.025], r"$\alpha$", 0, max(2, 1.5 * parameters.alpha), parameters.alpha),
        "beta": (_slider, [0.13, 0.43, 0.14, 0.025], r"$\beta$", 0, max(2, 1.5 * parameters.beta), parameters.beta),
        "gamma": (_log_slider, [0.13, 0.37, 0.14, 0.025], r"$\gamma$", parameters.gamma, -5, 1),
        "sigma": (_slider, [0.38, 0.49, 0.14, 0.025], r"$\sigma$", 0.1, max(10, 1.5 * parameters.sigma), parameters.sigma),
        "brush": (_slider, [0.70, 0.72, 0.17, 0.025], "Brush", 1, 40, 3),
        "amplitude": (_slider, [0.70, 0.64, 0.17, 0.025], "Amplitude", 0.01, 4, 0.5),
    }
    sliders = {name: factory(fig, *args) for name, (factory, *args) in specs.items()}
    integer = {"valstep": 1, "valfmt": "%0.0f"}
    sliders["spacing"] = _slider(fig, [0.70, 0.56, 0.17, 0.025], "Spacing", 1, 24, DEFAULT_VECTOR_DISPLAY_SPACING, **integer)
    rows = {"cost": 0.37, "steps": 0.31, "iterations": 0.25, "learning_rate": 0.19}
    for x, prefix, settings in (
        (0.69, "spline", parameters.spline),
        (0.84, "regression", parameters.regression),
    ):
        position = lambda name: [x, rows[name], 0.08, 0.025]
        sliders[f"{prefix}_cost"] = _log_slider(fig, position("cost"), "", settings.cost)
        sliders[f"{prefix}_steps"] = _slider(fig, position("steps"), "", 1, MAX_STEPS, settings.steps, **integer)
        sliders[f"{prefix}_iterations"] = _slider(fig, position("iterations"), "", 1, max(MAX_ITERATIONS, settings.iterations), settings.iterations, **integer)
        sliders[f"{prefix}_learning_rate"] = _log_slider(fig, position("learning_rate"), "", settings.learning_rate, -5, 1)
    return sliders


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
        "Model, drawing, and solver controls. Press P or Esc to close.",
    )
    panels = {
        "model": build_panel(fig, [0.06, 0.14, 0.50, 0.70], "MODEL"),
        "draw": build_panel(fig, [0.60, 0.52, 0.34, 0.32], "DRAW"),
        "solver": build_panel(fig, [0.60, 0.07, 0.34, 0.43], "SOLVERS"),
    }
    _annotate_panels(panels)
    radios = _build_radios(fig, parameters, device)
    sliders = _build_sliders(fig, parameters)
    optimized = CheckButtons(
        fig.add_axes([0.21, 0.285, 0.20, 0.04], facecolor=PANEL_COLOR, zorder=102),
        ("Momentum", "Acceleration", "Jerk"),
        tuple(
            name in parameters.optimized_fields for name in OPTIMIZABLE_INITIAL_FIELDS
        ),
        layout=(1, 3),
    )
    for label in optimized.labels:
        label.set_fontsize(8.5)

    control_times = ControlTimeEditor(
        fig.add_axes([0.16, 0.17, 0.34, 0.055], facecolor=PANEL_COLOR, zorder=102),
        n_steps=parameters.spline.steps,
        control_steps=parameters.control_nodes,
        on_add=on_control_add,
        on_move=on_control_move,
        on_remove=on_control_remove,
        on_select=on_control_select,
        on_message=on_message,
    )
    menu = ParameterMenu(
        backdrop,
        panels,
        sliders,
        radios,
        optimized,
        control_times,
        build_close_button(fig, [0.42, 0.02, 0.16, 0.045], "CLOSE  [P]"),
    )
    for name in LOG_SLIDERS:
        menu._format(name)
    menu.set_visible(False)
    return menu
