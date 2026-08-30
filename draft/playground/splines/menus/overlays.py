"""Image and field display controls."""

from dataclasses import dataclass
from typing import Any

from matplotlib.widgets import Button, CheckButtons, RadioButtons

from ..core import SplineParameters
from ..styles import (
    CURRENT_IMAGE_LABELS,
    CURRENT_LABELS,
    DUAL_COLOR,
    INK_COLOR,
    INPUT_LABELS,
    PANEL_COLOR,
    PRIMAL_COLOR,
)
from .common import (
    build_close_button,
    build_modal_backdrop,
    build_panel,
    set_radio_visible,
    set_widgets_visible,
)
from .control_times import ControlTimeEditor


@dataclass
class OverlayMenu:
    backdrop_ax: Any
    panels: dict[str, Any]
    radios: dict[str, RadioButtons]
    checks: dict[str, CheckButtons]
    labels: dict[str, Any]
    control_times: ControlTimeEditor
    close_button: Button

    @property
    def axes(self) -> tuple[Any, ...]:
        return (
            self.backdrop_ax,
            *self.panels.values(),
            *(radio.ax for radio in self.radios.values()),
            *(check.ax for check in self.checks.values()),
            self.control_times.axis,
            self.close_button.ax,
        )

    @property
    def widgets(self) -> tuple[Any, ...]:
        return (*self.radios.values(), *self.checks.values(), self.close_button)

    def bind(
        self,
        *,
        on_input,
        on_input_image,
        on_current_image,
        on_image_mode,
        on_current_field,
        on_target_mode,
        on_target_loss,
    ) -> None:
        callbacks = {
            "input": on_input,
            "image_mode": on_image_mode,
            "current_field": on_current_field,
            "target_mode": on_target_mode,
        }
        for name, callback in callbacks.items():
            self.radios[name].on_clicked(callback)
        self.checks["input_image"].on_clicked(on_input_image)
        self.checks["current_image"].on_clicked(on_current_image)
        self.checks["target_loss"].on_clicked(on_target_loss)

    def set_visible(
        self,
        visible: bool,
        *,
        show_control_selector: bool,
        target_mode: str,
    ) -> None:
        set_widgets_visible(self.axes, self.widgets, visible)
        for radio in self.radios.values():
            set_radio_visible(radio, visible)
        self.set_control_selector_visible(visible and show_control_selector)
        self.set_target_controls_visible(visible, target_mode)

    def set_control_selector_visible(self, visible: bool) -> None:
        self.labels["control_time"].set_visible(visible)
        self.control_times.set_visible(visible)

    def set_target_controls_visible(self, visible: bool, target_mode: str) -> None:
        show = visible and target_mode == "Global loss"
        self.labels["target_options"].set_visible(show)
        self.checks["target_loss"].ax.set_visible(show)
        self.checks["target_loss"].active = show


def _heading(axis, x: float, y: float, text: str, color: str = INK_COLOR):
    return axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        color=color,
    )


def _radio(fig, position, labels, active, color) -> RadioButtons:
    return RadioButtons(
        fig.add_axes(position, facecolor=PANEL_COLOR, zorder=102),
        labels,
        active=active,
        activecolor=color,
    )


def _check(fig, position, labels, active) -> CheckButtons:
    check = CheckButtons(
        fig.add_axes(position, facecolor=PANEL_COLOR, zorder=102),
        labels,
        active,
    )
    for label in check.labels:
        label.set_fontsize(9)
    return check


def build_overlay_menu(
    fig,
    parameters: SplineParameters,
    *,
    on_control_select,
) -> OverlayMenu:
    panels = {
        "source": build_panel(fig, [0.04, 0.15, 0.28, 0.68], "SOURCE"),
        "current": build_panel(fig, [0.36, 0.15, 0.28, 0.68], "CURRENT"),
        "target": build_panel(fig, [0.68, 0.15, 0.28, 0.68], "TARGET / ERROR"),
    }
    _heading(panels["source"], 0.10, 0.72, "Editable field", DUAL_COLOR)
    control_label = _heading(
        panels["source"], 0.10, 0.40, r"Control field time  $\tau_c$"
    )
    _heading(panels["current"], 0.10, 0.72, "Base image")
    _heading(panels["current"], 0.10, 0.45, "Field overlay")
    _heading(panels["target"], 0.10, 0.85, "Display")
    target_options = _heading(panels["target"], 0.10, 0.48, "Global loss curves")

    radios = {
        "input": _radio(fig, [0.07, 0.44, 0.22, 0.18], INPUT_LABELS, 0, DUAL_COLOR),
        "image_mode": _radio(fig, [0.39, 0.49, 0.22, 0.13], CURRENT_IMAGE_LABELS, 0, "#168a8a"),
        "current_field": _radio(fig, [0.39, 0.18, 0.22, 0.25], CURRENT_LABELS, 5, PRIMAL_COLOR),
        "target_mode": _radio(fig, [0.71, 0.55, 0.22, 0.15], ("Target", "Absolute error", "Global loss"), 0, "#168a8a"),
    }
    checks = {
        "input_image": _check(fig, [0.07, 0.68, 0.22, 0.055], ("Show input image",), (True,)),
        "current_image": _check(fig, [0.39, 0.68, 0.22, 0.055], ("Show current image",), (True,)),
        "target_loss": _check(fig, [0.71, 0.29, 0.22, 0.16], ("Full loss", "Data loss", "Regularized cost"), (True, True, True)),
    }
    control_times = ControlTimeEditor(
        fig.add_axes([0.07, 0.27, 0.22, 0.105], facecolor=PANEL_COLOR, zorder=102),
        n_steps=parameters.spline.steps,
        control_steps=parameters.control_nodes,
        on_select=on_control_select,
        editable=False,
    )
    menu = OverlayMenu(
        build_modal_backdrop(
            fig,
            "VIEW MENU",
            "Choose the image and field shown in each panel. Press V or Esc to close.",
        ),
        panels,
        radios,
        checks,
        {"control_time": control_label, "target_options": target_options},
        control_times,
        build_close_button(fig, [0.42, 0.06, 0.16, 0.055], "CLOSE  [V]"),
    )
    menu.set_visible(False, show_control_selector=False, target_mode="Target")
    return menu
