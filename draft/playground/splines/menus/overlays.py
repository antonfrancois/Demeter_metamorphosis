"""Three-column image and field-overlay menu."""

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
from .common import build_modal_backdrop, build_panel, set_radio_visible
from .control_times import ControlTimeEditor


@dataclass
class OverlayMenu:
    backdrop_ax: Any
    column_axes: dict[str, Any]
    input_radio: RadioButtons
    control_time_label: Any
    control_time_selector: ControlTimeEditor
    input_image_toggle: CheckButtons
    current_image_toggle: CheckButtons
    current_image_radio: RadioButtons
    current_radio: RadioButtons
    target_radio: RadioButtons
    close_button: Button

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            *self.column_axes.values(),
            self.input_radio.ax,
            self.control_time_selector.axis,
            self.input_image_toggle.ax,
            self.current_image_toggle.ax,
            self.current_image_radio.ax,
            self.current_radio.ax,
            self.target_radio.ax,
            self.close_button.ax,
        ]

    @property
    def radios(self) -> tuple[RadioButtons, ...]:
        return (
            self.input_radio,
            self.current_image_radio,
            self.current_radio,
            self.target_radio,
        )

    @property
    def widgets(self) -> list[Any]:
        return [
            self.input_radio,
            self.input_image_toggle,
            self.current_image_toggle,
            self.current_image_radio,
            self.current_radio,
            self.target_radio,
            self.close_button,
        ]

    def set_visible(self, visible: bool, *, show_control_selector: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)
        for radio in self.radios:
            set_radio_visible(radio, visible)
        for widget in self.widgets:
            widget.active = visible
        self.set_control_selector_visible(visible and show_control_selector)

    def set_control_selector_visible(self, visible: bool) -> None:
        self.control_time_label.set_visible(visible)
        self.control_time_selector.set_visible(visible)


def build_overlay_menu(
    fig,
    parameters: SplineParameters,
    *,
    on_control_select,
) -> OverlayMenu:
    backdrop = build_modal_backdrop(
        fig,
        "VIEW",
        "Choose the base image and field shown in each panel.  Press M or Esc to close.",
    )
    columns = {
        "source": build_panel(fig, [0.04, 0.15, 0.28, 0.68], "SOURCE"),
        "current": build_panel(fig, [0.36, 0.15, 0.28, 0.68], "CURRENT"),
        "target": build_panel(fig, [0.68, 0.15, 0.28, 0.68], "TARGET / ERROR"),
    }
    columns["source"].text(
        0.10,
        0.72,
        "Editable field",
        transform=columns["source"].transAxes,
        fontsize=9,
        fontweight="bold",
        color=DUAL_COLOR,
    )
    control_time_label = columns["source"].text(
        0.10,
        0.40,
        r"Control field time  $\tau_c$",
        transform=columns["source"].transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK_COLOR,
    )
    columns["current"].text(
        0.10,
        0.72,
        "Base image",
        transform=columns["current"].transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK_COLOR,
    )
    columns["current"].text(
        0.10,
        0.45,
        "Field overlay",
        transform=columns["current"].transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK_COLOR,
    )
    columns["target"].text(
        0.10,
        0.85,
        "Display",
        transform=columns["target"].transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK_COLOR,
    )

    input_radio = RadioButtons(
        fig.add_axes([0.07, 0.44, 0.22, 0.18], facecolor=PANEL_COLOR, zorder=102),
        INPUT_LABELS,
        active=0,
        activecolor=DUAL_COLOR,
    )
    input_image_toggle = CheckButtons(
        fig.add_axes([0.07, 0.68, 0.22, 0.055], facecolor=PANEL_COLOR, zorder=102),
        ("Show input image",),
        (True,),
    )
    input_image_toggle.labels[0].set_fontsize(9)
    control_time_selector = ControlTimeEditor(
        fig.add_axes([0.07, 0.27, 0.22, 0.105], facecolor=PANEL_COLOR, zorder=102),
        n_steps=parameters.n_steps,
        control_steps=parameters.control_steps,
        on_select=on_control_select,
        editable=False,
    )
    current_image_radio = RadioButtons(
        fig.add_axes([0.39, 0.49, 0.22, 0.13], facecolor=PANEL_COLOR, zorder=102),
        CURRENT_IMAGE_LABELS,
        active=0,
        activecolor="#168a8a",
    )
    current_image_toggle = CheckButtons(
        fig.add_axes([0.39, 0.68, 0.22, 0.055], facecolor=PANEL_COLOR, zorder=102),
        ("Show current image",),
        (True,),
    )
    current_image_toggle.labels[0].set_fontsize(9)
    current_radio = RadioButtons(
        fig.add_axes([0.39, 0.18, 0.22, 0.25], facecolor=PANEL_COLOR, zorder=102),
        CURRENT_LABELS,
        active=5,
        activecolor=PRIMAL_COLOR,
    )
    target_radio = RadioButtons(
        fig.add_axes([0.71, 0.52, 0.22, 0.13], facecolor=PANEL_COLOR, zorder=102),
        ("Target", "Absolute error"),
        active=0,
        activecolor="#168a8a",
    )
    close = Button(
        fig.add_axes([0.42, 0.06, 0.16, 0.055], zorder=102),
        "CLOSE  [M]",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = OverlayMenu(
        backdrop,
        columns,
        input_radio,
        control_time_label,
        control_time_selector,
        input_image_toggle,
        current_image_toggle,
        current_image_radio,
        current_radio,
        target_radio,
        close,
    )
    menu.set_visible(False, show_control_selector=False)
    return menu
