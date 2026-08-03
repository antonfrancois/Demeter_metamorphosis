"""Model-aware image actions."""

from dataclasses import dataclass
from typing import Any

from matplotlib.widgets import Button

from ..styles import INK_COLOR, PANEL_COLOR
from .common import build_modal_backdrop


@dataclass
class ImageMenu:
    backdrop_ax: Any
    panel_ax: Any
    heading: Any
    load_source_button: Button
    load_target_button: Button
    manage_spline_button: Button
    close_button: Button

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            self.panel_ax,
            self.load_source_button.ax,
            self.load_target_button.ax,
            self.manage_spline_button.ax,
            self.close_button.ax,
        ]

    @property
    def widgets(self) -> list[Button]:
        return [
            self.load_source_button,
            self.load_target_button,
            self.manage_spline_button,
            self.close_button,
        ]

    def set_visible(self, visible: bool, model: str) -> None:
        self.backdrop_ax.set_visible(visible)
        self.panel_ax.set_visible(visible)
        self.close_button.ax.set_visible(visible)
        self.close_button.active = visible
        classic = visible and model == "classic"
        spline = visible and model == "splines"
        self.heading.set_text("CLASSIC IMAGES" if model == "classic" else "SPLINE IMAGES")
        for button in (self.load_source_button, self.load_target_button):
            button.ax.set_visible(classic)
            button.active = classic
        self.manage_spline_button.ax.set_visible(spline)
        self.manage_spline_button.active = spline


def build_image_menu(fig) -> ImageMenu:
    backdrop = build_modal_backdrop(
        fig,
        "IMAGES",
        "Image actions follow the model selected under Parameter.",
    )
    panel = fig.add_axes([0.28, 0.28, 0.44, 0.46], facecolor=PANEL_COLOR, zorder=101)
    panel.set_xticks([])
    panel.set_yticks([])
    heading = panel.text(
        0.5,
        0.78,
        "",
        transform=panel.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )

    def button(position, label):
        widget = Button(
            fig.add_axes(position, zorder=102),
            label,
            color="#edf3f2",
            hovercolor="#d7e4e2",
        )
        widget.label.set_fontsize(9)
        return widget

    load_source = button([0.32, 0.43, 0.17, 0.08], "LOAD SOURCE IMAGE")
    load_target = button([0.51, 0.43, 0.17, 0.08], "LOAD TARGET IMAGE")
    manage_spline = button([0.39, 0.41, 0.22, 0.10], "MANAGE SPLINE IMAGES")
    close = Button(
        fig.add_axes([0.42, 0.14, 0.16, 0.055], zorder=102),
        "RETURN  [I]",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = ImageMenu(
        backdrop,
        panel,
        heading,
        load_source,
        load_target,
        manage_spline,
        close,
    )
    menu.set_visible(False, "splines")
    return menu
