"""Load and save action menu."""

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from matplotlib.widgets import Button

from ..styles import INK_COLOR, PANEL_COLOR
from .common import build_modal_backdrop


@dataclass
class FileMenu:
    backdrop_ax: Any
    panel_ax: Any
    buttons: list[Button]
    close_button: Button

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            self.panel_ax,
            *(button.ax for button in self.buttons),
            self.close_button.ax,
        ]

    @property
    def widgets(self) -> list[Button]:
        return [*self.buttons, self.close_button]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)
        for widget in self.widgets:
            widget.active = visible


def build_file_menu(
    fig,
    actions: tuple[tuple[str, Callable[[], None]], ...],
) -> FileMenu:
    backdrop = build_modal_backdrop(
        fig,
        "LOAD / SAVE",
        "Load or save fields, projects, and trajectory video.",
    )
    panel = fig.add_axes([0.28, 0.20, 0.44, 0.62], facecolor=PANEL_COLOR, zorder=101)
    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(
        0.5,
        0.88,
        "PROJECT FILES",
        transform=panel.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    buttons = []
    for index, (label, action) in enumerate(actions):
        row, column = divmod(index, 2)
        left = (
            0.42
            if index == len(actions) - 1 and len(actions) % 2
            else 0.33 + 0.18 * column
        )
        position = [left, 0.60 - 0.13 * row, 0.16, 0.075]
        button = Button(
            fig.add_axes(position, zorder=102),
            label,
            color="#edf3f2",
            hovercolor="#d7e4e2",
        )
        button.label.set_fontsize(9)
        button.on_clicked(
            lambda _event, callback=action: callback()
        )
        buttons.append(button)
    close = Button(
        fig.add_axes([0.42, 0.10, 0.16, 0.055], zorder=102),
        "CLOSE  [L]",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = FileMenu(backdrop, panel, buttons, close)
    menu.set_visible(False)
    return menu
