"""Load and save action menu."""

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from matplotlib.widgets import Button

from ..styles import INK_COLOR, PANEL_COLOR
from .common import (
    build_action_button,
    build_close_button,
    build_modal_backdrop,
    set_widgets_visible,
)


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
        set_widgets_visible(self.axes, self.widgets, visible)


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
        button = build_action_button(fig, position, label)
        button.on_clicked(
            lambda _event, callback=action: callback()
        )
        buttons.append(button)
    close = build_close_button(
        fig,
        [0.42, 0.10, 0.16, 0.055],
        "CLOSE  [L]",
    )
    menu = FileMenu(backdrop, panel, buttons, close)
    menu.set_visible(False)
    return menu
