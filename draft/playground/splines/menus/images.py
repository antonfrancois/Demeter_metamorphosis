"""Classic endpoint image actions."""

from dataclasses import dataclass
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
class ImageMenu:
    backdrop_ax: Any
    panel_ax: Any
    load_source_button: Button
    load_target_button: Button
    close_button: Button

    def set_visible(self, visible: bool) -> None:
        buttons = (
            self.load_source_button,
            self.load_target_button,
            self.close_button,
        )
        set_widgets_visible(
            (self.backdrop_ax, self.panel_ax, *(button.ax for button in buttons)),
            buttons,
            visible,
        )


def build_image_menu(fig) -> ImageMenu:
    backdrop = build_modal_backdrop(
        fig,
        "IMAGES",
        "Load the source and endpoint target for classic registration.",
    )
    panel = fig.add_axes([0.28, 0.28, 0.44, 0.46], facecolor=PANEL_COLOR, zorder=101)
    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(
        0.5,
        0.78,
        "CLASSIC IMAGES",
        transform=panel.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )

    load_source = build_action_button(
        fig, [0.32, 0.43, 0.17, 0.08], "LOAD SOURCE IMAGE"
    )
    load_target = build_action_button(
        fig, [0.51, 0.43, 0.17, 0.08], "LOAD TARGET IMAGE"
    )
    close = build_close_button(
        fig,
        [0.42, 0.14, 0.16, 0.055],
        "CLOSE  [I]",
    )
    menu = ImageMenu(
        backdrop,
        panel,
        load_source,
        load_target,
        close,
    )
    menu.set_visible(False)
    return menu
