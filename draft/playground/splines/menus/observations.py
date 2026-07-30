"""Timed spline-image placement menu."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.widgets import Button

from ..styles import INK_COLOR, PANEL_COLOR, TARGET_COLOR
from .common import build_modal_backdrop


class ObservationTimeEditor:
    """Select image rows and place them on endpoint-inclusive mesh nodes."""

    def __init__(self, axis, *, on_select, on_place, on_unplace) -> None:
        self.axis = axis
        self.on_select = on_select
        self.on_place = on_place
        self.on_unplace = on_unplace
        self.n_steps = 1
        self.names: tuple[str, ...] = ()
        self.times: tuple[float | None, ...] = ()
        self.selected = 0
        self._connection = axis.figure.canvas.mpl_connect(
            "button_press_event", self._on_press
        )

    def set_state(
        self,
        n_steps: int,
        names: tuple[str, ...],
        times: tuple[float | None, ...],
        selected: int,
    ) -> None:
        self.n_steps = n_steps
        self.names = names
        self.times = times
        self.selected = min(max(selected, 0), max(0, len(names) - 1))
        self.draw()

    def set_visible(self, visible: bool) -> None:
        self.axis.set_visible(visible)

    def draw(self) -> None:
        axis = self.axis
        axis.clear()
        axis.set_facecolor(PANEL_COLOR)
        count = max(1, len(self.names))
        label_width = max(2.5, 0.38 * self.n_steps)
        axis.set_xlim(-label_width, self.n_steps + 0.4)
        axis.set_ylim(count - 0.5, -0.5)
        axis.set_xlabel("Observation node", color=INK_COLOR, fontsize=9)
        axis.set_xticks(range(0, self.n_steps + 1))
        axis.tick_params(axis="x", labelsize=7, colors="#63747a")
        axis.set_yticks([])
        axis.grid(axis="x", color="#d2dcde", linewidth=0.6)
        if not self.names:
            axis.text(
                0.5,
                0.5,
                "Add images, then place each one on the timeline.",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#63747a",
            )
            return

        paths = tuple(Path(name) for name in self.names)
        basename_counts = Counter(path.name for path in paths)
        for index, (path, time) in enumerate(zip(paths, self.times)):
            if index == self.selected:
                axis.axhspan(index - 0.42, index + 0.42, color="#dcebea", zorder=0)
            status = "[S]" if time == 0 else "[x]" if time is not None else "[ ]"
            name = path.name
            if basename_counts[name] > 1 and path.parent.name:
                name = f"{name} ({path.parent.name})"
            else:
                name = name[:28]
            axis.text(
                -label_width + 0.1,
                index,
                f"{status} {name}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK_COLOR,
                fontweight="bold" if index == self.selected else "normal",
            )
            if time is not None:
                step = round(time * self.n_steps)
                axis.plot(
                    step,
                    index,
                    marker="D",
                    markersize=7,
                    color=TARGET_COLOR,
                    zorder=4,
                )
                axis.text(
                    step,
                    index - 0.27,
                    f"t={time:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=TARGET_COLOR,
                )

    def _on_press(self, event) -> None:
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return
        if not self.names:
            return
        row = min(max(round(event.ydata), 0), len(self.names) - 1)
        self.on_select(row)
        if event.button == 3:
            self.on_unplace(row)
            return
        if event.button != 1 or event.xdata < 0:
            return
        step = min(max(round(event.xdata), 0), self.n_steps)
        self.on_place(row, step / self.n_steps)


@dataclass
class ObservationMenu:
    backdrop_ax: Any
    panel_ax: Any
    editor: ObservationTimeEditor
    load_directory_button: Button
    add_images_button: Button
    remove_button: Button
    close_button: Button

    @property
    def widgets(self) -> list[Button]:
        return [
            self.load_directory_button,
            self.add_images_button,
            self.remove_button,
            self.close_button,
        ]

    @property
    def axes(self) -> list[Any]:
        return [
            self.backdrop_ax,
            self.panel_ax,
            self.editor.axis,
            *(widget.ax for widget in self.widgets),
        ]

    def set_visible(self, visible: bool) -> None:
        for axis in self.axes:
            axis.set_visible(visible)
        for widget in self.widgets:
            widget.active = visible
        self.editor.set_visible(visible)


def build_observation_menu(
    fig,
    *,
    on_select,
    on_place,
    on_unplace,
) -> ObservationMenu:
    backdrop = build_modal_backdrop(
        fig,
        "SPLINE IMAGES",
        "Place one source at node 0 and every observation on a later mesh node.",
    )
    panel = fig.add_axes([0.07, 0.15, 0.86, 0.69], facecolor=PANEL_COLOR, zorder=101)
    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(
        0.5,
        0.92,
        "IMAGE TRAJECTORY",
        transform=panel.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    editor_axis = fig.add_axes([0.12, 0.31, 0.76, 0.36], zorder=102)
    editor = ObservationTimeEditor(
        editor_axis,
        on_select=on_select,
        on_place=on_place,
        on_unplace=on_unplace,
    )

    def button(position, label):
        widget = Button(
            fig.add_axes(position, zorder=102),
            label,
            color="#edf3f2",
            hovercolor="#d7e4e2",
        )
        widget.label.set_fontsize(8.5)
        return widget

    load_directory = button([0.18, 0.20, 0.19, 0.06], "LOAD TIMED DIRECTORY")
    add_images = button([0.405, 0.20, 0.19, 0.06], "ADD IMAGES")
    remove = button([0.63, 0.20, 0.19, 0.06], "REMOVE SELECTED IMAGE")
    close = Button(
        fig.add_axes([0.42, 0.07, 0.16, 0.055], zorder=102),
        "RETURN TO IMAGES",
        color="#168a8a",
        hovercolor="#20a3a3",
    )
    close.label.set_color("white")
    menu = ObservationMenu(
        backdrop,
        panel,
        editor,
        load_directory,
        add_images,
        remove,
        close,
    )
    menu.set_visible(False)
    return menu
