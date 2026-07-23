"""Shared Matplotlib modal helpers."""

from matplotlib.colors import to_rgba
from matplotlib.widgets import RadioButtons

from ..styles import INK_COLOR


def build_modal_backdrop(fig, title: str, subtitle: str):
    axis = fig.add_axes([0, 0, 1, 1], facecolor="#17262d", zorder=100)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.5,
        0.94,
        title,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=19,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.895,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#bed0d5",
        fontsize=10.5,
    )
    return axis


def build_panel(fig, position, title: str):
    axis = fig.add_axes(position, facecolor="#f7f7f2", zorder=101)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#91a3a9")
    axis.text(
        0.5,
        0.91,
        title,
        transform=axis.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=INK_COLOR,
    )
    return axis


def set_radio_visible(radio: RadioButtons, visible: bool) -> None:
    radio.ax.set_visible(visible)
    for label in radio.labels:
        label.set_visible(visible)
    buttons = getattr(radio, "_buttons", None)
    if buttons is not None:
        buttons.set_visible(visible)
    else:
        for circle in getattr(radio, "circles", ()):
            circle.set_visible(visible)


def set_radio_active_color(radio: RadioButtons, index: int, color: str) -> None:
    radio.activecolor = color
    buttons = getattr(radio, "_buttons", None)
    if buttons is not None:
        facecolors = buttons.get_facecolors()
        facecolors[:, 3] = 0
        facecolors[index] = to_rgba(color)
        buttons.set_facecolors(facecolors)
        return
    for circle_index, circle in enumerate(getattr(radio, "circles", ())):
        circle.set_facecolor(
            color if circle_index == index else radio.ax.get_facecolor()
        )
