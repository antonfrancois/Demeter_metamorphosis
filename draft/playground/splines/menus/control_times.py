"""Interactive normalized control-time editor."""

from collections.abc import Callable

from ..styles import DUAL_COLOR, INK_COLOR, PANEL_COLOR


class ControlTimeEditor:
    """Select control times and optionally edit their mesh-snapped positions."""

    def __init__(
        self,
        axis,
        *,
        n_steps: int,
        control_steps: tuple[int, ...],
        on_select: Callable[[int], None],
        on_add: Callable[[float], None] | None = None,
        on_move: Callable[[int, float], None] | None = None,
        on_remove: Callable[[int], None] | None = None,
        on_message: Callable[[str], None] | None = None,
        editable: bool = True,
    ) -> None:
        self.axis = axis
        self.n_steps = n_steps
        self.control_steps = tuple(control_steps)
        self.selected_index = 0
        self.on_add = on_add
        self.on_move = on_move
        self.on_remove = on_remove
        self.on_select = on_select
        self.on_message = on_message
        self.editable = editable
        self.active = False
        self._drag_index: int | None = None
        self._drag_step: int | None = None
        self._dirty = False
        self._marker_artists = []
        self._annotation = None
        canvas = axis.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        self._draw()

    def set_state(
        self,
        n_steps: int,
        control_steps: tuple[int, ...],
        selected_index: int,
    ) -> None:
        steps = tuple(control_steps)
        selected = min(
            max(int(selected_index), 0),
            max(0, len(steps) - 1),
        )
        if (
            self.n_steps == n_steps
            and self.control_steps == steps
            and self.selected_index == selected
            and self._drag_index is None
        ):
            return
        self.n_steps = n_steps
        self.control_steps = steps
        self.selected_index = selected
        self._drag_index = None
        self._drag_step = None
        if self.axis.get_visible():
            self._draw()
        else:
            self._dirty = True

    def set_visible(self, visible: bool) -> None:
        self.axis.set_visible(visible)
        self.active = visible
        if visible and self._dirty:
            self._draw()
        if not visible:
            self._dirty = self._dirty or self._drag_index is not None
            self._drag_index = None
            self._drag_step = None

    def _display_steps(self) -> tuple[int, ...]:
        if self._drag_index is None or self._drag_step is None:
            return self.control_steps
        steps = list(self.control_steps)
        steps[self._drag_index] = self._drag_step
        return tuple(steps)

    def _draw(self) -> None:
        visible = self.axis.get_visible()
        self.axis.clear()
        self.axis.set_visible(visible)
        self._dirty = False
        self._marker_artists = []
        self._annotation = None
        self.axis.set_facecolor(PANEL_COLOR)
        self.axis.set_xlim(-0.03, 1.03)
        self.axis.set_ylim(0, 1)
        self.axis.set_yticks([])
        self.axis.set_xticks([])
        for spine in self.axis.spines.values():
            spine.set_visible(False)
        self.axis.hlines(0.55, 0, 1, color="#aeb9bc", linewidth=2, zorder=1)
        for position, label in ((0, "0"), (1, "1")):
            self.axis.text(
                position,
                0.40,
                label,
                ha="center",
                va="top",
                fontsize=8,
                color="#63747a",
            )
        if self.n_steps > 1:
            self.axis.vlines(
                [step / self.n_steps for step in range(1, self.n_steps)],
                0.48,
                0.62,
                color="#cad2d3",
                linewidth=0.7,
                zorder=1,
            )
        for index, step in enumerate(self._display_steps()):
            selected = index == self.selected_index
            (artist,) = self.axis.plot(
                step / self.n_steps,
                0.55,
                marker="v",
                markersize=10 if selected else 8,
                markerfacecolor=DUAL_COLOR,
                markeredgecolor="white" if selected else INK_COLOR,
                markeredgewidth=1.5 if selected else 0.8,
                zorder=3,
            )
            self._marker_artists.append(artist)
        if self.control_steps:
            step = self._display_steps()[self.selected_index]
            self._annotation = self.axis.text(
                0.5,
                0.04,
                rf"$\tau={step}/{self.n_steps}={step / self.n_steps:.3f}$",
                transform=self.axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_COLOR,
            )
        else:
            self._annotation = self.axis.text(
                0.5,
                0.04,
                "No control times",
                transform=self.axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#63747a",
            )
        self.axis.figure.canvas.draw_idle()

    def _draw_drag_preview(self, index: int, step: int) -> None:
        if index >= len(self._marker_artists) or self._annotation is None:
            self._draw()
            return
        self._marker_artists[index].set_xdata([step / self.n_steps])
        self._annotation.set_text(
            rf"$\tau={step}/{self.n_steps}={step / self.n_steps:.3f}$"
        )
        self.axis.figure.canvas.draw_idle()

    def _nearest_index(self, event) -> int | None:
        if not self.control_steps or event.x is None:
            return None
        distances = []
        for step in self.control_steps:
            x, _ = self.axis.transData.transform((step / self.n_steps, 0.55))
            distances.append(abs(float(event.x) - float(x)))
        index = min(range(len(distances)), key=distances.__getitem__)
        return index if distances[index] <= 12 else None

    def _snapped_step(self, xdata: float | None) -> int | None:
        if xdata is None or self.n_steps < 2:
            return None
        return min(max(round(float(xdata) * self.n_steps), 1), self.n_steps - 1)

    def _bounded_drag_step(self, index: int, step: int) -> int:
        lower = self.control_steps[index - 1] + 1 if index else 1
        upper = (
            self.control_steps[index + 1] - 1
            if index + 1 < len(self.control_steps)
            else self.n_steps - 1
        )
        return min(max(step, lower), upper)

    def _on_press(self, event) -> None:
        if not self.active or event.inaxes is not self.axis:
            return
        nearest = self._nearest_index(event)
        if event.button == 3:
            if self.editable and nearest is not None and self.on_remove is not None:
                self.on_remove(nearest)
            return
        if event.button != 1:
            return
        if nearest is not None:
            self.selected_index = nearest
            self.on_select(nearest)
            if self.editable:
                self._drag_index = nearest
                self._drag_step = self.control_steps[nearest]
            self._draw()
            return
        if not self.editable:
            return
        step = self._snapped_step(event.xdata)
        if step is None:
            if self.on_message is not None:
                self.on_message("At least two steps are required for a control time.")
        elif step in self.control_steps:
            self.on_select(self.control_steps.index(step))
        elif self.on_add is not None:
            self.on_add(step / self.n_steps)

    def _on_motion(self, event) -> None:
        if (
            not self.active
            or not self.editable
            or self._drag_index is None
            or event.inaxes is not self.axis
        ):
            return
        step = self._snapped_step(event.xdata)
        if step is None:
            return
        step = self._bounded_drag_step(self._drag_index, step)
        if step != self._drag_step:
            self._drag_step = step
            self._draw_drag_preview(self._drag_index, step)

    def _on_release(self, event) -> None:
        if self._drag_index is None:
            return
        index = self._drag_index
        step = self._drag_step
        self._drag_index = None
        self._drag_step = None
        if (
            step is not None
            and step != self.control_steps[index]
            and self.on_move is not None
        ):
            self.on_move(index, step / self.n_steps)
        else:
            self._draw()
