"""Reusable scalar-field painting interaction for the spline playground."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

from ..field_playground_core import erase_stroke, paint_scalar_stroke


UNDO_LIMIT = 16


@dataclass
class _Stroke:
    points: list[tuple[float, float]]
    button: int
    erase: bool
    sigma: float
    amplitude: float
    artist: Any = None


class ScalarFieldEditor:
    """Paint whichever scalar tensor is selected in a shared field dictionary."""

    def __init__(
        self,
        figure,
        axis,
        fields: dict[str, torch.Tensor],
        *,
        active_key: str,
        brush: Callable[[], float],
        amplitude: Callable[[], float],
        color: str,
        on_change: Callable[[str], None],
    ) -> None:
        self.figure = figure
        self.axis = axis
        self.fields = fields
        self.active_key = active_key
        self.brush = brush
        self.amplitude = amplitude
        self.color = color
        self.on_change = on_change
        self.enabled = True
        self.history: dict[str, deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=UNDO_LIMIT)
        )
        self.stroke: _Stroke | None = None
        self._last_preview_draw = 0.0
        canvas = figure.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)

    @property
    def field(self) -> torch.Tensor:
        return self.fields[self.active_key]

    def set_active(self, key: str) -> None:
        if key not in self.fields:
            raise KeyError(key)
        self.cancel()
        self.active_key = key

    def replace(self, value: torch.Tensor) -> None:
        if value.shape != self.field.shape or value.shape[:2] != (1, 1):
            raise ValueError(
                f"field must have shape {tuple(self.field.shape)}, got {tuple(value.shape)}"
            )
        self.cancel()
        self._commit(
            self.active_key,
            value.detach().cpu().to(self.field).contiguous(),
        )
        self.on_change("Field replaced. Press Run.")

    def clear(self) -> bool:
        self.cancel()
        if torch.count_nonzero(self.field) == 0:
            return False
        self._commit(self.active_key, torch.zeros_like(self.field))
        self.on_change("Field cleared. Press Run.")
        return True

    def clear_all(self) -> bool:
        self.cancel()
        changed = False
        for key, field in tuple(self.fields.items()):
            if torch.count_nonzero(field) == 0:
                continue
            self._commit(key, torch.zeros_like(field))
            changed = True
        if changed:
            self.on_change("All fields cleared. Press Run.")
        return changed

    def undo(self) -> bool:
        self.cancel()
        history = self.history[self.active_key]
        if not history:
            return False
        self.fields[self.active_key] = history.pop()
        self.on_change("Undo applied. Press Run.")
        return True

    def clear_history(self) -> None:
        self.history.clear()

    def _commit(self, key: str, value: torch.Tensor) -> None:
        self.history[key].append(self.fields[key])
        self.fields[key] = value

    def _toolbar_is_active(self) -> bool:
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return bool(toolbar is not None and getattr(toolbar, "mode", ""))

    def on_press(self, event) -> None:
        if (
            not self.enabled
            or event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
            or event.button not in (1, 3)
            or self._toolbar_is_active()
        ):
            return
        self.cancel()
        self._last_preview_draw = 0.0
        point = (float(event.xdata), float(event.ydata))
        erase = "shift" in (event.key or "").lower()
        self.stroke = _Stroke(
            points=[point],
            button=int(event.button),
            erase=erase,
            sigma=float(self.brush()),
            amplitude=float(self.amplitude()),
        )

    def on_motion(self, event) -> None:
        stroke = self.stroke
        if (
            stroke is None
            or event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        point = (float(event.xdata), float(event.ydata))
        previous = stroke.points[-1]
        if np.hypot(point[0] - previous[0], point[1] - previous[1]) < 0.5:
            return
        stroke.points.append(point)
        x, y = zip(*stroke.points)
        if stroke.artist is None:
            (stroke.artist,) = self.axis.plot(
                x,
                y,
                color="#7d898f" if stroke.erase else self.color,
                linewidth=max(2, stroke.sigma / 3),
                alpha=0.9,
                zorder=20,
            )
        else:
            stroke.artist.set_data(x, y)
        now = perf_counter()
        if now - self._last_preview_draw >= 1 / 30:
            self.figure.canvas.draw_idle()
            self._last_preview_draw = now

    def on_release(self, event) -> None:
        stroke = self.stroke
        if stroke is None:
            return
        release_button = getattr(event, "button", stroke.button)
        if release_button not in (None, stroke.button):
            return
        if (
            event.inaxes is self.axis
            and event.xdata is not None
            and event.ydata is not None
        ):
            point = (float(event.xdata), float(event.ydata))
            if point != stroke.points[-1]:
                stroke.points.append(point)
        if stroke.artist is not None:
            stroke.artist.remove()
        self.stroke = None

        current = self.field
        if stroke.erase:
            updated = erase_stroke(current, stroke.points, stroke.sigma)
        else:
            sign = -1 if stroke.button == 3 else 1
            updated = paint_scalar_stroke(
                current,
                stroke.points,
                stroke.sigma,
                sign * stroke.amplitude,
            )
        self._commit(self.active_key, updated)
        self.on_change("Field edited. Press Run.")

    def cancel(self) -> bool:
        if self.stroke is None:
            return False
        if self.stroke.artist is not None:
            with suppress(ValueError):
                self.stroke.artist.remove()
        self.stroke = None
        return True
