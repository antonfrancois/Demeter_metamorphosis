"""Interactive vector/scalar field editor for Metamorphosplines prototypes.

Version: July 16, 2026.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("QtAgg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from draft.playground.field_playground_core import (
    AnalysisResult,
    FORMAT_VERSION,
    LoadedField,
    SCALAR_KINDS,
    VECTOR_KINDS,
    add_vector_arrow,
    analyze_field,
    coerce_field,
    coerce_image,
    erase_stroke,
    load_field_file,
    mode_for_kind,
    paint_scalar_stroke,
    resize_field,
)

IMAGE_BANK = PROJECT_ROOT / "examples" / "im2Dbank"
DEFAULT_IMAGE = IMAGE_BANK / "BraTS2021_00090_80_.png"
VECTOR_DISPLAY_RELATIVE_THRESHOLD = 0.03
MAX_DISPLAY_ARROWS = 2500
UNDO_LIMIT = 12
PANEL_FONT_SIZE = 11
PRIMAL_COLOR = "#ffd166"
DUAL_COLOR = "#ef8354"
PRIMAL_KINDS = frozenset((VECTOR_KINDS[0], SCALAR_KINDS[0]))
SIGNED_FIELD_CMAP = LinearSegmentedColormap.from_list(
    "signed_field",
    (
        (0.0, (0.08, 0.36, 0.72, 1.0)),
        (0.5, (0.50, 0.50, 0.50, 0.0)),
        (1.0, (0.84, 0.18, 0.16, 1.0)),
    ),
    N=257,
)


def _variable_color(kind: str) -> str:
    return PRIMAL_COLOR if kind in PRIMAL_KINDS else DUAL_COLOR


def _counterpart_color(kind: str) -> str:
    return DUAL_COLOR if kind in PRIMAL_KINDS else PRIMAL_COLOR


def resolve_image_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    for path in (candidate, IMAGE_BANK / candidate, IMAGE_BANK / f"reg_test_{value}.png"):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"could not find image {value!r}; pass a path, im2Dbank filename, or shorthand such as '01'"
    )


def load_image(
    value: str | Path, size: tuple[int, int] | None = None
) -> tuple[torch.Tensor, Path]:
    path = resolve_image_path(value)
    array = mpimg.imread(path)
    if array.ndim == 3:
        array = np.dot(array[..., :3], [0.2989, 0.5870, 0.1140])
    image = coerce_image(np.asarray(array).copy())
    if size is not None and tuple(image.shape[-2:]) != tuple(size):
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return image, path


def _parameter_values(alpha: Any, beta: Any, gamma: Any, rho: Any) -> dict[str, float]:
    return dict(alpha=float(alpha), beta=float(beta), gamma=float(gamma), rho=float(rho))


@dataclass
class Drag:
    mode: str
    start: tuple[float, float]
    end: tuple[float, float]
    points: list[tuple[float, float]]
    button: int
    erase: bool
    sigma: float
    amplitude: float
    artist: Any = None
    background: Any = None

    @property
    def is_vector_arrow(self) -> bool:
        return self.mode == "vector" and not self.erase and self.button == 1


class FieldPlayground:
    """Single-window editor. Numerical and file operations live in the core module."""

    def __init__(
        self,
        image: torch.Tensor,
        image_path: str | Path | None = None,
        *,
        device: str = "cpu",
        alpha: float = 0.2,
        beta: float = 0.2,
        gamma: float = 0.001,
        rho: float = 0.5,
        output_path: str | Path | None = None,
    ):
        parameters = _parameter_values(alpha, beta, gamma, rho)
        self.image = coerce_image(image)
        self.image_path = str(image_path) if image_path is not None else None
        self.device = device
        self.output_path = Path(output_path).expanduser() if output_path else None
        height, width = self.image.shape[-2:]
        self.fields = {
            "vector": torch.zeros((1, 2, height, width)),
            "scalar": torch.zeros((1, 1, height, width)),
        }
        self.history = {"vector": deque(maxlen=UNDO_LIMIT), "scalar": deque(maxlen=UNDO_LIMIT)}
        self.kind = "vector_momentum"
        self.analysis: AnalysisResult | None = None
        self.error_colorbar = None
        self.drag: Drag | None = None
        self.scalar_detail = "error"
        self._syncing_widgets = False

        self._build_figure(parameters)
        self._connect_events()
        self._set_status("Amplitude scales edits; Spacing is display-only. Draw, then Run.")
        self._render()

    @property
    def mode(self) -> str:
        return mode_for_kind(self.kind)

    def _build_figure(self, parameters: dict[str, float]) -> None:
        for setting, reserved in {
            "keymap.save": {"s", "ctrl+s"},
            "keymap.home": {"r"},
            "keymap.back": {"c"},
        }.items():
            plt.rcParams[setting] = [key for key in plt.rcParams[setting] if key not in reserved]

        self.fig = plt.figure(figsize=(16, 8.5), facecolor="#edf1f3")
        grid = self.fig.add_gridspec(
            1, 3, left=0.035, right=0.76, bottom=0.15, top=0.81, wspace=0.16
        )
        self.axes = [self.fig.add_subplot(grid[0, index]) for index in range(3)]
        self.input_ax, self.output_ax, self.detail_ax = self.axes
        image = self.image[0, 0].numpy()
        self._base_images = {}
        for axis in self.axes:
            axis.set_facecolor("#10191e")
            axis.set_aspect("equal")
            axis.set_xlim(-0.5, image.shape[1] - 0.5)
            axis.set_ylim(-0.5, image.shape[0] - 0.5)
            axis.set_axis_off()
            self._base_images[axis] = axis.imshow(
                image, cmap="gray", origin="lower", vmin=0, vmax=1
            )
        self.fig.suptitle(
            "Metamorphosplines Field Playground",
            x=0.47,
            y=0.975,
            fontsize=17,
            fontweight="semibold",
            color="#24333b",
        )
        self.fig.text(
            0.47,
            0.91,
            r"$Lv=-\alpha\Delta v-\beta\nabla(\nabla\!\cdot v)+\gamma v,\qquad K=L^{-1}$",
            ha="center",
            fontsize=12,
            color="#26343c",
        )
        self.fig.text(
            0.47,
            0.865,
            r"$a=A_Iu=\rho\,K(u\nabla I)\!\cdot\!\nabla I+(1-\rho)u$",
            ha="center",
            fontsize=12,
            color="#26343c",
        )

        panel_color = "#f8fafb"
        self.mode_radio = RadioButtons(
            self.fig.add_axes([0.815, 0.82, 0.165, 0.105], facecolor=panel_color),
            ("Vector field", "Scalar field"),
            active=0,
            activecolor="#168a8a",
        )
        self.kind_radio = RadioButtons(
            self.fig.add_axes([0.815, 0.705, 0.165, 0.1], facecolor=panel_color),
            (r"Primal: velocity $v$", r"Dual: momentum $m$"),
            active=1,
            radio_props={"facecolor": (PRIMAL_COLOR, DUAL_COLOR)},
        )

        def slider(y, label, low, high, value, **kwargs):
            widget = Slider(
                self.fig.add_axes([0.875, y, 0.09, 0.024], facecolor=panel_color),
                label,
                low,
                high,
                valinit=value,
                **kwargs,
            )
            widget.label.set_x(-0.16)
            return widget

        self.brush_slider = slider(0.65, "Brush", 1, 40, 1)
        self.amplitude_slider = slider(0.595, "Amplitude", 0.05, 4, 1)
        self.spacing_slider = slider(0.54, "Spacing", 1, 24, 6, valstep=1)
        self.rho_slider = slider(0.47, r"$\rho$", 0, 0.95, parameters["rho"])
        self.alpha_slider = slider(
            0.415, r"$\alpha$", 0, max(2, 1.5 * parameters["alpha"]), parameters["alpha"]
        )
        self.beta_slider = slider(
            0.36, r"$\beta$", 0, max(2, 1.5 * parameters["beta"]), parameters["beta"]
        )
        log_gamma = float(np.log10(parameters["gamma"]))
        self.gamma_slider = slider(
            0.305,
            r"$\gamma$",
            min(-4, np.floor(log_gamma) - 1),
            max(0, np.ceil(log_gamma) + 1),
            log_gamma,
        )
        self.gamma_slider.valtext.set_text(f"{parameters['gamma']:.3g}")

        self.run_button = Button(
            self.fig.add_axes([0.815, 0.235, 0.165, 0.048]),
            "Run",
            color="#168a8a",
            hovercolor="#20a3a3",
        )
        self.run_button.label.set_color("white")
        self.load_button = Button(self.fig.add_axes([0.815, 0.175, 0.078, 0.043]), "Load")
        self.save_button = Button(self.fig.add_axes([0.902, 0.175, 0.078, 0.043]), "Save")
        self.undo_button = Button(self.fig.add_axes([0.815, 0.12, 0.078, 0.043]), "Undo")
        self.clear_button = Button(self.fig.add_axes([0.902, 0.12, 0.078, 0.043]), "Clear")
        detail_position = self.detail_ax.get_position()
        detail_center = (detail_position.x0 + detail_position.x1) / 2
        button_gap = 0.008
        error_width, deformation_width = 0.075, 0.125
        button_left = detail_center - (
            error_width + button_gap + deformation_width
        ) / 2
        button_y = detail_position.y1 + 0.07
        self.scalar_error_button = Button(
            self.fig.add_axes([button_left, button_y, error_width, 0.036]),
            "Error",
        )
        self.scalar_deformation_button = Button(
            self.fig.add_axes(
                [
                    button_left + error_width + button_gap,
                    button_y,
                    deformation_width,
                    0.036,
                ]
            ),
            "Deformation velocity",
        )
        self.scalar_detail_buttons = {
            "error": self.scalar_error_button,
            "deformation": self.scalar_deformation_button,
        }
        for button in self.scalar_detail_buttons.values():
            button.ax.set_visible(False)
            button.label.set_fontsize(PANEL_FONT_SIZE)
        self.status_text = self.fig.text(0.815, 0.055, "", fontsize=8.5, color="#26343c", wrap=True)
        self.norm_text = None
        self.fig.text(
            0.02,
            0.985,
            "Left-drag    add / paint\n"
            "Right-drag   erase vector / negative scalar\n"
            "Shift-drag   erase\n"
            "R            run\n"
            "Ctrl+Z       undo\n"
            "Ctrl+O       load\n"
            "Ctrl+S       save",
            fontsize=8.2,
            color="#53656f",
            va="top",
        )

        self.mode_radio.on_clicked(self._on_mode)
        self.kind_radio.on_clicked(self._on_kind)
        self.run_button.on_clicked(lambda _event: self.run())
        self.load_button.on_clicked(lambda _event: self.load_dialog())
        self.save_button.on_clicked(lambda _event: self.save_dialog())
        self.undo_button.on_clicked(lambda _event: self.undo())
        self.clear_button.on_clicked(lambda _event: self.clear())
        self.scalar_error_button.on_clicked(
            lambda _event: self._set_scalar_detail("error")
        )
        self.scalar_deformation_button.on_clicked(
            lambda _event: self._set_scalar_detail("deformation")
        )
        self.spacing_slider.on_changed(self._on_spacing_change)
        for widget in (self.rho_slider, self.alpha_slider, self.beta_slider):
            widget.on_changed(self._on_parameter_change)
        self.gamma_slider.on_changed(self._on_gamma_change)
        for widget in (
            self.mode_radio,
            self.kind_radio,
            self.spacing_slider,
            self.rho_slider,
            self.alpha_slider,
            self.beta_slider,
            self.gamma_slider,
        ):
            widget.drawon = False

    def _connect_events(self) -> None:
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("resize_event", lambda _event: self._cancel_drag())

    def _parameters(self) -> dict[str, float]:
        return {
            "alpha": float(self.alpha_slider.val),
            "beta": float(self.beta_slider.val),
            "gamma": 10 ** float(self.gamma_slider.val),
            "rho": float(self.rho_slider.val),
        }

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)

    def _cancel_drag(self) -> bool:
        drag = self.drag
        if drag is None:
            return False
        if drag.background is not None and self.fig.canvas.supports_blit:
            self.fig.canvas.restore_region(drag.background)
            self.fig.canvas.blit(self.input_ax.bbox)
        if drag.artist is not None:
            drag.artist.set_animated(False)
            try:
                drag.artist.remove()
            except ValueError:
                pass
        self.drag = None
        return True

    def _invalidate(self, message: str, *, immediate: bool = False) -> None:
        self.analysis = None
        self._set_status(message)
        self._render(immediate=immediate)

    def run(self) -> None:
        self._cancel_drag()
        self._set_status("Computing...")
        self.fig.canvas.draw_idle()
        try:
            self.analysis = analyze_field(
                self.image,
                self.fields[self.mode],
                self.kind,
                device=self.device,
                **self._parameters(),
            )
            self._set_status("Computation complete.")
        except Exception as error:
            self.analysis = None
            self._set_status(f"ERROR: {type(error).__name__}: {error}")
        self._render()

    def refresh(self) -> None:
        self.run()

    def _on_parameter_change(self, _value: float) -> None:
        if self._syncing_widgets:
            return
        if self.analysis is None:
            self._set_status("Parameters changed. Press Run.")
            self.fig.canvas.draw_idle()
        else:
            self._invalidate("Parameters changed. Press Run.")

    def _on_gamma_change(self, value: float) -> None:
        self.gamma_slider.valtext.set_text(f"{10 ** float(value):.3g}")
        self._on_parameter_change(value)

    def _on_spacing_change(self, _value: float) -> None:
        self._cancel_drag()
        if self.mode == "vector":
            self._clear_axis_dynamic(self.input_ax)
            self._create_energy_text()
            current = self.fields["vector"]
            self._plot_vector(
                self.input_ax,
                current,
                f"Input: {self._kind_title(self.kind)}",
                _variable_color(self.kind),
            )
            if self.analysis is not None:
                self._clear_axis_dynamic(self.output_ax)
                self._plot_vector_output()
                self._set_energy(
                    r"\Vert v\Vert_V^2 = \Vert m\Vert_{V^*}^2",
                    self.analysis.squared_norm,
                )
        elif self.analysis is not None and self.scalar_detail == "deformation":
            self._clear_axis_dynamic(self.detail_ax)
            self._plot_scalar_deformation_velocity()
        self.fig.canvas.draw_idle()

    def _set_scalar_detail(self, detail: str) -> None:
        if detail == self.scalar_detail:
            return
        self.scalar_detail = detail
        if self.mode == "scalar":
            self._render()

    def _sync_scalar_detail_buttons(self) -> None:
        visible = self.mode == "scalar"
        for detail, button in self.scalar_detail_buttons.items():
            active = detail == self.scalar_detail
            color = "#168a8a" if active else "#f8fafb"
            button.color = color
            button.hovercolor = "#20a3a3" if active else "#e4ecef"
            button.ax.set_facecolor(color)
            button.label.set_color("white" if active else "#26343c")
            button.ax.set_visible(visible)

    def _sync_radios(self) -> None:
        self._syncing_widgets = True
        try:
            mode = self.mode
            kinds = VECTOR_KINDS if mode == "vector" else SCALAR_KINDS
            labels = (
                (r"Primal: velocity $v$", r"Dual: momentum $m$")
                if mode == "vector"
                else (r"Primal: acceleration $a$", r"Dual: covector $u$")
            )
            mode_index = 0 if mode == "vector" else 1
            if self.mode_radio.index_selected != mode_index:
                self.mode_radio.set_active(mode_index)
            for text, label in zip(self.kind_radio.labels, labels):
                text.set_text(label)
            kind_index = kinds.index(self.kind)
            if self.kind_radio.index_selected != kind_index:
                self.kind_radio.set_active(kind_index)
        finally:
            self._syncing_widgets = False

    def _on_mode(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        index = (VECTOR_KINDS if self.mode == "vector" else SCALAR_KINDS).index(self.kind)
        kinds = VECTOR_KINDS if self.mode_radio.index_selected == 0 else SCALAR_KINDS
        new_kind = kinds[index]
        if new_kind == self.kind:
            return
        self.kind = new_kind
        if self.mode == "scalar":
            self.scalar_detail = "error"
        self._sync_radios()
        self._invalidate("Mode changed. Press Run.")

    def _on_kind(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        kinds = VECTOR_KINDS if self.mode == "vector" else SCALAR_KINDS
        new_kind = kinds[self.kind_radio.index_selected]
        if new_kind == self.kind:
            return
        self.kind = new_kind
        self._invalidate("Field interpretation changed. Press Run.")

    def _toolbar_is_active(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(toolbar is not None and getattr(toolbar, "mode", ""))

    def _on_press(self, event) -> None:
        if (
            event.inaxes is not self.input_ax
            or event.xdata is None
            or event.ydata is None
            or event.button not in (1, 3)
            or self._toolbar_is_active()
        ):
            return
        self._cancel_drag()
        point = (float(event.xdata), float(event.ydata))
        erase = event.button == 3 and self.mode == "vector"
        erase |= "shift" in (event.key or "").lower()
        self.drag = Drag(
            mode=self.mode,
            start=point,
            end=point,
            points=[point],
            button=int(event.button),
            erase=erase,
            sigma=float(self.brush_slider.val),
            amplitude=float(self.amplitude_slider.val),
        )
        if self.fig.canvas.supports_blit:
            self.drag.background = self.fig.canvas.copy_from_bbox(self.input_ax.bbox)

    def _on_motion(self, event) -> None:
        drag = self.drag
        if drag is None or event.inaxes is not self.input_ax or event.xdata is None or event.ydata is None:
            return
        point = (float(event.xdata), float(event.ydata))
        if np.hypot(point[0] - drag.end[0], point[1] - drag.end[1]) < 0.5:
            return
        drag.end = point
        if drag.is_vector_arrow:
            if drag.artist is None:
                drag.artist = FancyArrowPatch(
                    drag.start,
                    point,
                    arrowstyle="->",
                    color=_variable_color(self.kind),
                    linewidth=2.2,
                    mutation_scale=12,
                    zorder=10,
                    animated=self.fig.canvas.supports_blit,
                )
                self.input_ax.add_patch(drag.artist)
            else:
                drag.artist.set_positions(drag.start, point)
        else:
            drag.points.append(point)
            x, y = zip(*drag.points)
            if drag.artist is None:
                (drag.artist,) = self.input_ax.plot(
                    x,
                    y,
                    color="#87949b" if drag.erase else _variable_color(self.kind),
                    linewidth=max(2, drag.sigma / 3),
                    alpha=0.8,
                    zorder=10,
                    animated=self.fig.canvas.supports_blit,
                )
            else:
                drag.artist.set_data(x, y)
        canvas = self.fig.canvas
        if not canvas.supports_blit:
            canvas.draw_idle()
            return
        if drag.background is None:
            canvas.draw()
            drag.background = canvas.copy_from_bbox(self.input_ax.bbox)
        else:
            canvas.restore_region(drag.background)
        self.input_ax.draw_artist(drag.artist)
        canvas.blit(self.input_ax.bbox)

    def _on_release(self, event) -> None:
        drag = self.drag
        if drag is None:
            return
        release_button = getattr(event, "button", drag.button)
        if release_button not in (None, drag.button):
            return
        if event.inaxes is self.input_ax and event.xdata is not None and event.ydata is not None:
            drag.end = (float(event.xdata), float(event.ydata))
        if drag.is_vector_arrow and np.hypot(
            drag.end[0] - drag.start[0], drag.end[1] - drag.start[1]
        ) < 0.5:
            self._cancel_drag()
            return
        if not drag.is_vector_arrow and drag.points[-1] != drag.end:
            drag.points.append(drag.end)
        if drag.artist is not None:
            drag.artist.set_animated(False)
            drag.artist.remove()
        self.drag = None

        field = self.fields[drag.mode]
        if drag.is_vector_arrow:
            new_field = add_vector_arrow(
                field, drag.start, drag.end, drag.sigma, drag.amplitude
            )
        elif drag.erase:
            new_field = erase_stroke(field, drag.points, drag.sigma)
        else:
            sign = -1 if drag.button == 3 else 1
            new_field = paint_scalar_stroke(
                field, drag.points, drag.sigma, sign * drag.amplitude
            )
        self.history[drag.mode].append(field)
        self.fields[drag.mode] = new_field
        self.analysis = None
        self._set_status("Field edited. Press Run.")
        self._render(immediate=True)

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key == "ctrl+z":
            self.undo()
        elif key == "ctrl+s":
            self.save_dialog(quick=True)
        elif key == "ctrl+o":
            self.load_dialog()
        elif key == "r":
            self.run()
        elif key == "c":
            self.clear()
        elif key == "v":
            self.mode_radio.set_active(0)
        elif key == "s":
            self.mode_radio.set_active(1)
        elif key == "escape":
            self._cancel_drag()

    def undo(self) -> None:
        self._cancel_drag()
        if not self.history[self.mode]:
            self._set_status(f"Nothing to undo in {self.mode} mode.")
            self.fig.canvas.draw_idle()
            return
        self.fields[self.mode] = self.history[self.mode].pop()
        self._invalidate("Undo applied. Press Run.", immediate=True)

    def clear(self) -> None:
        cancelled = self._cancel_drag()
        field = self.fields[self.mode]
        if torch.count_nonzero(field) == 0:
            if cancelled:
                self._render(immediate=True)
            return
        self.history[self.mode].append(field)
        self.fields[self.mode] = torch.zeros_like(field)
        self._invalidate("Field cleared. Press Run.", immediate=True)

    def set_template(
        self,
        loaded: LoadedField,
        source: str | Path | None = None,
        *,
        restore_parameters: bool = True,
    ) -> None:
        self._cancel_drag()
        mode = mode_for_kind(loaded.kind)
        original_size = tuple(loaded.field.shape[-2:])
        target_size = tuple(self.image.shape[-2:])
        self.fields[mode] = resize_field(
            loaded.field,
            target_size,
            scale_vector_displacement=loaded.kind == "velocity",
        )
        self.history[mode].clear()
        self.kind = loaded.kind
        self._sync_radios()
        restored = restore_parameters and self._restore_parameters(
            loaded.metadata.get("parameters")
        )
        resized = "" if original_size == target_size else f"; resized {original_size} to {target_size}"
        parameter_note = "; parameters restored" if restored else ""
        self._invalidate(
            f"Loaded {loaded.kind} from {source or 'memory'}{resized}{parameter_note}. Press Run.",
            immediate=True,
        )

    def _restore_parameters(self, parameters: Any) -> bool:
        if not isinstance(parameters, dict):
            return False
        values = _parameter_values(
            parameters["alpha"], parameters["beta"], parameters["gamma"], parameters["rho"]
        )
        log_gamma = float(np.log10(values["gamma"]))
        widget_values = {
            self.alpha_slider: values["alpha"],
            self.beta_slider: values["beta"],
            self.gamma_slider: log_gamma,
            self.rho_slider: values["rho"],
        }
        self._syncing_widgets = True
        try:
            for widget, value in widget_values.items():
                if widget is not self.rho_slider:
                    padding = 1 if widget is self.gamma_slider else max(1, value * 0.5)
                    widget.valmin = min(widget.valmin, value - (1 if widget is self.gamma_slider else 0))
                    widget.valmax = max(widget.valmax, value + padding)
                    widget.ax.set_xlim(widget.valmin, widget.valmax)
                widget.set_val(value)
        finally:
            self._syncing_widgets = False
        self.gamma_slider.valtext.set_text(f"{values['gamma']:.3g}")
        return True

    def _choose_file(self, *, save: bool) -> Path | None:
        try:
            from matplotlib.backends.qt_compat import QtWidgets
        except ImportError:
            self._set_status("Qt file dialog unavailable; use --field or --output.")
            self.fig.canvas.draw_idle()
            return None
        file_dialog = QtWidgets.QFileDialog
        if save:
            filename, _ = file_dialog.getSaveFileName(
                None, "Save field", str(PROJECT_ROOT / "draft"), "PyTorch field (*.pt)"
            )
        else:
            filename, _ = file_dialog.getOpenFileName(
                None,
                "Load field",
                str(PROJECT_ROOT / "draft"),
                "Field files (*.pt *.pth *.npy *.npz);;All files (*)",
            )
        return Path(filename) if filename else None

    def load_dialog(self) -> None:
        self._cancel_drag()
        path = self._choose_file(save=False)
        if path is None:
            return
        try:
            self.set_template(load_field_file(path), path)
        except Exception as error:
            self._set_status(f"LOAD ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def make_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "field": self.fields[self.mode].detach().cpu(),
            "field_kind": self.kind,
            "image": self.image.detach().cpu(),
            "image_path": self.image_path or "",
            "dx_convention": "pixel",
            "parameters": self._parameters(),
        }
        if self.analysis is not None:
            payload["diagnostics"] = {
                "relative_roundtrip": self.analysis.relative_roundtrip,
                "squared_norm": self.analysis.squared_norm,
            }
            if self.analysis.solver_iterations is not None:
                payload["diagnostics"].update(
                    solver_residual=self.analysis.solver_residual,
                    solver_iterations=self.analysis.solver_iterations,
                    solver_time=self.analysis.solver_time,
                )
            if self.analysis.operator_time is not None:
                payload["diagnostics"]["operator_time"] = self.analysis.operator_time
            if self.analysis.deformation_energy_contribution is not None:
                payload["diagnostics"]["deformation_energy_contribution"] = (
                    self.analysis.deformation_energy_contribution
                )
            payload["counterpart"] = self.analysis.counterpart
            payload["roundtrip"] = self.analysis.roundtrip
            if self.analysis.deformation_velocity is not None:
                payload["deformation_velocity"] = self.analysis.deformation_velocity
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        if not path.suffix:
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.make_payload(), path)
        self.output_path = path
        self._set_status(f"Saved {self.kind} field to {path}.")
        self.fig.canvas.draw_idle()
        return path

    def save_dialog(self, quick: bool = False) -> None:
        self._cancel_drag()
        path = self.output_path if quick and self.output_path is not None else self._choose_file(save=True)
        if path is None:
            return
        try:
            self.save(path)
        except Exception as error:
            self._set_status(f"SAVE ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def _clear_axis_dynamic(self, axis) -> None:
        if (
            self.error_colorbar is not None
            and self.error_colorbar.mappable.axes is axis
        ):
            self.error_colorbar.remove()
            self.error_colorbar = None
        base_image = self._base_images[axis]
        for image in list(axis.images):
            if image is not base_image:
                image.remove()
        for artists in (axis.collections, axis.lines, axis.patches, axis.texts):
            for artist in list(artists):
                artist.remove()
        base_image.set_visible(True)
        axis.set_title("")

    def _clear_dynamic_artists(self) -> None:
        for axis in self.axes:
            self._clear_axis_dynamic(axis)

    def _create_energy_text(self) -> None:
        self.norm_text = self.input_ax.text(
            0.5,
            -0.075,
            "",
            transform=self.input_ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_FONT_SIZE,
            color="#26343c",
            clip_on=False,
        )

    def _render(self, *, immediate: bool = False) -> None:
        self._cancel_drag()
        self._sync_scalar_detail_buttons()
        self._clear_dynamic_artists()
        self._create_energy_text()

        current = self.fields[self.mode]
        if self.mode == "vector":
            self._render_vector_mode(current)
        else:
            self._render_scalar_mode(current)
        self.fig.canvas.draw() if immediate else self.fig.canvas.draw_idle()

    def _render_vector_mode(self, current: torch.Tensor) -> None:
        self._plot_vector(
            self.input_ax,
            current,
            f"Input: {self._kind_title(self.kind)}",
            _variable_color(self.kind),
        )
        if self.analysis is None:
            self._plot_message(self.output_ax, "Press Run")
            self._plot_message(self.detail_ax, "No derived field yet")
            return

        self._plot_vector_output()
        error = (self.analysis.roundtrip - current).square().sum(1, keepdim=True).sqrt()
        reference = current.square().sum(1, keepdim=True).sqrt()
        relative_error = self._relative_l2_error(error, reference)
        self._plot_scalar(
            self.detail_ax,
            relative_error,
            self._roundtrip_error_title(self.kind),
            signed=False,
            log_scale=True,
            log_floor=torch.finfo(current.dtype).eps,
            show_base=False,
        )
        self._add_error_metrics(self.detail_ax, error, reference)
        self._set_energy(
            r"\Vert v\Vert_V^2 = \Vert m\Vert_{V^*}^2",
            self.analysis.squared_norm,
        )

    def _plot_vector_output(self) -> None:
        assert self.analysis is not None
        velocity_input = self.kind == "velocity"
        operator_name = "L" if velocity_input else "K"
        output_title = (
            r"Output: momentum $m=Lv$"
            if velocity_input
            else r"Output: velocity $v=Km$"
        )
        self._plot_vector(
            self.output_ax,
            self.analysis.counterpart,
            output_title,
            _counterpart_color(self.kind),
            footer=rf"${operator_name}$ avg time = {self._format_time(self.analysis.operator_time)}",
        )

    def _render_scalar_mode(self, current: torch.Tensor) -> None:
        self._plot_scalar(
            self.input_ax, current, f"Input: {self._kind_title(self.kind)}"
        )
        if self.analysis is None:
            self._plot_message(self.output_ax, "Press Run")
            self._plot_message(self.detail_ax, "No derived field yet")
            return

        acceleration_input = self.kind == "a"
        output_title = (
            r"Output: covector $u=A_I^{-1}a$"
            if acceleration_input
            else r"Output: acceleration $a=A_Iu$"
        )
        self._plot_scalar(self.output_ax, self.analysis.counterpart, output_title)
        error = (self.analysis.roundtrip - current).abs()
        reference = current.abs()
        if self.scalar_detail == "error":
            relative_error = self._relative_l2_error(error, reference)
            self._plot_scalar(
                self.detail_ax,
                relative_error,
                self._roundtrip_error_title(self.kind),
                signed=False,
                log_scale=True,
                log_floor=torch.finfo(current.dtype).eps,
                show_base=False,
            )
            self._add_error_metrics(self.detail_ax, error, reference)
        else:
            self._plot_scalar_deformation_velocity()
        if acceleration_input:
            self._add_solver_metrics(self.output_ax)
        else:
            self._add_cometric_metrics(self.output_ax)
        self._set_energy(
            r"\Vert a\Vert_{A_I^{-1}}^2 = \Vert u\Vert_{A_I}^2",
            self.analysis.squared_norm,
        )

    def _plot_scalar_deformation_velocity(self) -> None:
        assert self.analysis is not None
        self._plot_vector(
            self.detail_ax,
            self.analysis.deformation_velocity,
            r"Deformation velocity $-\sqrt{\rho}\,K(u\nabla I)$",
            PRIMAL_COLOR,
            footer=(
                r"$\rho\,\Vert K(u\nabla I)\Vert_V^2 = "
                rf"{self._latex_number(self.analysis.deformation_energy_contribution)}$"
            ),
        )

    @staticmethod
    def _kind_title(kind: str) -> str:
        return {
            "velocity": r"velocity $v$",
            "vector_momentum": r"momentum $m$",
            "u": r"scalar covector $u$",
            "a": r"acceleration $a$",
        }[kind]

    @staticmethod
    def _roundtrip_error_title(kind: str) -> str:
        numerator, reference = {
            "velocity": (r"\Vert K(Lv)-v\Vert_2", r"\Vert v\Vert_2"),
            "vector_momentum": (r"\Vert L(Km)-m\Vert_2", r"\Vert m\Vert_2"),
            "u": (r"\left|A_I^{-1}(A_Iu)-u\right|", r"\left|u\right|"),
            "a": (r"\left|A_I(A_I^{-1}a)-a\right|", r"\left|a\right|"),
        }[kind]
        return rf"${numerator}/\mathrm{{RMS}}({reference})$ (log scale)"

    @staticmethod
    def _latex_number(value: float) -> str:
        if np.isinf(value):
            return r"\infty"
        if np.isnan(value):
            return r"\mathrm{nan}"
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        return rf"{value / 10**exponent:.3g}\times 10^{{{exponent}}}"

    @staticmethod
    def _format_time(elapsed: float) -> str:
        return f"{elapsed * 1e3:.3g} ms" if elapsed < 1 else f"{elapsed:.3g} s"

    @staticmethod
    def _relative_l2_error(
        error: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        reference_rms = reference.square().mean().sqrt()
        if reference_rms == 0:
            if error.max() == 0:
                return torch.zeros_like(error)
            return torch.full_like(error, float("inf"))
        return error / reference_rms

    @classmethod
    def _relative_l2_metrics(
        cls, error: torch.Tensor, reference: torch.Tensor
    ) -> tuple[float, float]:
        relative_error = cls._relative_l2_error(error, reference)
        return (
            float(relative_error.square().mean().sqrt()),
            float(relative_error.max()),
        )

    def _error_lines(
        self, error: torch.Tensor, reference: torch.Tensor
    ) -> tuple[str, str]:
        mean, maximum = self._relative_l2_metrics(error, reference)
        return (
            rf"$\mathrm{{mean}} = {self._latex_number(mean)}$",
            rf"$\mathrm{{max}} = {self._latex_number(maximum)}$",
        )

    @staticmethod
    def _add_metrics(axis, lines: tuple[str, ...]) -> None:
        axis.text(
            0.5,
            -0.075,
            "\n".join(lines),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_FONT_SIZE,
            color="#26343c",
            clip_on=False,
        )

    def _add_cometric_metrics(self, axis) -> None:
        self._add_metrics(
            axis,
            (rf"$A_I$ avg time = {self._format_time(self.analysis.operator_time)}",),
        )

    def _add_error_metrics(
        self, axis, error: torch.Tensor, reference: torch.Tensor
    ) -> None:
        self._add_metrics(axis, self._error_lines(error, reference))

    def _add_solver_metrics(self, axis) -> None:
        self._add_metrics(
            axis,
            (
                rf"$\mathrm{{residual}} = {self._latex_number(self.analysis.solver_residual)}$",
                rf"$\mathrm{{iterations}} = {self.analysis.solver_iterations}$",
                rf"$A_I^{{-1}}$ avg time = {self._format_time(self.analysis.solver_time)}",
            ),
        )

    def _set_energy(self, norm: str, value: float) -> None:
        self.norm_text.set_text(rf"${norm} = {self._latex_number(value)}$")

    def _plot_base_image(self, axis) -> None:
        self._base_images[axis].set_visible(True)

    def _plot_vector(
        self,
        axis,
        field: torch.Tensor,
        title: str,
        color: str,
        *,
        footer: str | None = None,
    ) -> None:
        self._plot_base_image(axis)
        display_field = field.detach().cpu()
        magnitude = display_field.square().sum(dim=1).sqrt()[0]
        maximum = float(magnitude.max())
        visible = magnitude >= max(
            1e-8, VECTOR_DISPLAY_RELATIVE_THRESHOLD * maximum
        )
        visible_count = int(visible.sum())
        factor = 1.0
        if visible_count:
            q95 = float(torch.quantile(magnitude[visible], 0.95))
            target = float(np.clip(0.06 * min(magnitude.shape), 12, 48))
            factor = target / q95
            pooled = F.max_pool2d(
                magnitude[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            spacing = int(self.spacing_slider.val)
            spacing = max(
                spacing,
                int(np.ceil(np.sqrt(visible_count / MAX_DISPLAY_ARROWS))),
            )
            mask = torch.zeros_like(visible)
            mask[::spacing, ::spacing] = True
            mask |= visible & (magnitude == pooled)
            y, x = torch.where(mask & visible)
            if y.numel() > MAX_DISPLAY_ARROWS:
                keep = torch.linspace(
                    0, y.numel() - 1, MAX_DISPLAY_ARROWS
                ).round().long()
                y, x = y[keep], x[keep]
            axis.quiver(
                x.numpy(),
                y.numpy(),
                (display_field[0, 0, y, x] * factor).numpy(),
                (display_field[0, 1, y, x] * factor).numpy(),
                color=color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
            )
        if footer is not None:
            axis.text(
                0.5,
                -0.075,
                footer,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=PANEL_FONT_SIZE,
                color="#26343c",
                clip_on=False,
            )
        suffix = "" if abs(factor - 1) < 0.02 else f"  [x{factor:.2g}]"
        axis.set_title(
            title + suffix, fontsize=PANEL_FONT_SIZE, color="#24333b", pad=8
        )

    def _plot_scalar(
        self,
        axis,
        field: torch.Tensor,
        title: str,
        *,
        signed: bool = True,
        log_scale: bool = False,
        log_floor: float = 0,
        show_base: bool = True,
    ) -> None:
        self._base_images[axis].set_visible(show_base)
        values = field[0, 0].detach().cpu()
        if torch.count_nonzero(values):
            if log_scale:
                visible = values > log_floor
                if torch.any(visible):
                    visible_values = values[visible]
                    maximum = float(visible_values.max())
                    minimum = log_floor or float(visible_values.min())
                    if minimum == maximum:
                        minimum /= 10
                    display = values.clamp_min(minimum).numpy()
                    heatmap = axis.imshow(
                        display,
                        cmap="magma",
                        origin="lower",
                        norm=LogNorm(vmin=minimum, vmax=maximum),
                    )
                    colorbar_axis = axis.inset_axes([1.02, 0, 0.035, 1])
                    self.error_colorbar = self.fig.colorbar(heatmap, cax=colorbar_axis)
            else:
                absolute = values.abs()
                limit = max(float(torch.quantile(absolute.flatten(), 0.99)), 1e-8)
                display = np.ma.masked_where(
                    absolute.numpy() < 0.001 * limit, values.numpy()
                )
                axis.imshow(
                    display,
                    cmap=SIGNED_FIELD_CMAP if signed else "magma",
                    origin="lower",
                    vmin=-limit if signed else 0,
                    vmax=limit,
                    alpha=0.68 if signed else 0.78,
                )
        axis.set_title(title, fontsize=PANEL_FONT_SIZE, color="#24333b", pad=8)

    def _plot_message(self, axis, message: str) -> None:
        self._plot_base_image(axis)
        axis.text(
            0.5,
            0.5,
            message,
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=PANEL_FONT_SIZE,
            bbox={"facecolor": "#24333b", "alpha": 0.88, "edgecolor": "none", "pad": 8},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Image path, im2Dbank filename, or shorthand")
    parser.add_argument("--field", help="Existing .pt/.pth/.npy/.npz field template")
    parser.add_argument("--size", nargs=2, type=int, metavar=("H", "W"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--output", help="Path used by Ctrl+S")
    parser.add_argument("--screenshot", help="Save a rendered screenshot")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> FieldPlayground:
    args = build_parser().parse_args(argv)
    loaded = load_field_file(args.field) if args.field else None
    saved = loaded.metadata.get("parameters", {}) if loaded is not None else {}
    parameters = _parameter_values(
        args.alpha if args.alpha is not None else saved.get("alpha", 0.2),
        args.beta if args.beta is not None else saved.get("beta", 0.2),
        args.gamma if args.gamma is not None else saved.get("gamma", 0.001),
        args.rho if args.rho is not None else saved.get("rho", 0.5),
    )

    size = tuple(args.size) if args.size else None
    if args.image is not None:
        image, image_path = load_image(args.image, size)
    elif loaded is not None and loaded.image is not None:
        image = loaded.image
        if size is not None and tuple(image.shape[-2:]) != size:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        image_path = loaded.metadata.get("image_path") or None
    else:
        image, image_path = load_image(DEFAULT_IMAGE, size)

    app = FieldPlayground(
        image,
        image_path,
        device=args.device,
        output_path=args.output,
        **parameters,
    )
    if loaded is not None:
        app.set_template(loaded, args.field, restore_parameters=False)
    if args.screenshot:
        app.fig.savefig(args.screenshot, dpi=140, facecolor=app.fig.get_facecolor())
    if not args.no_show:
        plt.show()
    return app


if __name__ == "__main__":
    main()
