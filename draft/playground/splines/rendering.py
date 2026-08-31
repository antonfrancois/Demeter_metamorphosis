"""Panel rendering and field-overlay primitives for the spline playground."""

from contextlib import suppress
from itertools import compress
from typing import Any

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from draft.playground.field_playground_core import (
    DEFAULT_VECTOR_DISPLAY_SPACING,
    prepare_scalar_display,
    prepare_vector_display,
    scaled_field_title,
)

from .core import (
    SplineParameters,
    SplineTrajectory,
    cometric_squared_norm,
    metric_squared_norm,
)
from .styles import (
    DUAL_COLOR,
    FIELD_CLASS,
    FIELD_SYMBOL,
    INK_COLOR,
    PANEL_COLOR,
    PRIMAL_COLOR,
)


def _signed_cmap(name: str) -> LinearSegmentedColormap:
    base = colormaps["cool"]
    negative = (*base(0.25)[:3], 0.92)
    positive = (*base(0.75)[:3], 0.92)
    transparent_negative = (*negative[:3], 0.0)
    transparent_positive = (*positive[:3], 0.0)
    return LinearSegmentedColormap.from_list(
        name,
        (
            (0.0, negative),
            (0.49, transparent_negative),
            (0.51, transparent_positive),
            (1.0, positive),
        ),
        N=257,
    )


DUAL_CMAP = _signed_cmap("spline_dual")
PRIMAL_CMAP = _signed_cmap("spline_primal")


def field_color(field_class: str) -> str:
    return PRIMAL_COLOR if field_class == "primal" else DUAL_COLOR


def _current_title(
    cache: SplineTrajectory | None,
    image_mode: str,
    current_field: str | None,
    index: int,
    show_image: bool,
) -> str:
    if cache is None:
        return "Current image (run required)" if show_image else "Current field (run required)"
    if not show_image:
        return "No field overlay" if current_field is None else rf"${FIELD_SYMBOL[current_field]}$"
    title = {
        "full": rf"Current image $I_{{{index}}}$",
        "deformation": rf"Deformation only $I_{{D,{index}}}$",
        "photometric": rf"Photometric only $I_{{\mathrm{{phot}},{index}}}$",
    }[image_mode]
    if current_field is not None:
        title += rf" + ${FIELD_SYMBOL[current_field]}$"
    return title


def _set_current_image(
    artist,
    current: torch.Tensor,
    *,
    show_image: bool,
    photometric: bool,
) -> None:
    artist.set_data(current[0] if show_image else torch.zeros_like(current[0]))
    artist.set_cmap("gray")
    if show_image and photometric:
        lower = min(0.0, float(torch.quantile(current.flatten(), 0.01)))
        upper = max(1.0, float(torch.quantile(current.flatten(), 0.99)))
        artist.set_clim(lower, upper)
    else:
        artist.set_clim(0, 1)


def latex_number(value: float) -> str:
    if np.isinf(value):
        return r"\infty"
    if np.isnan(value):
        return r"\mathrm{nan}"
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    return rf"{value / 10**exponent:.3g}\times 10^{{{exponent}}}"


class SplineRenderer:
    """Render source, current, and target panels from explicit application state."""

    def __init__(
        self,
        axes: tuple[Any, Any, Any],
        images: tuple[Any, Any, Any],
        colorbar_axes: tuple[Any, Any, Any],
        footers: tuple[Any, Any, Any],
        dynamic_artists: dict[Any, list[Any]],
    ) -> None:
        self.source_ax, self.current_ax, self.target_ax = axes
        self.source_image, self.current_image, self.target_image = images
        self.colorbar_axes = dict(zip(axes, colorbar_axes, strict=True))
        self.colorbars: dict[Any, Any] = {}
        self.colorbar_visibility = dict.fromkeys(axes, False)
        self.source_footer, self.current_footer, self.target_footer = footers
        self.dynamic_artists = dynamic_artists
        self.vector_spacing = DEFAULT_VECTOR_DISPLAY_SPACING

    def clear_dynamic(self, axis: Any) -> None:
        self.hide_colorbar(axis)
        for artist in self.dynamic_artists[axis]:
            with suppress(ValueError):
                artist.remove()
        self.dynamic_artists[axis].clear()

    def hide_colorbar(self, axis: Any) -> None:
        self.colorbar_visibility[axis] = False
        self.colorbar_axes[axis].set_visible(False)

    def show_colorbar(self, axis: Any, mappable: Any) -> None:
        colorbar = self.colorbars.get(axis)
        if colorbar is None:
            colorbar = axis.figure.colorbar(
                mappable,
                cax=self.colorbar_axes[axis],
                orientation="horizontal",
            )
            self.colorbars[axis] = colorbar
        else:
            colorbar.update_normal(mappable)
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
        colorbar.ax.tick_params(
            axis="x",
            colors=INK_COLOR,
            labelsize=8,
            length=2,
            pad=2,
        )
        colorbar.outline.set_edgecolor(INK_COLOR)
        self.colorbar_visibility[axis] = True
        self.colorbar_axes[axis].set_visible(True)

    def set_colorbars_visible(self, visible: bool) -> None:
        for axis, colorbar_axis in self.colorbar_axes.items():
            colorbar_axis.set_visible(
                visible and self.colorbar_visibility[axis]
            )

    @staticmethod
    def configure_image_axis(axis: Any, image: torch.Tensor) -> None:
        height, width = image.shape[-2:]
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        axis.set_box_aspect(None)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(-0.5, height - 0.5)
        axis.set_autoscalex_on(False)
        axis.set_autoscaley_on(False)
        axis.set_facecolor("#11191d")
        axis.set_axis_off()

    def plot_scalar_overlay(
        self,
        axis: Any,
        field: torch.Tensor,
        field_class: str,
    ) -> float:
        prepared = prepare_scalar_display(field)
        if prepared is None:
            return 1.0
        display, limit = prepared
        artist = axis.imshow(
            display,
            cmap=PRIMAL_CMAP if field_class == "primal" else DUAL_CMAP,
            origin="lower",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        self.dynamic_artists[axis].append(artist)
        self.show_colorbar(axis, artist)
        return 1.0

    def plot_vector_overlay(
        self,
        axis: Any,
        field: torch.Tensor,
        field_class: str,
    ) -> float:
        values, x, y, factor = prepare_vector_display(
            field,
            spacing=self.vector_spacing,
        )
        if not y.numel():
            return factor
        artist = axis.quiver(
            x.numpy(),
            y.numpy(),
            (values[0, 0, y, x] * factor).numpy(),
            (values[0, 1, y, x] * factor).numpy(),
            color=field_color(field_class),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            zorder=8,
        )
        self.dynamic_artists[axis].append(artist)
        return factor

    def plot_field(self, axis: Any, field: torch.Tensor, field_class: str) -> float:
        channels = field.shape[-3]
        if channels == 1:
            return self.plot_scalar_overlay(axis, field, field_class)
        if channels == 2:
            return self.plot_vector_overlay(axis, field, field_class)
        raise ValueError(f"cannot display a field with {channels} channels")

    @staticmethod
    def current_image_tensor(
        source: torch.Tensor,
        cache: SplineTrajectory | None,
        image_mode: str,
        index: int,
    ) -> torch.Tensor:
        if cache is None:
            return source[0]
        return {
            "full": cache.images,
            "deformation": cache.deformed_source,
            "photometric": cache.photometric_only,
        }[image_mode][index]

    @staticmethod
    def displayed_image_symbol(
        cache: SplineTrajectory | None,
        image_mode: str,
    ) -> str:
        if cache is None:
            return "I_0"
        return {
            "full": "I(t)",
            "deformation": "I_D(t)",
            "photometric": r"I_{\mathrm{phot}}(t)",
        }[image_mode]

    @staticmethod
    def input_title(
        input_kind: str,
        control_index: int,
        parameters: SplineParameters,
    ) -> str:
        title = {
            "initial_momentum": r"Source + initial momentum $p_0$",
            "initial_acceleration": r"Source + initial acceleration $a_0$",
            "initial_jerk": r"Source + initial jerk $r_0$",
        }.get(input_kind)
        if title is not None:
            return title
        time = parameters.projected_control_times[control_index]
        return rf"Source + $r({time:.3g}^+)$"

    def render_source(
        self,
        source: torch.Tensor,
        field: torch.Tensor,
        input_kind: str,
        control_index: int,
        parameters: SplineParameters,
        drawing_amplitude: float,
        show_image: bool,
    ) -> None:
        self.clear_dynamic(self.source_ax)
        self.configure_image_axis(self.source_ax, source)
        self.source_image.set_data(
            source[0, 0] if show_image else torch.zeros_like(source[0, 0])
        )
        self.source_image.set_clim(0, 1)
        field_class = (
            "primal" if input_kind == "initial_acceleration" else "dual"
        )
        self.plot_field(self.source_ax, field, field_class)
        title = self.input_title(input_kind, control_index, parameters)
        if not show_image:
            title = title.removeprefix("Source + ")
        self.source_ax.set_title(
            scaled_field_title(title, drawing_amplitude),
            color=INK_COLOR,
            fontsize=11,
            pad=9,
        )
        expression = {
            "initial_momentum": r"\Vert p_0\Vert_{I_0}^2",
            "initial_acceleration": r"\Vert a_0\Vert_{I_0}^2",
            "initial_jerk": r"\Vert r_0\Vert_{I_0^*}^2",
        }.get(input_kind)
        if expression is None:
            time = parameters.projected_control_times[control_index]
            expression = rf"\Vert r({time:.3g}^+)\Vert_{{I_0^*}}^2"
        norm = (
            metric_squared_norm
            if input_kind in ("initial_momentum", "initial_acceleration")
            else cometric_squared_norm
        )
        value = 0.0 if torch.count_nonzero(field) == 0 else norm(
            source, field, parameters
        )
        self.source_footer.set_text(rf"${expression} = {latex_number(value)}$")

    def render_current(
        self,
        source: torch.Tensor,
        cache: SplineTrajectory | None,
        image_mode: str,
        current_field: str | None,
        index: int,
        show_image: bool,
    ) -> None:
        self.clear_dynamic(self.current_ax)
        self.configure_image_axis(self.current_ax, source)
        current = self.current_image_tensor(source, cache, image_mode, index)
        _set_current_image(
            self.current_image,
            current,
            show_image=show_image,
            photometric=image_mode == "photometric" and cache is not None,
        )
        title = _current_title(cache, image_mode, current_field, index, show_image)
        factor = 1.0
        if cache is not None and current_field is not None:
            factor = self.plot_field(
                self.current_ax,
                cache.field(current_field)[index],
                FIELD_CLASS[current_field],
            )
        elif cache is None:
            message = self.current_ax.text(
                0.5,
                0.5,
                "Run spline or classic",
                transform=self.current_ax.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                bbox={
                    "facecolor": INK_COLOR,
                    "alpha": 0.88,
                    "edgecolor": "none",
                    "pad": 7,
                },
            )
            self.dynamic_artists[self.current_ax].append(message)
        self.current_ax.set_title(
            scaled_field_title(title, factor),
            color=INK_COLOR,
            fontsize=11,
            pad=9,
        )

        if cache is None or current_field is None:
            self.current_footer.set_text("")
            return
        expression = {
            "momentum": r"\Vert p(t)\Vert_{I_t}^2",
            "force": r"\Vert u(t)\Vert_{I_t^*}^2",
            "acceleration": r"\Vert a(t)\Vert_{I_t}^2",
            "jerk": r"\Vert r(t)\Vert_{I_t^*}^2",
            "velocity": r"\Vert v(t)\Vert_V^2",
            "vector_momentum": r"\Vert m(t)\Vert_{V^*}^2",
        }[current_field]
        value = cache.field_energy(current_field, index)
        self.current_footer.set_text(rf"${expression} = {latex_number(value)}$")

    def render_target(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        cache: SplineTrajectory | None,
        image_mode: str,
        target_mode: str,
        index: int,
        target_index: int,
        target_number: int,
        target_count: int,
        target_time: float | None,
        loss_curves: dict[str, torch.Tensor] | None = None,
        shown_loss_curves: tuple[bool, bool, bool] = (True, True, True),
        regularized_loss_label: str = "Regularized cost",
    ) -> None:
        self.clear_dynamic(self.target_ax)
        if target_mode == "Global loss":
            self.target_image.set_visible(False)
            self.target_ax.set_axis_on()
            self.target_ax.set_facecolor(PANEL_COLOR)
            self.target_ax.grid(False)
            self.target_ax.set_aspect("auto")
            self.target_ax.set_box_aspect(source.shape[-2] / source.shape[-1])
            self.target_ax.set_yscale("linear")
            self.target_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            labels = {
                "full": "Full loss",
                "data": "Data loss",
                "regularized": regularized_loss_label,
            }
            colors = {
                "full": "C0",
                "data": "C1",
                "regularized": "C2",
            }
            if loss_curves is None:
                message = self.target_ax.text(
                    0.5,
                    0.5,
                    "Register a model to show loss curves",
                    transform=self.target_ax.transAxes,
                    ha="center",
                    va="center",
                    color=INK_COLOR,
                    fontsize=11,
                )
                self.dynamic_artists[self.target_ax].append(message)
            else:
                for name, values in compress(loss_curves.items(), shown_loss_curves):
                    values = torch.as_tensor(values).detach().cpu()
                    line, = self.target_ax.plot(
                        np.arange(len(values)),
                        values.numpy(),
                        label=labels[name],
                        color=colors[name],
                    )
                    self.dynamic_artists[self.target_ax].append(line)
                if self.target_ax.lines:
                    legend = self.target_ax.legend()
                    self.dynamic_artists[self.target_ax].append(legend)
                    self.target_ax.relim(visible_only=True)
                    self.target_ax.set_autoscalex_on(True)
                    self.target_ax.set_autoscaley_on(True)
                    self.target_ax.autoscale_view()
                else:
                    self.target_ax.set_xlim(0, 1)
                    self.target_ax.set_ylim(0, 1)
            self.target_ax.set_xlabel("Outer iteration")
            self.target_ax.set_ylabel("Objective value")
            self.target_ax.set_title("Optimization loss")
            self.target_footer.set_text("")
            return

        displayed = self.current_image_tensor(source, cache, image_mode, index)
        self.target_image.set_visible(True)
        self.configure_image_axis(self.target_ax, source)
        if target_mode == "Target":
            self.target_image.set_data(target[0, 0])
            self.target_image.set_cmap("gray")
            self.target_image.set_clim(0, 1)
            location = "unplaced" if target_time is None else f"t={target_time:.3g}"
            title = f"Target {target_number}/{target_count} at {location}"
        else:
            error = (displayed - target[0]).abs()[0]
            maximum = max(float(torch.quantile(error.flatten(), 0.99)), 1e-8)
            self.target_image.set_data(error)
            self.target_image.set_cmap("magma")
            self.target_image.set_clim(0, maximum)
            self.show_colorbar(self.target_ax, self.target_image)
            symbol = self.displayed_image_symbol(cache, image_mode)
            title = (
                rf"Absolute error $|{symbol}-I_{{{target_number}}}|$"
            )
        self.target_ax.set_title(title, color=INK_COLOR, fontsize=11, pad=9)

        mse = (
            float(cache.target_mse[target_index, index])
            if cache is not None and image_mode == "full"
            else float((displayed - target[0]).square().mean())
        )
        symbol = self.displayed_image_symbol(cache, image_mode)
        self.target_footer.set_text(
            rf"$\mathrm{{MSE}}({symbol},I_{{\mathrm{{target}}}})"
            rf" = \frac{{1}}{{|\Omega|}}\Vert {symbol}-I_{{\mathrm{{target}}}}\Vert_2^2"
            rf" = {latex_number(mse)}$"
        )
