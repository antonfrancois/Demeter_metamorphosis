"""Panel rendering and field-overlay primitives for the spline playground."""

from typing import Any

from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np
import torch

from draft.playground.field_playground_core import (
    prepare_scalar_display,
    prepare_vector_display,
    scaled_field_title,
)

from .core import SplineParameters, SplineTrajectory, cometric_squared_norm
from .styles import (
    DUAL_COLOR,
    FIELD_CLASS,
    FIELD_SYMBOL,
    INK_COLOR,
    PRIMAL_COLOR,
)


def _signed_cmap(name: str, color: str) -> LinearSegmentedColormap:
    red, green, blue, _ = to_rgba(color)
    dark = (0.32 * red, 0.32 * green, 0.32 * blue, 0.92)
    bright = (red, green, blue, 0.92)
    transparent = (red, green, blue, 0.0)
    return LinearSegmentedColormap.from_list(
        name,
        ((0.0, dark), (0.49, transparent), (0.51, transparent), (1.0, bright)),
        N=257,
    )


DUAL_CMAP = _signed_cmap("spline_dual", DUAL_COLOR)
PRIMAL_CMAP = _signed_cmap("spline_primal", PRIMAL_COLOR)


def field_color(field_class: str) -> str:
    return PRIMAL_COLOR if field_class == "primal" else DUAL_COLOR


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
        footers: tuple[Any, Any, Any],
        dynamic_artists: dict[Any, list[Any]],
    ) -> None:
        self.source_ax, self.current_ax, self.target_ax = axes
        self.source_image, self.current_image, self.target_image = images
        self.source_footer, self.current_footer, self.target_footer = footers
        self.dynamic_artists = dynamic_artists

    def clear_dynamic(self, axis: Any) -> None:
        for artist in self.dynamic_artists[axis]:
            try:
                artist.remove()
            except ValueError:
                pass
        self.dynamic_artists[axis].clear()

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
        return 1.0

    def plot_vector_overlay(
        self,
        axis: Any,
        field: torch.Tensor,
        field_class: str,
    ) -> float:
        values, x, y, factor = prepare_vector_display(field)
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
        if input_kind == "initial_momentum":
            return r"Source + initial momentum $p_0$"
        if input_kind == "initial_force":
            return r"Source + initial force $u_0$"
        if input_kind == "initial_jerk":
            return r"Source + initial jerk $r_0$"
        time = parameters.mesh_control_times[control_index]
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
        self.source_image.set_data(
            source[0, 0] if show_image else torch.zeros_like(source[0, 0])
        )
        self.source_image.set_clim(0, 1)
        self.plot_field(self.source_ax, field, "dual")
        title = self.input_title(input_kind, control_index, parameters)
        if not show_image:
            title = title.removeprefix("Source + ")
        self.source_ax.set_title(
            scaled_field_title(title, drawing_amplitude),
            color=INK_COLOR,
            fontsize=11,
            pad=9,
        )
        if input_kind in ("initial_momentum", "initial_force"):
            symbol = "p_0" if input_kind == "initial_momentum" else "u_0"
            expression = rf"\Vert {symbol}\Vert_{{A_{{I_0}}}}^2"
            value = (
                0.0
                if torch.count_nonzero(field) == 0
                else cometric_squared_norm(source, field, parameters)
            )
        elif input_kind == "initial_jerk":
            value = float(field.square().sum())
            expression = r"\Vert r_0\Vert_2^2"
        else:
            value = float(field.square().sum())
            time = parameters.mesh_control_times[control_index]
            expression = rf"\Vert r({time:.3g}^+)\Vert_2^2"
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
        current = self.current_image_tensor(source, cache, image_mode, index)
        self.current_image.set_data(
            current[0] if show_image else torch.zeros_like(current[0])
        )
        self.current_image.set_cmap("gray")
        if not show_image:
            self.current_image.set_clim(0, 1)
        elif image_mode == "photometric" and cache is not None:
            lower = min(0.0, float(torch.quantile(current.flatten(), 0.01)))
            upper = max(1.0, float(torch.quantile(current.flatten(), 0.99)))
            self.current_image.set_clim(lower, upper)
        else:
            self.current_image.set_clim(0, 1)

        if not show_image:
            if cache is None:
                title = "Current field (run required)"
            elif current_field is None:
                title = "No field overlay"
            else:
                title = rf"${FIELD_SYMBOL[current_field]}$"
        elif cache is None:
            title = "Current image (run required)"
        else:
            title = {
                "full": rf"Current image $I_{{{index}}}$",
                "deformation": rf"Deformation only $I_{{D,{index}}}$",
                "photometric": rf"Photometric only $I_{{\mathrm{{phot}},{index}}}$",
            }[image_mode]
        factor = 1.0
        if cache is not None and current_field is not None:
            factor = self.plot_field(
                self.current_ax,
                cache.field(current_field)[index],
                FIELD_CLASS[current_field],
            )
            if show_image:
                title += rf" + ${FIELD_SYMBOL[current_field]}$"
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
            "momentum": r"\Vert p(t)\Vert_{A_{I(t)}}^2",
            "force": r"\Vert u(t)\Vert_{A_{I(t)}}^2",
            "acceleration": r"\Vert a(t)\Vert_{A_{I(t)}^{-1}}^2",
            "jerk": r"\Vert r(t)\Vert_2^2",
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
        target_count: int,
        target_time: float | None,
    ) -> None:
        self.clear_dynamic(self.target_ax)
        displayed = self.current_image_tensor(source, cache, image_mode, index)
        if target_mode == "Target":
            self.target_image.set_data(target[0, 0])
            self.target_image.set_cmap("gray")
            self.target_image.set_clim(0, 1)
            location = "unplaced" if target_time is None else f"t={target_time:.3g}"
            title = f"Target {target_index + 1}/{target_count} at {location}"
        else:
            error = (displayed - target[0]).abs()[0]
            maximum = max(float(torch.quantile(error.flatten(), 0.99)), 1e-8)
            self.target_image.set_data(error)
            self.target_image.set_cmap("magma")
            self.target_image.set_clim(0, maximum)
            symbol = self.displayed_image_symbol(cache, image_mode)
            title = (
                rf"Absolute error $|{symbol}-I_{{{target_index + 1}}}|$"
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
