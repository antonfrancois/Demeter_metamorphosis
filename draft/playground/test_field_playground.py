import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.quiver import Quiver
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from draft.cometric_inversion import CometricOperator
from draft.conjugate_gradient import conjugate_gradient
from draft.sobolevfluid_operator import SobolevFluidOperator
from draft.playground.field_playground_core import (
    TIMING_SAMPLES,
    TIMING_WARMUPS,
    _outlier_filtered_mean,
    _timed_call,
)
from draft.playground.field_playground import (
    DEFAULT_IMAGE,
    DUAL_COLOR,
    FieldPlayground,
    LoadedField,
    PRIMAL_COLOR,
    SIGNED_FIELD_CMAP,
    VECTOR_DISPLAY_RELATIVE_THRESHOLD,
    add_vector_arrow,
    analyze_field,
    coerce_field,
    coerce_image,
    erase_stroke,
    load_image,
    load_field_file,
    paint_scalar_stroke,
    resize_field,
)


def test_vector_and_scalar_editing_primitives():
    vector = torch.zeros(1, 2, 31, 41)
    vector = add_vector_arrow(vector, (20, 15), (25, 12), sigma=3, gain=2)
    assert torch.allclose(vector[0, :, 15, 20], torch.tensor([10.0, -6.0]))
    maximum_index = vector.square().sum(dim=1)[0].argmax()
    assert divmod(int(maximum_index), vector.shape[-1]) == (15, 20)
    assert vector[0, :, 0, 0].norm() < 1e-6

    scalar = torch.zeros(1, 1, 31, 41)
    scalar = paint_scalar_stroke(scalar, [(8, 10), (30, 20)], sigma=2, amplitude=-1.5)
    assert scalar.min() < -1.4
    erased = erase_stroke(scalar, [(8, 10)], sigma=3)
    assert erased[0, 0, 10, 8].abs() < 1e-6
    assert erased[0, 0, 20, 30] < -1


def test_timing_uses_warmup_and_outlier_filtered_mean():
    regular = [0.9, 1.0, 1.0, 1.0, 1.1, 1.1]
    assert _outlier_filtered_mean(regular + [10.0]) == pytest.approx(
        np.mean(regular)
    )

    calls = 0

    def identity(value):
        nonlocal calls
        calls += 1
        return value

    value = torch.ones(1)
    result, elapsed = _timed_call(identity, value)
    assert result is value
    assert calls == TIMING_WARMUPS + TIMING_SAMPLES
    assert elapsed >= 0


def test_field_layout_coercion_and_vector_resize():
    grid_field = np.zeros((10, 20, 2), dtype=np.float32)
    grid_field[..., 0] = 4
    field, mode = coerce_field(grid_field)
    assert mode == "vector"
    assert field.shape == (1, 2, 10, 20)

    resized = resize_field(field, (20, 10))
    assert resized.shape == (1, 2, 20, 10)
    assert torch.allclose(resized[:, 0], torch.full((1, 20, 10), 2.0))
    momentum_resized = resize_field(
        field, (20, 10), scale_vector_displacement=False
    )
    assert torch.allclose(momentum_resized[:, 0], torch.full((1, 20, 10), 4.0))


def test_channel_last_layouts_and_npz_semantics(tmp_path):
    default_image, default_path = load_image(DEFAULT_IMAGE)
    assert default_path == DEFAULT_IMAGE.resolve()
    assert default_image.shape[:2] == (1, 1)

    signed_colors = SIGNED_FIELD_CMAP(np.array([0.0, 0.5, 1.0]))
    assert signed_colors[1, 3] == pytest.approx(0)
    assert not np.any(np.all(signed_colors[:, :3] == 0, axis=1))
    assert not np.any(np.all(signed_colors[:, :3] == 1, axis=1))

    image = coerce_image(np.full((1, 6, 7, 1), 0.3, dtype=np.float32))
    assert image.shape == (1, 1, 6, 7)
    assert torch.allclose(image, torch.full_like(image, 0.3))

    scalar, mode = coerce_field(np.zeros((6, 7, 1), dtype=np.float32))
    assert scalar.shape == (1, 1, 6, 7)
    assert mode == "scalar"
    assert coerce_field(torch.zeros(1, 2, 7, 2))[0].shape == (1, 2, 7, 2)
    assert coerce_image(torch.zeros(1, 1, 7, 3)).shape == (1, 1, 7, 3)

    float64_field, _ = coerce_field(np.zeros((6, 7), dtype=np.float64))
    assert float64_field.dtype == torch.float64

    path = tmp_path / "momentum.npz"
    np.savez(path, vector_momentum=np.zeros((6, 7, 2), dtype=np.float32))
    loaded = load_field_file(path)
    assert loaded.kind == "vector_momentum"
    assert loaded.field.shape == (1, 2, 6, 7)


def test_vector_analysis_round_trips():
    image = torch.rand(1, 1, 25, 29)
    velocity = torch.randn(1, 2, 25, 29)
    result = analyze_field(
        image,
        velocity,
        "velocity",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
    )
    assert result.counterpart.shape == velocity.shape
    assert result.roundtrip.shape == velocity.shape
    assert result.relative_roundtrip < 2e-5
    assert result.squared_norm > 0
    assert result.operator_time >= 0


def test_sobolev_operator_round_trips_and_has_exact_nyquist_symbol():
    torch.manual_seed(3)
    for dtype, tolerance in ((torch.float32, 2e-4), (torch.float64, 1e-11)):
        field = torch.randn(1, 2, 32, 34, dtype=dtype)
        operator = SobolevFluidOperator(alpha=0.2, beta=0.2, gamma=0.001)
        sharp_flat = operator.apply_operator(operator.apply_inverse(field))
        flat_sharp = operator.apply_inverse(operator.apply_operator(field))
        assert (sharp_flat - field).norm() / field.norm() < tolerance
        assert (flat_sharp - field).norm() / field.norm() < tolerance

        other = torch.randn_like(field)
        torch.testing.assert_close(
            (operator.apply_operator(field) * other).sum(),
            (field * operator.apply_operator(other)).sum(),
        )
        assert (field * operator.apply_operator(field)).sum() > 0

        _, mixed_symbol, _ = operator._symbol(field)
        assert torch.count_nonzero(mixed_symbol[0]) == 0
        assert torch.count_nonzero(mixed_symbol[:, 0]) == 0
        assert torch.count_nonzero(mixed_symbol[16]) == 0
        assert torch.count_nonzero(mixed_symbol[:, -1]) == 0


def test_scalar_forward_and_inverse_round_trip():
    torch.manual_seed(4)
    image = torch.rand(1, 1, 20, 24)
    covector = torch.randn(1, 1, 20, 24) * 0.1
    result = analyze_field(
        image,
        covector,
        "u",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        cg_eps=1e-7,
    )
    assert result.counterpart.shape == covector.shape
    assert result.deformation_velocity.shape == (1, 2, 20, 24)
    assert result.relative_roundtrip < 1e-4
    assert result.operator_time >= 0
    operator = SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3)
    cometric = CometricOperator(image, 0.25, operator)
    expected_response = operator.apply_inverse(
        covector * cometric.image_gradient[:, 0]
    )
    expected_velocity = -(0.25**0.5) * expected_response
    torch.testing.assert_close(result.deformation_velocity, expected_velocity)
    assert (result.deformation_velocity * expected_response).sum() < 0
    vector_momentum = covector * cometric.image_gradient[:, 0]
    deformation_energy = 0.25 * float((vector_momentum * expected_response).sum())
    intensity_energy = 0.75 * float(covector.square().sum())
    assert result.deformation_energy_contribution == pytest.approx(deformation_energy)
    assert result.squared_norm == pytest.approx(
        intensity_energy + deformation_energy,
        rel=2e-6,
    )

    inverse = analyze_field(
        image,
        result.counterpart,
        "a",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        cg_eps=1e-7,
    )
    assert inverse.relative_roundtrip < 1e-4
    assert torch.allclose(inverse.counterpart, covector, atol=2e-5, rtol=2e-4)
    assert inverse.operator_time is None
    assert inverse.solver_residual <= 1e-7
    assert inverse.solver_iterations >= 1
    assert inverse.solver_time >= 0
    torch.testing.assert_close(
        inverse.deformation_velocity,
        expected_velocity,
        atol=2e-5,
        rtol=2e-4,
    )
    assert inverse.deformation_energy_contribution == pytest.approx(
        deformation_energy,
        rel=2e-4,
    )


def test_cometric_inverse_exposes_solver_info():
    torch.manual_seed(5)
    image = torch.rand(1, 1, 8, 9)
    acceleration = torch.rand_like(image)
    solution, iterations, elapsed, residual = CometricOperator(
        image,
        0.25,
        lambda value: value,
    ).inverse(
        acceleration,
        eps=1e-7,
        return_info=True,
    )
    assert solution.shape == acceleration.shape
    assert residual <= 1e-7
    assert iterations >= 1
    assert elapsed >= 0


def test_cometric_is_single_channel_and_keeps_idiomatic_call():
    image = torch.rand(1, 1, 8, 9)
    covector = torch.rand_like(image)
    acceleration = CometricOperator(image, 0.25, lambda value: value)(covector)
    assert acceleration.shape == covector.shape
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        CometricOperator(torch.rand(1, 2, 8, 9), 0.25, lambda value: value)


def test_relative_l2_metrics_are_rms_and_pointwise_maximum():
    error = torch.tensor([1.0, 3.0])
    reference = torch.tensor([2.0, 2.0])
    mean, maximum = FieldPlayground._relative_l2_metrics(error, reference)
    assert mean == pytest.approx(np.sqrt(5) / 2)
    assert maximum == pytest.approx(1.5)


def test_conjugate_gradient_solves_spd_system_and_zero_rhs():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    solution, iterations, residual = conjugate_gradient(matrix.mv, rhs, 1e-12)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    assert iterations == 2
    expected_residual = (rhs - matrix.mv(solution)).norm() / rhs.norm().clamp_min(1)
    assert residual == pytest.approx(float(expected_residual))

    solution, iterations, residual = conjugate_gradient(
        matrix.mv, torch.zeros_like(rhs), 1e-12
    )
    assert torch.equal(solution, torch.zeros_like(rhs))
    assert iterations == 0
    assert residual == 0


def test_cometric_rejects_nonpositive_cg_tolerance():
    image = torch.zeros(1, 1, 8, 9)
    acceleration = torch.zeros_like(image)
    operator = lambda value: value
    with pytest.raises(ValueError, match="eps"):
        CometricOperator(image, 0.5, operator).inverse(acceleration, eps=0)


def test_vector_edit_after_kind_switch_updates_analysis_and_display():
    image = torch.zeros(1, 1, 32, 36)
    app = FieldPlayground(image, device="cpu")
    app._on_press(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=3.0, ydata=6.0, button=1, key=None
        )
    )
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=9.0, ydata=6.0)
    )
    app._on_release(
        SimpleNamespace(inaxes=app.input_ax, xdata=9.0, ydata=6.0, button=1)
    )
    initial = app.fields["vector"].clone()
    app.run()

    app.kind_radio.set_active(0)
    assert app.kind == "velocity"
    app._on_press(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=24.0, ydata=20.0, button=1, key=None
        )
    )
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=24.0, ydata=27.0)
    )
    app._on_release(
        SimpleNamespace(inaxes=app.input_ax, xdata=24.0, ydata=27.0, button=1)
    )
    updated = app.fields["vector"]
    assert not torch.equal(updated, initial)

    app.run()
    operator = SobolevFluidOperator(alpha=0.2, beta=0.2, gamma=0.001)
    expected = operator.apply_operator(updated)
    stale = operator.apply_operator(initial)
    torch.testing.assert_close(app.analysis.counterpart, expected)
    assert not torch.allclose(app.analysis.counterpart, stale)

    quivers = [
        artist for artist in app.output_ax.collections if isinstance(artist, Quiver)
    ]
    assert len(quivers) == 1
    magnitude = expected.square().sum(1).sqrt()[0]
    visible = magnitude >= max(
        1e-8, VECTOR_DISPLAY_RELATIVE_THRESHOLD * float(magnitude.max())
    )
    factor = float(np.clip(0.06 * min(magnitude.shape), 12, 48)) / float(
        torch.quantile(magnitude[visible], 0.95)
    )
    x = torch.as_tensor(quivers[0].X, dtype=torch.long)
    y = torch.as_tensor(quivers[0].Y, dtype=torch.long)
    np.testing.assert_allclose(
        quivers[0].U,
        (expected[0, 0, y, x] * factor).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        quivers[0].V,
        (expected[0, 1, y, x] * factor).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert any(
        (x_coord - 24) ** 2 + (y_coord - 20) ** 2 <= 4
        for x_coord, y_coord in zip(quivers[0].X, quivers[0].Y)
    )
    plt.close(app.fig)


def test_saved_template_reloads_and_headless_ui_renders(tmp_path):
    image = torch.zeros(1, 1, 32, 36)
    image[..., 8:24, 10:26] = 1
    app = FieldPlayground(image, device="cpu")
    assert app.brush_slider.val == 1
    assert app.status_text.get_text() == "Loaded image size: 36 x 32 px (W x H)."
    app._set_status("Field cleared. Press Run.")
    assert app.status_text.get_text() == (
        "Field cleared. Press Run.\nLoaded image size: 36 x 32 px (W x H)."
    )
    assert app.alpha_slider.val == pytest.approx(0.2)
    assert app.beta_slider.val == pytest.approx(0.2)
    assert app.gamma_slider.valtext.get_text() == "0.001"
    assert app.analysis is None
    assert app.kind == "vector_momentum"
    assert not app.scalar_error_button.ax.get_visible()
    assert not app.scalar_deformation_button.ax.get_visible()
    radio_colors = app.kind_radio._buttons.get_facecolors()
    np.testing.assert_allclose(radio_colors[1], to_rgba(DUAL_COLOR))
    assert radio_colors[0, 3] == 0
    app.mode_radio.set_active(1)
    assert app.kind == "u"
    assert app.scalar_detail == "error"
    assert app.scalar_error_button.ax.get_visible()
    assert app.scalar_deformation_button.ax.get_visible()
    detail_position = app.detail_ax.get_position()
    button_left = app.scalar_error_button.ax.get_position()
    button_right = app.scalar_deformation_button.ax.get_position()
    assert (button_left.x0 + button_right.x1) / 2 == pytest.approx(
        (detail_position.x0 + detail_position.x1) / 2
    )
    assert 0 < button_left.y0 - detail_position.y1 < 0.08
    assert button_left.height >= 0.035
    app._set_scalar_detail("deformation")
    app.mode_radio.set_active(0)
    assert not app.scalar_error_button.ax.get_visible()
    assert not app.scalar_deformation_button.ax.get_visible()
    app.mode_radio.set_active(1)
    assert app.scalar_detail == "error"
    app.mode_radio.set_active(0)

    display_field = torch.full((1, 2, 32, 36), 1e-3)
    display_field[0, :, 16, 18] = torch.tensor([5.0, 2.0])
    app.fields["vector"] = display_field
    app._invalidate("test")
    arrows = [artist for artist in app.input_ax.collections if isinstance(artist, Quiver)]
    assert len(arrows) == 1
    assert arrows[0].U.size == 1
    target = float(np.clip(0.06 * min(display_field.shape[-2:]), 12, 48))
    assert arrows[0].U[0] == pytest.approx(5.0 * target / np.hypot(5.0, 2.0))
    np.testing.assert_allclose(arrows[0].get_facecolors()[0], to_rgba(DUAL_COLOR))
    assert app.input_ax.title.get_color() == "#24333b"
    assert "display" not in app.input_ax.get_title()
    field = add_vector_arrow(torch.zeros(1, 2, 32, 36), (12, 12), (20, 17), 4)
    app.set_template(
        LoadedField(
            field,
            "vector_momentum",
            metadata={
                "parameters": {"alpha": 3.0, "beta": 0.3, "gamma": 10.0, "rho": 0.2}
            },
        ),
        "test template",
    )
    assert app.alpha_slider.val == pytest.approx(3.0)
    assert 10 ** app.gamma_slider.val == pytest.approx(10.0)
    assert app.rho_slider.val == pytest.approx(0.2)
    assert app.analysis is None

    press = SimpleNamespace(
        inaxes=app.input_ax, xdata=12.0, ydata=12.0, button=1, key=None
    )
    app._on_press(press)
    app._on_motion(
        SimpleNamespace(inaxes=app.output_ax, xdata=0.5, ydata=0.5)
    )
    assert app.drag.points == [(12.0, 12.0)]
    app._cancel_drag()

    app._on_press(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=20.0, ydata=20.0, button=1, key=None
        )
    )
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=25.0, ydata=24.0)
    )
    preview_artist = app.drag.artist
    assert app.drag.points == [(20.0, 20.0)]
    assert app.drag.end == (25.0, 24.0)
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=27.0, ydata=26.0)
    )
    assert app.drag.artist is preview_artist
    np.testing.assert_allclose(preview_artist.get_edgecolor(), to_rgba(DUAL_COLOR))
    assert app.drag.points == [(20.0, 20.0)]
    assert len(app.input_ax.patches) == 1
    assert app.drag.background is not None
    app._on_release(
        SimpleNamespace(inaxes=app.input_ax, xdata=27.0, ydata=26.0)
    )
    assert app.analysis is None
    assert app.drag is None
    edited_field = app.fields["vector"].clone()
    expected_field = add_vector_arrow(field, (20, 20), (27, 26), sigma=1, gain=1)
    assert torch.equal(edited_field, expected_field)

    app.run()
    assert app.analysis is not None
    assert app.output_ax.get_title().startswith(r"Output: velocity $v=Km$")
    assert "display" not in app.output_ax.get_title()
    output_footer = app.output_ax.texts[-1].get_text()
    assert "$K$ avg time = " in output_footer
    assert "q_{95}" not in output_footer
    assert app.output_ax.texts[-1].get_fontsize() == app.output_ax.title.get_fontsize()
    output_arrows = [
        artist for artist in app.output_ax.collections if isinstance(artist, Quiver)
    ]
    np.testing.assert_allclose(
        output_arrows[0].get_facecolors()[0], to_rgba(PRIMAL_COLOR)
    )
    assert app.output_ax.title.get_color() == "#24333b"
    energy = app.norm_text.get_text()
    assert r"\Vert v\Vert_V^2 = \Vert m\Vert_{V^*}^2" in energy
    assert r"\langle" not in energy
    assert app.norm_text.axes is app.input_ax
    assert app.norm_text.get_position() == (0.5, -0.075)
    assert app.norm_text.get_fontsize() == app.input_ax.title.get_fontsize()
    assert isinstance(app.error_colorbar.mappable.norm, LogNorm)
    assert app.error_colorbar.ax.get_ylabel() == ""
    error = (app.analysis.roundtrip - app.fields["vector"]).square().sum(1).sqrt()
    precision_floor = torch.finfo(error.dtype).eps
    assert app.error_colorbar.mappable.norm.vmin == pytest.approx(precision_floor)
    heatmap = app.error_colorbar.mappable.get_array()
    assert not np.ma.getmaskarray(heatmap).any()
    assert float(heatmap.min()) == pytest.approx(precision_floor)
    assert r"\Vert L(Km)-m\Vert_2/\mathrm{RMS}(\Vert m\Vert_2)" in (
        app.detail_ax.get_title()
    )
    error_legend = app.detail_ax.texts[-1].get_text()
    assert r"\mathrm{mean} = " in error_legend
    assert r"\mathrm{max} = " in error_legend
    assert ":" not in error_legend
    assert app.detail_ax.texts[-1].get_fontsize() == app.detail_ax.title.get_fontsize()
    assert "q_{99}" not in error_legend
    assert "|e|" not in error_legend
    reference = app.fields["vector"].square().sum(1).sqrt()
    relative_error = app._relative_l2_error(error, reference)
    np.testing.assert_allclose(
        heatmap,
        relative_error[0].clamp_min(precision_floor).numpy(),
        rtol=1e-5,
        atol=0,
    )
    relative_mean, relative_max = app._relative_l2_metrics(error, reference)
    assert relative_mean == pytest.approx(app.analysis.relative_roundtrip)
    assert app._latex_number(relative_mean) in error_legend
    assert app._latex_number(relative_max) in error_legend
    assert relative_max >= app.analysis.relative_roundtrip
    error_colorbar = app.error_colorbar
    detail_artist_ids = tuple(
        id(artist)
        for artists in (app.detail_ax.images, app.detail_ax.collections, app.detail_ax.texts)
        for artist in artists
    )
    app.spacing_slider.set_val(7)
    assert app.error_colorbar is error_colorbar
    assert detail_artist_ids == tuple(
        id(artist)
        for artists in (app.detail_ax.images, app.detail_ax.collections, app.detail_ax.texts)
        for artist in artists
    )
    assert r"\Vert v\Vert_V^2 = \Vert m\Vert_{V^*}^2" in app.norm_text.get_text()
    analysis = app.analysis
    app.kind_radio.set_active(app.kind_radio.index_selected)
    assert app.analysis is analysis
    app._on_press(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=10.0, ydata=10.0, button=1, key=None
        )
    )
    app._on_release(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=10.0, ydata=10.0, button=1
        )
    )
    assert app.analysis is analysis
    base_images = tuple(app._base_images.values())
    axes_with_colorbar = len(app.fig.axes)
    app.refresh()
    assert len(app.fig.axes) == axes_with_colorbar
    assert tuple(app._base_images.values()) == base_images
    assert all(base in axis.images for axis, base in app._base_images.items())
    app.kind_radio.set_active(0)
    app.run()
    radio_colors = app.kind_radio._buttons.get_facecolors()
    np.testing.assert_allclose(radio_colors[0], to_rgba(PRIMAL_COLOR))
    assert radio_colors[1, 3] == 0
    assert r"\Vert K(Lv)-v\Vert_2/\mathrm{RMS}(\Vert v\Vert_2)" in (
        app.detail_ax.get_title()
    )
    assert app.error_colorbar.ax.get_ylabel() == ""
    assert "$L$ avg time = " in app.output_ax.texts[-1].get_text()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"

    app.mode_radio.set_active(1)
    assert app.error_colorbar is None
    assert app.scalar_detail == "error"
    assert app.scalar_error_button.ax.get_visible()
    assert app.scalar_deformation_button.ax.get_visible()
    app.fields["scalar"].fill_(0.1)
    app.kind_radio.set_active(0)
    app.run()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"
    assert app.input_ax.images[-1].get_cmap().name == SIGNED_FIELD_CMAP.name
    assert app.output_ax.images[-1].get_cmap().name == SIGNED_FIELD_CMAP.name
    assert isinstance(app.error_colorbar.mappable.norm, LogNorm)
    assert r"\left|A_I(A_I^{-1}a)-a\right|" in app.detail_ax.get_title()
    assert r"\mathrm{RMS}(\left|a\right|)" in app.detail_ax.get_title()
    scalar_error_legend = app.detail_ax.texts[-1].get_text()
    assert r"\mathrm{mean} = " in scalar_error_legend
    assert r"\mathrm{max} = " in scalar_error_legend
    assert ":" not in scalar_error_legend
    solver_legend = app.output_ax.texts[-1].get_text()
    solver_lines = solver_legend.splitlines()
    assert len(solver_lines) == 3
    assert r"\mathrm{residual} = " in solver_lines[0]
    assert app._latex_number(app.analysis.solver_residual) in solver_lines[0]
    assert r"\mathrm{iterations} = " in solver_lines[1]
    assert r"$A_I^{-1}$ avg time = " in solver_lines[2]
    assert r"\mathrm{mean}" not in solver_legend
    assert r"\mathrm{max}" not in solver_legend
    assert app.make_payload()["diagnostics"]["solver_residual"] == (
        app.analysis.solver_residual
    )
    scalar_energy = app.norm_text.get_text()
    assert r"\Vert a\Vert_{A_I^{-1}}^2 = \Vert u\Vert_{A_I}^2" in scalar_energy
    assert r"\langle" not in scalar_energy
    scalar_panel_artist_ids = tuple(
        id(artist)
        for axis in (app.input_ax, app.output_ax, app.detail_ax)
        for artists in (axis.images, axis.collections, axis.texts)
        for artist in artists
    )
    scalar_error_colorbar = app.error_colorbar
    app.spacing_slider.set_val(8)
    assert scalar_panel_artist_ids == tuple(
        id(artist)
        for axis in (app.input_ax, app.output_ax, app.detail_ax)
        for artists in (axis.images, axis.collections, axis.texts)
        for artist in artists
    )
    assert app.error_colorbar is scalar_error_colorbar
    assert app.norm_text.get_text() == scalar_energy

    app._set_scalar_detail("deformation")
    assert app.error_colorbar is None
    assert app.scalar_deformation_button.label.get_color() == "white"
    deformation_arrows = [
        artist for artist in app.detail_ax.collections if isinstance(artist, Quiver)
    ]
    assert len(deformation_arrows) == 1
    np.testing.assert_allclose(
        deformation_arrows[0].get_facecolors()[0], to_rgba(PRIMAL_COLOR)
    )
    assert app.detail_ax.get_title().startswith(
        r"Deformation velocity $-\sqrt{\rho}\,K(u\nabla I)$"
    )
    deformation_energy = app.detail_ax.texts[-1].get_text()
    assert r"\rho\,\Vert K(u\nabla I)\Vert_V^2 = " in deformation_energy
    assert app._latex_number(app.analysis.deformation_energy_contribution) in (
        deformation_energy
    )
    assert app.detail_ax.texts[-1].get_fontsize() == app.detail_ax.title.get_fontsize()
    displayed_velocity = app.analysis.deformation_velocity
    magnitude = displayed_velocity.square().sum(1).sqrt()[0]
    visible = magnitude >= max(
        1e-8, VECTOR_DISPLAY_RELATIVE_THRESHOLD * float(magnitude.max())
    )
    factor = float(np.clip(0.06 * min(magnitude.shape), 12, 48)) / float(
        torch.quantile(magnitude[visible], 0.95)
    )
    x = torch.as_tensor(deformation_arrows[0].X, dtype=torch.long)
    y = torch.as_tensor(deformation_arrows[0].Y, dtype=torch.long)
    np.testing.assert_allclose(
        deformation_arrows[0].U,
        (displayed_velocity[0, 0, y, x] * factor).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deformation_arrows[0].V,
        (displayed_velocity[0, 1, y, x] * factor).numpy(),
        rtol=1e-5,
        atol=1e-6,
    )

    app._set_scalar_detail("error")
    assert isinstance(app.error_colorbar.mappable.norm, LogNorm)
    app.kind_radio.set_active(1)
    app.run()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"
    assert r"A_I^{-1}(A_Iu)-u" in app.detail_ax.get_title()
    cometric_legend = app.output_ax.texts[-1].get_text()
    assert "$A_I$ avg time = " in cometric_legend
    assert r"\mathrm{mean}" not in cometric_legend
    assert r"\mathrm{max}" not in cometric_legend
    app.mode_radio.set_active(0)
    assert not app.scalar_error_button.ax.get_visible()
    assert not app.scalar_deformation_button.ax.get_visible()
    app.kind_radio.set_active(1)

    output = app.save(tmp_path / "field.pt")
    app.fig.savefig(tmp_path / "playground.png")
    loaded = load_field_file(output)

    assert loaded.kind == "vector_momentum"
    assert torch.equal(loaded.field, edited_field)
    assert loaded.image.shape == image.shape
    assert (tmp_path / "playground.png").stat().st_size > 10_000
    plt.close(app.fig)
