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
from draft.playground.field_playground import (
    DUAL_COLOR,
    FieldPlayground,
    LoadedField,
    PRIMAL_COLOR,
    add_vector_arrow,
    analyze_field,
    coerce_field,
    coerce_image,
    erase_stroke,
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
    image = coerce_image(np.full((1, 6, 7, 1), 0.3, dtype=np.float32))
    assert image.shape == (1, 1, 6, 7)
    assert torch.allclose(image, torch.full_like(image, 0.3))
    with pytest.raises(ValueError, match="one channel"):
        coerce_image(np.zeros((6, 7, 3), dtype=np.float32))

    scalar, mode = coerce_field(np.zeros((6, 7, 1), dtype=np.float32))
    assert scalar.shape == (1, 1, 6, 7)
    assert mode == "scalar"
    with pytest.raises(ValueError, match="ambiguous"):
        coerce_field(np.zeros((1, 7, 2), dtype=np.float32))
    assert coerce_field(torch.zeros(1, 2, 7, 2))[0].shape == (1, 2, 7, 2)
    assert coerce_image(torch.zeros(1, 1, 7, 3)).shape == (1, 1, 7, 3)

    float64_field, _ = coerce_field(np.zeros((6, 7), dtype=np.float64))
    assert float64_field.dtype == torch.float64

    path = tmp_path / "momentum.npz"
    np.savez(path, vector_momentum=np.zeros((6, 7, 2), dtype=np.float32))
    loaded = load_field_file(path)
    assert loaded.kind == "vector_momentum"
    assert loaded.field.shape == (1, 2, 6, 7)

    conflicting = tmp_path / "conflicting.npz"
    np.savez(
        conflicting,
        vector_momentum=np.zeros((6, 7, 2), dtype=np.float32),
        field_kind=np.array("velocity"),
    )
    with pytest.raises(ValueError, match="conflicts"):
        load_field_file(conflicting)


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
    assert result.kernel_response.shape == (1, 2, 20, 24)
    assert result.relative_roundtrip < 1e-4
    assert result.operator_time >= 0
    operator = SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3)
    cometric = CometricOperator(image, 0.25, operator)
    expected_response = operator.apply_inverse(
        covector * cometric.image_gradient[:, 0]
    )
    torch.testing.assert_close(result.kernel_response, expected_response)

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
    assert inverse.solver_iterations >= 1
    assert inverse.solver_time >= 0


def test_cometric_inverse_exposes_solver_info():
    torch.manual_seed(5)
    image = torch.rand(1, 1, 8, 9)
    acceleration = torch.rand_like(image)
    solution, iterations, elapsed = CometricOperator(
        image,
        0.25,
        lambda value: value,
    ).inverse(
        acceleration,
        eps=1e-7,
        return_info=True,
    )
    assert solution.shape == acceleration.shape
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
    solution, iterations = conjugate_gradient(matrix.mv, rhs, 1e-12)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    assert iterations == 2

    solution, iterations = conjugate_gradient(
        matrix.mv, torch.zeros_like(rhs), 1e-12
    )
    assert torch.equal(solution, torch.zeros_like(rhs))
    assert iterations == 0


def test_invalid_cg_tolerance_and_field_file_inputs(tmp_path):
    image = torch.zeros(1, 1, 8, 9)
    acceleration = torch.zeros_like(image)
    operator = lambda value: value
    with pytest.raises(ValueError, match="eps"):
        CometricOperator(image, 0.5, operator).inverse(acceleration, eps=0)

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {"field": torch.zeros(1, 1, 8, 9), "field_kind": "velocity"},
        invalid,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_field_file(invalid)


def test_saved_template_reloads_and_headless_ui_renders(tmp_path):
    image = torch.zeros(1, 1, 32, 36)
    image[..., 8:24, 10:26] = 1
    app = FieldPlayground(image, device="cpu")
    assert app.brush_slider.val == 1
    assert app.alpha_slider.val == pytest.approx(0.2)
    assert app.beta_slider.val == pytest.approx(0.2)
    assert app.gamma_slider.valtext.get_text() == "0.001"
    assert app.analysis is None
    assert app.kind == "vector_momentum"
    radio_colors = app.kind_radio._buttons.get_facecolors()
    np.testing.assert_allclose(radio_colors[1], to_rgba(DUAL_COLOR))
    assert radio_colors[0, 3] == 0
    app.mode_radio.set_active(1)
    assert app.kind == "u"
    app.mode_radio.set_active(0)

    display_field = torch.full((1, 2, 32, 36), 1e-3)
    display_field[0, :, 16, 18] = torch.tensor([5.0, 2.0])
    app.fields["vector"] = display_field
    app._invalidate("test")
    arrows = [artist for artist in app.input_ax.collections if isinstance(artist, Quiver)]
    assert len(arrows) == 1
    assert arrows[0].U.size == 1
    assert arrows[0].U[0] == pytest.approx(5.0)
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
    assert "$K$ time=" in output_footer
    assert "q_{95}" not in output_footer
    output_arrows = [
        artist for artist in app.output_ax.collections if isinstance(artist, Quiver)
    ]
    np.testing.assert_allclose(
        output_arrows[0].get_facecolors()[0], to_rgba(PRIMAL_COLOR)
    )
    assert app.output_ax.title.get_color() == "#24333b"
    energy = app.norm_text.get_text()
    assert r"\Vert v\Vert_V^2=\Vert m\Vert_{V^*}^2" in energy
    assert r"\langle" not in energy
    assert app.norm_text.axes is app.input_ax
    assert app.norm_text.get_position() == (0.5, -0.075)
    assert isinstance(app.error_colorbar.mappable.norm, LogNorm)
    assert app.error_colorbar.ax.get_ylabel() == ""
    error = (app.analysis.roundtrip - app.fields["vector"]).square().sum(1).sqrt()
    precision_floor = float(
        torch.finfo(error.dtype).eps
        * app.fields["vector"].norm()
        / np.sqrt(error.numel())
    )
    assert app.error_colorbar.mappable.norm.vmin == pytest.approx(precision_floor)
    heatmap = app.error_colorbar.mappable.get_array()
    assert not np.ma.getmaskarray(heatmap).any()
    assert float(heatmap.min()) == pytest.approx(precision_floor)
    assert r"\left\Vert L(Km)-m\right\Vert_2" in app.detail_ax.get_title()
    error_legend = app.detail_ax.texts[-1].get_text()
    assert r"\mathrm{mean}" in error_legend
    assert r"\mathrm{max}" in error_legend
    assert "q_{99}" not in error_legend
    assert "|e|" not in error_legend
    reference = app.fields["vector"].square().sum(1).sqrt()
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
    assert r"\Vert v\Vert_V^2=\Vert m\Vert_{V^*}^2" in app.norm_text.get_text()
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
    assert r"\left\Vert K(Lv)-v\right\Vert_2" in app.detail_ax.get_title()
    assert app.error_colorbar.ax.get_ylabel() == ""
    assert "$L$ time=" in app.output_ax.texts[-1].get_text()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"

    app.mode_radio.set_active(1)
    assert app.error_colorbar is None
    app.fields["scalar"].fill_(0.1)
    app.kind_radio.set_active(0)
    app.run()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"
    kernel_arrows = [
        artist for artist in app.detail_ax.collections if isinstance(artist, Quiver)
    ]
    np.testing.assert_allclose(
        kernel_arrows[0].get_facecolors()[0], to_rgba(PRIMAL_COLOR)
    )
    solver_legend = app.output_ax.texts[-1].get_text()
    assert solver_legend.count("\n") == 3
    assert "iterations" in solver_legend
    assert r"$A_I^{-1}$ time=" in solver_legend
    assert r"\mathrm{mean}" in solver_legend
    assert r"\mathrm{max}" in solver_legend
    assert "error" not in solver_legend
    scalar_energy = app.norm_text.get_text()
    assert r"\Vert a\Vert_{A_I^{-1}}^2=\Vert u\Vert_{A_I}^2" in scalar_energy
    assert r"\langle" not in scalar_energy
    scalar_panel_artist_ids = tuple(
        id(artist)
        for axis in (app.input_ax, app.output_ax)
        for artists in (axis.images, axis.collections, axis.texts)
        for artist in artists
    )
    app.spacing_slider.set_val(8)
    assert scalar_panel_artist_ids == tuple(
        id(artist)
        for axis in (app.input_ax, app.output_ax)
        for artists in (axis.images, axis.collections, axis.texts)
        for artist in artists
    )
    assert app.norm_text.get_text() == scalar_energy
    app.kind_radio.set_active(1)
    app.run()
    assert app.input_ax.title.get_color() == "#24333b"
    assert app.output_ax.title.get_color() == "#24333b"
    cometric_legend = app.output_ax.texts[-1].get_text()
    assert "$A_I$ time=" in cometric_legend
    assert r"\mathrm{mean}" in cometric_legend
    assert r"\mathrm{max}" in cometric_legend
    app.mode_radio.set_active(0)
    app.kind_radio.set_active(1)

    output = app.save(tmp_path / "field.pt")
    app.fig.savefig(tmp_path / "playground.png")
    loaded = load_field_file(output)

    assert loaded.kind == "vector_momentum"
    assert torch.equal(loaded.field, edited_field)
    assert loaded.image.shape == image.shape
    assert (tmp_path / "playground.png").stat().st_size > 10_000
    plt.close(app.fig)
