import os
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from demeter.metamorphosis.splines import MetamorphosisSplineIntegrator
from demeter.utils import torchbox as tb
from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import GaussianRKHS, SobolevFluidOperator
from demeter.utils.spline_data import (
    TimedImageBatch,
    load_timed_image_directory,
    save_timed_image_directory,
)
from draft.playground.field_playground_core import (
    load_field_file,
    prepare_vector_display,
    scaled_field_title,
)
from draft.playground.splines.app import DUAL_COLOR, SplinePlayground
from draft.playground.splines.core import (
    SolverSettings,
    SplineParameters,
    load_setup,
    minimum_mesh_steps,
    resolve_device,
    run_classic,
    run_spline,
    save_setup,
    zero_setup,
)
from draft.playground.splines.images import load_image
from draft.playground.splines.main import (
    _parameter_overrides,
    _replace_parameters,
    main as launch_playground,
)
from draft.playground.splines.menus.parameters import format_learning_rate
from draft.playground.splines.menus.observations import ObservationTimeEditor
from draft.playground.splines.project_io import load_project
from draft.playground.splines.registration import RegistrationResult, register_spline
from draft.playground.splines.rendering import DUAL_CMAP, PRIMAL_CMAP
from draft.playground.splines.styles import FIELD_CLASS, INK_COLOR


def make_parameters(
    *,
    steps: int = 16,
    cost: float = 0.01,
    iterations: int = 10,
    learning_rate: float = 1.0,
    **model,
) -> SplineParameters:
    return SplineParameters(
        spline=SolverSettings(cost, steps, iterations, learning_rate),
        **model,
    )


def test_parameters_require_ordered_interior_control_nodes():
    configured = make_parameters(steps=8, control_times=(0.25, 0.625))
    assert configured.control_nodes == (2, 5)
    assert configured.projected_control_times == (0.25, 0.625)
    assert configured.spline.learning_rate == pytest.approx(1.0)
    assert configured.optimized_fields == (
        "initial_momentum",
        "initial_acceleration",
        "initial_jerk",
    )
    assert configured.initialization == "cold"
    assert SplineParameters(initialization="WARM").initialization == "warm"
    independent_regression = SplineParameters(
        regression=SolverSettings(0.2, 8, 3, 0.4),
    )
    assert independent_regression.regression.cost == pytest.approx(0.2)
    assert independent_regression.regression.steps == 8
    assert independent_regression.regression.iterations == 3
    assert independent_regression.regression.learning_rate == pytest.approx(0.4)
    assert SplineParameters(model="classic").spline.learning_rate == pytest.approx(1.0)
    assert minimum_mesh_steps((0.25, 0.375, 1.0), max_steps=60) == 8

    midpoint = make_parameters(steps=16, control_times=(0.5,))
    refined = replace(midpoint, spline=replace(midpoint.spline, steps=40))
    assert refined.control_times == (0.5,)
    assert refined.control_nodes == (20,)
    restored = replace(refined, spline=replace(refined.spline, steps=16))
    assert restored.control_nodes == (8,)

    with pytest.raises(ValueError, match="strictly in"):
        make_parameters(steps=8, control_times=(0,))
    with pytest.raises(ValueError, match="final interior"):
        make_parameters(steps=8, control_times=(7 / 8,))
    with pytest.raises(ValueError, match="strictly increasing"):
        make_parameters(steps=8, control_times=(5 / 8, 2 / 8))
    with pytest.raises(ValueError, match="Sobolev"):
        SplineParameters(kernel="gaussian", model="splines")
    assert SplineParameters(rho=1, model="classic").rho == 1
    with pytest.raises(ValueError, match="rho"):
        SplineParameters(rho=1, model="splines")
    with pytest.raises(ValueError, match="learning_rate"):
        make_parameters(learning_rate=0)
    with pytest.raises(ValueError, match="initialization"):
        SplineParameters(initialization="previous")
    for values in (
        {"cost": 0},
        {"steps": 0},
        {"iterations": 0},
        {"learning_rate": 0},
    ):
        with pytest.raises((TypeError, ValueError)):
            SplineParameters(regression=SolverSettings(**values))
    with pytest.raises(ValueError, match="unknown field"):
        SplineParameters(optimized_fields=("control_jerks",))
    assert SplineParameters(
        optimized_fields=("initial_momentum", "initial_momentum")
    ).optimized_fields == ("initial_momentum",)
    with pytest.raises(ValueError, match="distinct"):
        make_parameters(
            steps=4,
            control_times=(0.5 - 1e-10, 0.5 + 1e-10),
        )
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected_device
    with pytest.raises(TypeError, match="progress_callback must be callable"):
        run_spline(
            zero_setup(torch.zeros(1, 1, 3, 4), parameters=make_parameters(steps=1)),
            device="cpu",
            progress_callback=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    ((1.0, "1"), (0.1, "0.1"), (0.09, "9e-2"), (0.025, "2.5e-2")),
)
def test_learning_rate_display_uses_scientific_notation_below_point_one(
    value, expected
):
    assert format_learning_rate(value) == expected


def test_new_playground_defaults_to_no_control_times(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    plt.imsave(source, np.zeros((4, 5)), cmap="gray", vmin=0, vmax=1)
    plt.imsave(target, np.ones((4, 5)), cmap="gray", vmin=0, vmax=1)

    app = launch_playground([str(source), str(target), "--device", "cpu", "--no-show"])

    assert app.parameters.control_times == ()
    assert app.parameters.control_nodes == ()
    assert app.make_setup().variables.n_controls == 0
    plt.close(app.fig)


def test_observation_menu_disambiguates_duplicate_image_names():
    figure, axis = plt.subplots()
    editor = ObservationTimeEditor(
        axis,
        on_select=lambda _index: None,
        on_place=lambda _index, _time: None,
        on_unplace=lambda _index: None,
    )
    editor.set_state(
        4,
        (
            "/images/im2Dbank/reg_test_01.png",
            "/images/im2Dbank_low/reg_test_01.png",
            "/images/im2Dbank/reg_test_02.png",
        ),
        (0.0, 0.5, 1.0),
        0,
    )

    labels = {text.get_text() for text in axis.texts}
    assert "[S] reg_test_01.png (im2Dbank)" in labels
    assert "[x] reg_test_01.png (im2Dbank_low)" in labels
    assert "[x] reg_test_02.png" in labels
    plt.close(figure)


def test_raster_io_preserves_visual_orientation_with_lower_origin_tensors(tmp_path):
    raster = np.zeros((4, 5), dtype=np.float32)
    raster[:2] = 1
    path = tmp_path / "top_half.png"
    plt.imsave(path, raster, cmap="gray", vmin=0, vmax=1)

    image, _ = load_image(path)
    assert float(image[0, 0, :2].max()) < 0.01
    assert float(image[0, 0, 2:].min()) > 0.99

    destination = save_timed_image_directory(
        TimedImageBatch(image, 1 - image, (1.0,)),
        tmp_path / "timed_images",
    )
    saved = plt.imread(destination / "source.png")
    assert float(saved[:2, ..., :3].min()) > 0.99
    assert float(saved[2:, ..., :3].max()) < 0.01
    restored = load_timed_image_directory(destination)
    torch.testing.assert_close(restored.source, image.float())


def test_run_uses_initial_acceleration_and_aligns_interval_fields_to_nodes():
    torch.manual_seed(12)
    source = torch.rand(1, 1, 8, 9)
    parameters = make_parameters(
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        cg_tolerance=1e-7,
        steps=2,
    )
    setup = zero_setup(source, source, parameters)
    setup.variables.initial_acceleration.normal_(std=1e-3)
    setup.variables.initial_momentum.normal_(std=1e-3)
    setup.variables.initial_jerk.normal_(std=1e-3)

    progress = []
    trajectory = run_spline(
        setup,
        device="cpu",
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    kernel = SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3)
    expected_force = CometricOperator(source, 0.25, kernel).inverse(
        setup.variables.initial_acceleration,
        eps=parameters.cg_tolerance,
    )

    torch.testing.assert_close(
        trajectory.acceleration[0],
        setup.variables.initial_acceleration[0],
    )
    torch.testing.assert_close(
        trajectory.force[0],
        expected_force[0],
        atol=2e-6,
        rtol=2e-4,
    )
    assert trajectory.images.shape == (3, 1, 8, 9)
    assert trajectory.deformed_source.shape == (3, 1, 8, 9)
    assert trajectory.photometric_only.shape == (3, 1, 8, 9)
    assert trajectory.momentum.shape == (3, 1, 8, 9)
    assert trajectory.force.shape == (3, 1, 8, 9)
    assert trajectory.acceleration.shape == (3, 1, 8, 9)
    assert trajectory.jerk.shape == (3, 1, 8, 9)
    assert trajectory.velocity.shape == (3, 2, 8, 9)
    assert trajectory.vector_momentum.shape == (3, 2, 8, 9)
    assert trajectory.target_mse.shape == (1, 3)
    for tensor in (
        trajectory.images,
        trajectory.deformed_source,
        trajectory.photometric_only,
        trajectory.momentum,
        trajectory.force,
        trajectory.acceleration,
        trajectory.jerk,
        trajectory.velocity,
        trajectory.vector_momentum,
        trajectory.target_mse,
    ):
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
    assert progress == [(1, 2), (2, 2)]
    torch.testing.assert_close(trajectory.deformed_source[0], source[0])
    torch.testing.assert_close(trajectory.photometric_only[0], source[0])
    torch.testing.assert_close(
        trajectory.photometric_only[1],
        source[0] + 0.5 * (1 - parameters.rho) * setup.variables.initial_momentum[0],
    )
    assert set(trajectory.field_energies) == {
        "momentum",
        "force",
        "acceleration",
        "jerk",
        "velocity",
        "vector_momentum",
    }
    assert all(values.shape == (3,) for values in trajectory.field_energies.values())
    torch.testing.assert_close(
        trajectory.field_energies["force"],
        trajectory.field_energies["acceleration"],
    )
    expected_vector_momentum = kernel.apply_operator(trajectory.velocity)
    torch.testing.assert_close(
        trajectory.vector_momentum,
        expected_vector_momentum,
    )
    expected_vector_energy = (trajectory.velocity * trajectory.vector_momentum).sum(
        dim=(1, 2, 3)
    )
    torch.testing.assert_close(
        trajectory.field_energies["vector_momentum"],
        expected_vector_energy,
    )
    torch.testing.assert_close(
        trajectory.field_energies["vector_momentum"],
        trajectory.field_energies["velocity"],
    )
    initial_cometric = CometricOperator(source, 0.25, kernel)
    for name, field, counterpart in (
        (
            "momentum",
            trajectory.momentum[0:1],
            initial_cometric(trajectory.momentum[0:1]),
        ),
        (
            "force",
            trajectory.force[0:1],
            initial_cometric(trajectory.force[0:1]),
        ),
        (
            "acceleration",
            trajectory.acceleration[0:1],
            initial_cometric.inverse(
                trajectory.acceleration[0:1], eps=parameters.cg_tolerance
            ),
        ),
        (
            "jerk",
            trajectory.jerk[0:1],
            initial_cometric(trajectory.jerk[0:1]),
        ),
    ):
        torch.testing.assert_close(
            trajectory.field_energies[name][0],
            (field * counterpart).sum(),
        )
    initial_gradient = tb.spatialGradient(
        source,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    initial_transport = -0.25 * kernel(
        setup.variables.initial_momentum * initial_gradient
    )
    identity = tb.make_regular_grid(
        source.shape[-2:],
        dx_convention="pixel",
        device=source.device,
    ).to(source)
    expected_deformed_source = tb.imgDeform(
        source,
        identity - 0.5 * tb.im2grid(initial_transport),
        dx_convention="pixel",
        clamp=False,
        boundary="periodic",
    )
    torch.testing.assert_close(
        trajectory.deformed_source[1],
        expected_deformed_source[0],
    )

    endpoint_cometric = CometricOperator(
        trajectory.images[-1][None],
        0.25,
        SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3),
    )
    torch.testing.assert_close(
        endpoint_cometric(trajectory.force[-1][None]),
        trajectory.acceleration[-1][None],
        atol=2e-5,
        rtol=2e-4,
    )
    endpoint_image = trajectory.images[-1][None]
    endpoint_gradient = tb.spatialGradient(
        endpoint_image,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    expected_velocity = -(0.25**0.5) * kernel(
        trajectory.momentum[-1][None] * endpoint_gradient
    )
    torch.testing.assert_close(trajectory.velocity[-1][None], expected_velocity)


def test_classic_run_matches_geodesic_spline_and_zeroes_spline_fields():
    torch.manual_seed(13)
    source = torch.rand(1, 1, 8, 9)
    target = torch.roll(source, shifts=1, dims=-1)
    setup = zero_setup(
        source,
        target,
        make_parameters(
            rho=0.25,
            gamma=0.3,
            steps=3,
            control_times=(),
        ),
    )
    setup.variables.initial_momentum.normal_(std=0.1)
    progress = []

    classic = run_classic(
        setup,
        device="cpu",
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    spline = run_spline(setup, device="cpu")

    for name in (
        "images",
        "deformed_source",
        "photometric_only",
        "momentum",
        "velocity",
        "vector_momentum",
        "target_mse",
    ):
        torch.testing.assert_close(
            getattr(classic, name),
            getattr(spline, name),
            atol=2e-6,
            rtol=2e-6,
        )
    for name in ("force", "acceleration", "jerk"):
        assert torch.count_nonzero(getattr(classic, name)) == 0
        assert torch.count_nonzero(classic.field_energies[name]) == 0
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_classic_run_rejects_drawn_non_momentum_fields():
    setup = zero_setup(
        torch.zeros(1, 1, 5, 6),
        parameters=make_parameters(steps=3, control_times=(1 / 3,)),
    )
    for field, message in (
        (setup.variables.initial_acceleration, "initial acceleration"),
        (setup.variables.initial_jerk, "initial jerk"),
        (setup.variables.control_jerks, "control jerk"),
    ):
        field.fill_(1)
        with pytest.raises(ValueError, match=message):
            run_classic(setup, device="cpu")
        field.zero_()


def test_gaussian_operator_runs_classic_and_is_rejected_for_splines(tmp_path):
    torch.manual_seed(14)
    source = torch.rand(1, 1, 7, 8)
    parameters = make_parameters(
        rho=0.25,
        steps=2,
        kernel="gaussian",
        sigma=1.5,
        model="classic",
    )
    setup = zero_setup(source, source, parameters)
    setup.variables.initial_momentum.normal_(std=0.05)

    trajectory = run_classic(setup, device="cpu")
    gradient = tb.spatialGradient(
        trajectory.images,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    expected_momentum = -(parameters.rho**0.5) * (trajectory.momentum * gradient)
    expected_velocity = GaussianRKHS(
        (1.5, 1.5),
        border_type="circular",
        normalized=False,
        kernel_reach=3,
    )(expected_momentum)

    torch.testing.assert_close(trajectory.vector_momentum, expected_momentum)
    torch.testing.assert_close(trajectory.velocity, expected_velocity)
    assert torch.isfinite(trajectory.images).all()
    restored = load_setup(save_setup(setup, tmp_path / "gaussian-setup.pt"))
    assert restored.parameters.kernel == "gaussian"
    assert restored.parameters.sigma == pytest.approx(1.5)
    with pytest.raises(ValueError, match="Gaussian cometric inversion"):
        run_spline(setup, device="cpu")


def test_classic_button_uses_shared_workspace_and_reports_invalid_fields():
    source = torch.zeros(1, 1, 8, 9)
    source[..., 2:6, 3:7] = 0.5
    setup = zero_setup(
        source,
        torch.roll(source, shifts=1, dims=-1),
        make_parameters(rho=0.25, gamma=0.3, steps=2),
    )
    setup.variables.initial_momentum.fill_(0.05)
    app = SplinePlayground(setup, device="cpu")

    app.run_classic()
    assert app.last_error is None
    assert app.cache is not None
    assert app.cache.images.shape == (3, 1, 8, 9)
    assert torch.count_nonzero(app.cache.force) == 0
    assert torch.count_nonzero(app.cache.acceleration) == 0
    assert torch.count_nonzero(app.cache.jerk) == 0
    assert "Classic complete" in app.status_text.get_text()
    app.set_time_index(2)
    assert app.current_ax.get_title().startswith("Current image")

    app.fields["initial_acceleration"].fill_(1)
    app.run_classic()
    assert app.cache is None
    assert isinstance(app.last_error, ValueError)
    assert "accepts only initial momentum" in app.status_text.get_text()
    plt.close(app.fig)


def test_setup_owns_canonical_finite_tensors():
    source = torch.zeros(1, 1, 5, 6)
    setup = zero_setup(source, source, make_parameters(steps=2))
    source.fill_(1)
    setup.variables.initial_momentum.fill_(2)
    assert torch.count_nonzero(setup.images.source) == 0
    assert torch.count_nonzero(setup.variables.initial_acceleration) == 0
    assert torch.count_nonzero(setup.variables.initial_jerk) == 0

    with pytest.raises(ValueError, match="source must have shape"):
        zero_setup(torch.zeros(1, 1, 2, 3, 4))
    invalid_target = torch.zeros(1, 1, 5, 6)
    invalid_target[..., 2, 3] = torch.nan
    with pytest.raises(ValueError, match="images must contain only finite"):
        zero_setup(torch.zeros(1, 1, 5, 6), invalid_target)
    with pytest.raises(ValueError, match="source must contain only finite"):
        zero_setup(torch.full((1, 1, 5, 6), torch.inf))


def test_control_right_limit_and_setup_round_trip(tmp_path):
    source = torch.zeros(1, 1, 6, 7)
    parameters = make_parameters(
        rho=0,
        steps=4,
        control_times=(0.5,),
        learning_rate=0.03,
        optimized_fields=("initial_momentum", "initial_jerk"),
        initialization="warm",
        regression=SolverSettings(0.04, 8, 7, 0.5),
    )
    setup = zero_setup(source, source, parameters)
    setup.variables.initial_jerk.fill_(1)
    setup.variables.control_jerks.fill_(3)
    setup.variables.initial_acceleration.fill_(0.5)

    path = save_setup(setup, tmp_path / "spline")
    restored = load_setup(path)
    trajectory = run_spline(restored, device="cpu")

    assert path.suffix == ".pt"
    assert restored.parameters == parameters
    assert torch.equal(
        restored.variables.initial_acceleration, setup.variables.initial_acceleration
    )
    assert torch.equal(restored.variables.control_jerks, setup.variables.control_jerks)
    assert trajectory.jerk[:, 0].mean(dim=(1, 2)).tolist() == [
        1.0,
        1.0,
        3.0,
        3.0,
        3.0,
    ]
    torch.testing.assert_close(
        trajectory.force[0], setup.variables.initial_acceleration[0]
    )
    torch.testing.assert_close(
        trajectory.deformed_source,
        source[0].expand_as(trajectory.deformed_source),
    )
    torch.testing.assert_close(
        trajectory.photometric_only,
        trajectory.images,
    )


def test_headless_editor_run_timeline_and_control_markers(tmp_path):
    source = torch.zeros(1, 1, 18, 20)
    source[..., 5:13, 6:14] = 0.6
    target = torch.roll(source, shifts=2, dims=-1)
    setup = zero_setup(
        source,
        target,
        make_parameters(rho=0, steps=4, control_times=(0.5,)),
    )
    app = SplinePlayground(setup, device="cpu")

    app.editor.on_press(
        SimpleNamespace(
            inaxes=app.source_ax,
            xdata=8.0,
            ydata=9.0,
            button=1,
            key=None,
        )
    )
    app.editor.on_motion(SimpleNamespace(inaxes=app.source_ax, xdata=13.0, ydata=9.0))
    app.editor.on_release(
        SimpleNamespace(
            inaxes=app.source_ax,
            xdata=13.0,
            ydata=9.0,
            button=1,
        )
    )
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0
    assert r"\Vert p_0\Vert_{I_0}^2" in app.source_footer.get_text()
    assert app.cache is None
    app.clear()
    assert torch.count_nonzero(app.fields["initial_momentum"]) == 0
    app.undo()
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0
    app.fields["initial_acceleration"].fill_(0.25)
    app.clear_all()
    assert all(torch.count_nonzero(field) == 0 for field in app.fields.values())
    app.undo()
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0

    app.overlay_radios["input"].set_active(3)
    assert app.editor.active_key == "control_jerk:0"
    assert r"\Vert r(0.5^+)\Vert_{I_0^*}^2" in app.source_footer.get_text()
    assert int(app.time_slider.val) == 2
    assert app.current_ax.get_title() == "Current image (run required)"
    assert "I_0" in app.target_footer.get_text()
    assert len(app._control_markers) == 2
    app._select_control_time(1)
    assert app.overlay_control_selector.selected_index == 0
    assert app.control_time_editor.selected_index == 0

    app.set_time_index(0)
    marker = next(iter(app._control_markers))
    app._on_pick(SimpleNamespace(artist=marker))
    assert int(app.time_slider.val) == 2

    app.run()
    assert app.cache is not None
    assert app.cache.images.shape == (5, 1, 18, 20)
    app.overlay_radios["current_field"].set_active(1)
    app.set_time_index(4)
    assert app.current_ax.get_title().endswith("$p$")
    assert app.current_colorbar_ax.get_visible()
    assert app.renderer.colorbars[app.current_ax].orientation == "horizontal"
    assert (
        app.renderer.colorbars[app.current_ax].mappable.get_cmap().name
        == DUAL_CMAP.name
    )
    assert r"\Vert p(t)\Vert_{I_t}^2" in app.current_footer.get_text()
    assert r"\mathrm{MSE}" in app.target_footer.get_text()
    assert "device: cpu" in app.status_text.get_text()
    assert "size: 18x20" in app.status_text.get_text()
    assert "steps:" not in app.status_text.get_text()
    assert app.time_slider.label.get_text() == "t"
    assert app.status_text.get_fontsize() >= 11
    assert all(
        footer.get_fontsize() >= 13
        for footer in (app.source_footer, app.current_footer, app.target_footer)
    )

    replacement_target = tmp_path / "replacement-target.png"
    plt.imsave(replacement_target, torch.ones(18, 20).numpy(), cmap="gray")
    cached_images = app.cache.images
    old_mse = app.cache.target_mse.clone()
    app.load_target(replacement_target)
    assert app.cache is not None
    assert app.cache.images is cached_images
    assert not torch.equal(app.cache.target_mse, old_mse)

    app.overlay_radios["target_mode"].set_active(1)
    assert app.target_image.get_cmap().name == "magma"
    assert "Absolute error" in app.target_ax.get_title()

    screenshot = tmp_path / "spline-playground.png"
    app.fig.savefig(screenshot, dpi=100)
    assert screenshot.stat().st_size > 10_000

    figure_size = tuple(app.fig.get_size_inches())
    app.load_source(PROJECT_ROOT / "examples/im2Dbank/simplex_tri_s_b.png")
    assert app.series.source.shape[-2:] == (100, 100)
    assert app.series.targets.shape[-2:] == (100, 100)
    assert all(field.shape[-2:] == (100, 100) for field in app.fields.values())
    assert tuple(app.fig.get_size_inches()) == figure_size
    assert app.source_ax.get_xlim() == pytest.approx((-0.5, 99.5))
    assert "size: 100x100" in app.status_text.get_text()
    plt.close(app.fig)

    spline_app = SplinePlayground(
        zero_setup(torch.zeros(1, 1, 18, 20), parameters=make_parameters()),
        device="cpu",
    )
    spline_app.add_images(
        (
            PROJECT_ROOT / "examples/im2Dbank_low/reg_test_01.png",
            PROJECT_ROOT / "examples/im2Dbank_low/reg_test_02.png",
        )
    )
    spline_app._place_image(2, 0.0)
    assert spline_app.series.source.shape[-2:] == (64, 64)
    assert spline_app.series.targets.shape[-2:] == (64, 64)
    assert "size: 64x64" in spline_app.status_text.get_text()
    plt.close(spline_app.fig)


def test_scalar_and_error_colorbars_precede_panel_footers():
    source = torch.zeros(1, 1, 12, 14)
    target = torch.ones_like(source)
    setup = zero_setup(
        source,
        target,
        make_parameters(rho=0, steps=2),
    )
    setup.variables.initial_momentum.copy_(
        torch.linspace(-0.1, 0.1, source.numel()).reshape_as(source)
    )
    setup.variables.initial_acceleration.fill_(0.1)
    app = SplinePlayground(setup, device="cpu")

    for cmap in (DUAL_CMAP, PRIMAL_CMAP):
        assert cmap(0.5)[3] == pytest.approx(0)
        assert cmap(0)[3] > 0.9
        assert cmap(1.0)[3] > 0.9
        assert cmap(0)[:3] != pytest.approx(cmap(1.0)[:3])
        assert cmap(0)[:3] == pytest.approx(matplotlib.colormaps["cool"](0.25)[:3])
        assert cmap(1.0)[:3] == pytest.approx(matplotlib.colormaps["cool"](0.75)[:3])

    source_colorbar = app.renderer.colorbars[app.source_ax]
    assert app.source_colorbar_ax.get_visible()
    assert app.source_footer.get_position()[1] == pytest.approx(-0.14)
    assert source_colorbar.orientation == "horizontal"
    assert source_colorbar.mappable.get_cmap().name == DUAL_CMAP.name

    app.overlay_radios["input"].set_active(1)
    source_colorbar = app.renderer.colorbars[app.source_ax]
    assert app.source_colorbar_ax.get_visible()
    assert source_colorbar.mappable.get_cmap().name == PRIMAL_CMAP.name

    app.overlay_radios["target_mode"].set_active(1)
    target_colorbar = app.renderer.colorbars[app.target_ax]
    assert app.target_colorbar_ax.get_visible()
    assert app.target_footer.get_position()[1] == pytest.approx(-0.14)
    assert target_colorbar.orientation == "horizontal"
    assert target_colorbar.mappable.get_cmap().name == "magma"
    assert target_colorbar.mappable.get_clim()[0] == 0

    app.fig.canvas.draw()
    figure_renderer = app.fig.canvas.get_renderer()
    for panel_axis, colorbar_axis, footer in (
        (app.source_ax, app.source_colorbar_ax, app.source_footer),
        (app.target_ax, app.target_colorbar_ax, app.target_footer),
    ):
        panel_bounds = panel_axis.get_window_extent(figure_renderer)
        colorbar_bounds = colorbar_axis.get_window_extent(figure_renderer)
        footer_bounds = footer.get_window_extent(figure_renderer)
        assert colorbar_bounds.y1 < panel_bounds.y0
        assert footer_bounds.y1 < colorbar_bounds.y0

    app.set_modal("view")
    assert not app.source_colorbar_ax.get_visible()
    assert not app.target_colorbar_ax.get_visible()
    app.set_modal(None)
    assert app.source_colorbar_ax.get_visible()
    assert app.target_colorbar_ax.get_visible()

    app.overlay_radios["target_mode"].set_active(0)
    assert not app.target_colorbar_ax.get_visible()
    assert app.target_footer.get_position()[1] == pytest.approx(-0.08)
    app.overlay_radios["target_mode"].set_active(2)
    assert not app.target_colorbar_ax.get_visible()
    assert app.target_footer.get_position()[1] == pytest.approx(-0.08)
    plt.close(app.fig)


def test_zero_control_selection_rho_and_new_setup_extent_are_consistent():
    setup = zero_setup(
        torch.zeros(1, 1, 10, 12),
        parameters=make_parameters(rho=0.99, steps=2),
    )
    app = SplinePlayground(setup, device="cpu")
    assert app.sliders["rho"].val == pytest.approx(0.99)
    assert app.sliders["spline_steps"].val == 2
    assert app.sliders["spline_iterations"].val == 10
    assert app.sliders["spline_learning_rate"].val == pytest.approx(0)
    assert app.sliders["spline_learning_rate"].valtext.get_text() == "1"
    assert app.parameter_menu.optimized_fields.get_status() == [True, True, True]
    assert app.radios["initialization"].value_selected == "Cold"
    assert [label.get_text() for label in app.radios["initialization"].labels] == [
        "Cold",
        "Geodesic",
    ]
    assert all(not slider.ax.get_visible() for slider in app.sliders.values())
    assert [
        label.get_text() for label in app.parameter_menu.optimized_fields.labels
    ] == [
        "Momentum",
        "Acceleration",
        "Jerk",
    ]
    assert app.sliders["spline_steps"].valmin == 1
    assert app.sliders["spline_steps"].valmax == 60
    assert app.make_setup().parameters.rho == pytest.approx(0.99)
    assert (
        app.buttons["view"].ax.get_position().y0
        > app.buttons["images"].ax.get_position().y0
    )
    assert (
        app.buttons["images"].ax.get_position().y0
        > app.buttons["files"].ax.get_position().y0
    )
    assert (
        app.buttons["files"].ax.get_position().y0
        > app.buttons["register"].ax.get_position().y0
    )
    assert (
        app.buttons["register"].ax.get_position().y0
        > app.buttons["run"].ax.get_position().y0
        > app.buttons["clear"].ax.get_position().y0
    )
    shortcuts = next(
        text for text in app.fig.texts if text.get_text().startswith("P  parameters")
    )
    assert app.buttons["view"].label.get_text() == "VIEW MENU  [V]"
    assert "V  view menu" in shortcuts.get_text()
    assert shortcuts.get_position() == pytest.approx((0.012, 0.975))
    assert shortcuts.get_ha() == "left"
    assert shortcuts.get_va() == "top"
    assert shortcuts.get_text().count("\n") == 9
    assert not any("A_I^{-1}" in text.get_text() for text in app.fig.texts)

    app._on_key_press(SimpleNamespace(key="p"))
    assert app.active_modal == "parameters"
    assert app.parameter_menu.backdrop_ax.get_visible()
    assert all(slider.active for slider in app.sliders.values())
    assert app.radios["initialization"].active
    app.radios["initialization"].set_active(1)
    assert app.parameters.initialization == "warm"
    assert all(
        app.sliders[name].ax.get_visible() and app.sliders[name].active
        for name in (
            "regression_cost",
            "regression_steps",
            "regression_iterations",
            "regression_learning_rate",
        )
    )
    app.sliders["regression_cost"].set_val(np.log10(0.03))
    app.sliders["regression_steps"].set_val(5)
    app.sliders["regression_iterations"].set_val(4)
    app.sliders["regression_learning_rate"].set_val(np.log10(0.09))
    assert app.parameters.regression.cost == pytest.approx(0.03)
    assert app.parameters.regression.steps == 5
    assert app.parameters.regression.iterations == 4
    assert app.parameters.regression.learning_rate == pytest.approx(0.09)
    assert app.sliders["regression_learning_rate"].valtext.get_text() == "9e-2"
    app.radios["initialization"].set_active(0)
    assert app.parameters.initialization == "cold"
    assert all(slider.active for slider in app.sliders.values())
    app.sliders["spline_iterations"].set_val(3)
    assert app.parameters.spline.iterations == 3
    app.sliders["spline_learning_rate"].set_val(np.log10(0.025))
    assert app.parameters.spline.learning_rate == pytest.approx(0.025)
    assert app.sliders["spline_learning_rate"].valtext.get_text() == "2.5e-2"
    app.parameter_menu.optimized_fields.set_active(1)
    assert app.parameters.optimized_fields == (
        "initial_momentum",
        "initial_jerk",
    )
    assert [label.get_text() for label in app.radios["kernel"].labels] == [
        "Sobolev",
        "Gaussian",
    ]
    app.radios["kernel"].set_active(1)
    assert app.parameters.kernel == "gaussian"
    assert all(slider.active for slider in app.sliders.values())
    app.sliders["sigma"].set_val(2.5)
    assert app.parameters.sigma == pytest.approx(2.5)
    app.radios["kernel"].set_active(0)
    assert app.parameters.kernel == "sobolev"
    assert app.radios["device"].active
    assert app.radios["device"].value_selected == "CPU"
    assert (
        app.radios["initialization"].ax.get_position().x1
        < app.radios["device"].ax.get_position().x0
    )
    solver_labels = {text.get_text() for text in app.parameter_menu.panels["solver"].texts}
    assert {"GEODESIC INITIALIZATION", "INITIALIZATION", "COMPUTE DEVICE"} <= solver_labels
    assert not any(button.active for button in app.file_menu.buttons)
    assert not app.source_ax.get_visible()
    app.fig.canvas.grab_mouse(app.sliders["rho"].ax)
    app.sliders["rho"].set_val(0.5)
    assert app.parameters.rho == pytest.approx(0.99)
    app.fig.canvas.release_mouse(app.sliders["rho"].ax)
    app._on_button_release(None)
    assert app.parameters.rho == pytest.approx(0.5)
    assert app._workspace_dirty
    app.sliders["spline_steps"].set_val(4)
    assert app.parameters.spline.steps == 4
    assert app.time_slider.valmax == 4
    app._on_key_press(SimpleNamespace(key="p"))
    assert not app.active_modal == "parameters"
    assert not app.radios["device"].active
    assert not app.radios["initialization"].active
    assert app.source_ax.get_visible()
    assert not app._workspace_dirty

    app.overlay_radios["input"].set_active(1)
    assert app.editor.active_key == "initial_acceleration"
    assert app.overlay_radios["input"].value_selected == r"Acceleration  $a_0$"
    assert "initial acceleration" in app.source_ax.get_title()
    assert r"\Vert a_0\Vert_{I_0}^2" in app.source_footer.get_text()
    assert FIELD_CLASS["momentum"] == "dual"
    assert FIELD_CLASS["acceleration"] == "primal"
    app.overlay_radios["input"].set_active(3)
    assert app.input_kind == "initial_jerk"
    assert app.editor.active_key == "initial_jerk"
    assert (
        app.overlay_radios["input"].value_selected
        == app.overlay_radios["input"].labels[2].get_text()
    )
    app.set_modal("view")
    assert not app.overlay_control_selector.axis.get_visible()
    app.set_modal(None)

    replacement = zero_setup(
        torch.zeros(1, 1, 6, 9),
        parameters=make_parameters(
            steps=4,
            control_times=(0.25, 0.5),
            learning_rate=0.04,
            optimized_fields=("initial_acceleration",),
            initialization="warm",
            regression=SolverSettings(0.02, 6, 5, 0.3),
        ),
    )
    app.apply_setup(replacement)
    expected_extent = [-0.5, 8.5, -0.5, 5.5]
    for image in (app.source_image, app.current_image, app.target_image):
        assert list(image.get_extent()) == expected_extent
    assert app.time_slider.valmax == 4
    assert app.sliders["spline_steps"].val == 4
    assert app.sliders["spline_steps"].valmin == 1
    assert app.sliders["spline_steps"].valmax == 60
    assert app.sliders["spline_learning_rate"].val == pytest.approx(np.log10(0.04))
    assert app.sliders["spline_learning_rate"].valtext.get_text() == "4e-2"
    assert app.parameter_menu.optimized_fields.get_status() == [False, True, False]
    assert app.radios["initialization"].value_selected == "Geodesic"
    assert app.parameters.initialization == "warm"
    assert app.sliders["alpha"].valmin == 0
    assert app.sliders["beta"].valmin == 0
    assert app.sliders["rho"].valmin == 0
    assert app.sliders["regression_cost"].val == pytest.approx(np.log10(0.02))
    assert app.sliders["regression_steps"].val == 6
    assert app.sliders["regression_iterations"].val == 5
    assert app.sliders["regression_learning_rate"].val == pytest.approx(np.log10(0.3))
    assert len(app._control_markers) == 4
    controls_heading = next(
        text for text in app.fig.texts if text.get_text() == "CONTROLS"
    )
    assert controls_heading.get_ha() == "center"
    operator_texts = [
        text.get_text() for text in app.parameter_menu.panels["model"].texts
    ]
    assert any("L v=" in text and "K=L" in text for text in operator_texts)
    assert any("Classic only" in text for text in operator_texts)
    assert [button.label.get_text() for button in app.file_menu.buttons] == [
        "LOAD FIELD",
        "LOAD PROJECT",
        "SAVE FIELD",
        "SAVE PROJECT",
        "SAVE VIDEO",
    ]

    app.overlay_radios["input"].set_active(3)
    app.set_modal("view")
    assert type(app.overlay_control_selector) is type(app.control_time_editor)
    assert not app.overlay_control_selector.editable
    assert app.overlay_control_selector.axis.get_visible()
    app.fig.canvas.draw()
    axis = app.overlay_control_selector.axis
    x, y = axis.transData.transform((0.5, 0.55))
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, app.fig.canvas, x, y, button=1)
        app.fig.canvas.callbacks.process(event_name, event)
    assert app.control_index == 1
    assert app.editor.active_key == "control_jerk:1"
    assert app.overlay_control_selector.selected_index == 1
    assert app.control_time_editor.selected_index == 1
    assert int(app.time_slider.val) == 2
    plt.close(app.fig)


def test_only_operator_choice_changes_the_operator():
    setup = zero_setup(
        torch.zeros(1, 1, 10, 12),
        parameters=make_parameters(steps=2),
    )
    app = SplinePlayground(setup, device="cpu")
    app.set_modal("parameters")
    app.fig.canvas.draw()
    for slider in (app.sliders["alpha"], app.sliders["beta"]):
        assert slider.valmin == 0
        assert slider.ax.get_xlim()[0] == 0

    def click(axis, position=(0.5, 0.5)) -> None:
        x, y = axis.transAxes.transform(position)
        for event_name in ("button_press_event", "button_release_event"):
            event = MouseEvent(event_name, app.fig.canvas, x, y, button=1)
            app.fig.canvas.callbacks.process(event_name, event)

    original_sigma = app.sliders["sigma"].val
    click(app.sliders["sigma"].ax)
    assert app.radios["kernel"].value_selected == "Sobolev"
    assert app.parameters.kernel == "sobolev"
    assert app.sliders["sigma"].val != original_sigma
    assert all(slider.active for slider in app.sliders.values())

    app.radios["kernel"].set_active(1)
    assert app.parameters.kernel == "gaussian"
    assert all(slider.active for slider in app.sliders.values())

    original_alpha = app.sliders["alpha"].val
    click(app.sliders["alpha"].ax)
    assert app.radios["kernel"].value_selected == "Gaussian"
    assert app.parameters.kernel == "gaussian"
    assert app.sliders["alpha"].val != original_alpha

    app.radios["kernel"].set_active(0)
    assert app.parameters.kernel == "sobolev"
    assert all(slider.active for slider in app.sliders.values())
    for slider in (
        app.sliders["alpha"],
        app.sliders["beta"],
        app.sliders["gamma"],
        app.sliders["sigma"],
    ):
        assert slider.track.get_alpha() in (None, 1)
    plt.close(app.fig)


def test_warm_initialization_uses_minimum_mesh_for_controls_and_observations():
    source = torch.zeros(1, 1, 5, 6)
    with pytest.raises(ValueError, match="regression mesh"):
        zero_setup(
            source,
            torch.ones_like(source),
            make_parameters(
                steps=4,
                regression=SolverSettings(steps=3),
                initialization="warm",
            ),
            target_times=(0.5,),
        )
    app = SplinePlayground(
        zero_setup(
            source,
            torch.cat((torch.ones_like(source), 2 * torch.ones_like(source))),
            make_parameters(
                steps=16,
                control_times=(0.25,),
                regression=SolverSettings(steps=3),
                initialization="cold",
            ),
            target_times=(0.375, 1.0),
        ),
        device="cpu",
    )
    app.set_modal("parameters")

    app.radios["initialization"].set_active(1)

    assert app.parameters.initialization == "warm"
    assert app.radios["initialization"].value_selected == "Geodesic"
    assert app.parameters.regression.steps == 8
    assert app.sliders["regression_steps"].val == 8
    assert all(
        app.sliders[name].ax.get_visible() and app.sliders[name].active
        for name in (
            "regression_cost",
            "regression_steps",
            "regression_iterations",
            "regression_learning_rate",
        )
    )

    app._move_control_time(0, 0.5)
    assert app.parameters.regression.steps == 8
    app._place_target(0, 0.25)
    assert app.series.times == [0.25, 1.0]
    assert app.parameters.regression.steps == 4
    assert app.sliders["regression_steps"].val == 4
    plt.close(app.fig)


def test_control_times_rescale_and_can_be_edited_on_the_parameter_timeline():
    source = torch.zeros(1, 1, 10, 12)
    setup = zero_setup(
        source,
        parameters=make_parameters(steps=16, control_times=(0.5,)),
    )
    setup.variables.control_jerks[0].fill_(3)
    app = SplinePlayground(setup, device="cpu")

    app.sliders["spline_steps"].set_val(60)
    assert app.parameters.control_times == (0.5,)
    assert app.parameters.control_nodes == (30,)
    assert torch.all(app.fields["control_jerk:0"] == 3)
    assert 30 in app._control_markers.values()
    app.sliders["spline_steps"].set_val(16)
    assert app.parameters.control_nodes == (8,)

    app._remove_control_time(0)
    assert app.parameters.control_times == ()
    app.set_modal("parameters")
    app.fig.canvas.draw()
    axis = app.control_time_editor.axis
    assert not axis.get_xticks().size
    endpoint_labels = {
        text.get_text(): text for text in axis.texts if text.get_text() in ("0", "1")
    }
    assert endpoint_labels["0"].get_position() == pytest.approx((0, 0.40))
    assert endpoint_labels["1"].get_position() == pytest.approx((1, 0.40))
    assert 0.55 - endpoint_labels["0"].get_position()[1] < 0.2

    def dispatch(name, xdata, button=1):
        x, y = axis.transData.transform((xdata, 0.55))
        event = MouseEvent(name, app.fig.canvas, x, y, button=button)
        app.fig.canvas.callbacks.process(name, event)

    dispatch("button_press_event", 0.5)
    dispatch("button_release_event", 0.5)
    assert app.parameters.control_times == (0.5,)
    assert app.parameters.control_nodes == (8,)
    app.fields["control_jerk:0"].fill_(7)

    dispatch("button_press_event", 0.5)
    dispatch("motion_notify_event", 0.75)
    dispatch("button_release_event", 0.75)
    assert app.parameters.control_times == (0.75,)
    assert app.parameters.control_nodes == (12,)
    assert torch.all(app.fields["control_jerk:0"] == 7)

    dispatch("button_press_event", 0.75, button=3)
    assert app.parameters.control_times == ()
    assert not any(key.startswith("control_jerk:") for key in app.fields)
    plt.close(app.fig)


def test_parameter_device_selector_defaults_to_cuda_and_can_choose_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    setup = zero_setup(
        torch.zeros(1, 1, 6, 8),
        parameters=make_parameters(steps=2),
    )
    app = SplinePlayground(setup)

    assert app.device == "cuda"
    assert [label.get_text() for label in app.radios["device"].labels] == [
        "CUDA",
        "CPU",
    ]
    assert app.radios["device"].value_selected == "CUDA"
    app.set_modal("parameters")
    app.radios["device"].set_active(1)
    assert app.device == "cpu"
    assert app.radios["device"].value_selected == "CPU"
    assert "device: cpu" in app.status_text.get_text()
    plt.close(app.fig)


def test_cli_step_override_preserves_control_time_and_field():
    setup = zero_setup(
        torch.zeros(1, 1, 5, 6),
        parameters=make_parameters(steps=16, control_times=(0.5,)),
    )
    setup.variables.control_jerks[0].fill_(4)
    args = SimpleNamespace(
        alpha=None,
        beta=None,
        gamma=None,
        rho=None,
        cg_tolerance=None,
        kernel="gaussian",
        sigma=2.5,
        model=None,
        steps=40,
        cost_cst=None,
        iterations=None,
        lbfgs_lr=0.025,
        control_steps=None,
    )
    parameters = _parameter_overrides(args, setup.parameters)
    replaced = _replace_parameters(setup, parameters)
    assert parameters.control_times == (0.5,)
    assert parameters.control_nodes == (20,)
    assert parameters.kernel == "gaussian"
    assert parameters.sigma == pytest.approx(2.5)
    assert parameters.spline.learning_rate == pytest.approx(0.025)
    torch.testing.assert_close(
        replaced.variables.control_jerks, setup.variables.control_jerks
    )


def test_overlay_menu_and_current_image_modes():
    source = torch.zeros(1, 1, 12, 14)
    source[..., 3:9, 4:10] = 0.5
    setup = zero_setup(
        source,
        source,
        make_parameters(rho=0.25, steps=3, control_times=(1 / 3,)),
    )
    setup.variables.initial_momentum.fill_(0.1)
    app = SplinePlayground(setup, device="cpu")

    assert app.overlay_radios["input"].labels[3].get_text().startswith("Control jerk")
    image_labels = [
        label.get_text() for label in app.overlay_radios["image_mode"].labels
    ]
    assert image_labels[1] == r"Deformation only  $I_D$"
    field_labels = [
        label.get_text() for label in app.overlay_radios["current_field"].labels
    ]
    assert r"$u=A_I^{-1}a$" in field_labels[2]
    assert r"$a=A_Iu$" in field_labels[3]
    assert r"$v=Km$" in field_labels[5]
    assert not app.active_modal == "view"
    assert not app.overlay_menu.backdrop_ax.get_visible()
    assert not app.overlay_control_selector.axis.get_visible()
    for radio in (app.overlay_radios["input"], app.overlay_radios["current_field"]):
        assert not radio._buttons.get_visible()
    app._show_run_progress(1, 2)
    assert "1/2 (50%)" in app.status_text.get_text()
    app._show_run_progress(199, 200)
    assert "199/200 (99%)" in app.status_text.get_text()
    app.fig.canvas.grab_mouse(app.sliders["rho"].ax)
    app.run()
    assert app.cache is None
    app.fig.canvas.release_mouse(app.sliders["rho"].ax)
    app.run()
    assert app.cache is not None
    assert app.time_slider.active
    assert app.editor.enabled

    app.fig.canvas.grab_mouse(app.sliders["rho"].ax)
    app.set_modal("view")
    assert not app.active_modal == "view"
    app.fig.canvas.release_mouse(app.sliders["rho"].ax)

    app._on_key_press(SimpleNamespace(key="v"))
    assert app.active_modal == "view"
    assert app.overlay_menu.backdrop_ax.get_visible()
    assert all(axis.get_visible() for axis in app.overlay_menu.panels.values())
    assert not app.overlay_control_selector.axis.get_visible()
    assert not app.overlay_menu.labels["control_time"].get_visible()
    assert not app.source_ax.get_visible()
    app.overlay_radios["input"].set_active(3)
    assert app.overlay_control_selector.axis.get_visible()
    assert app.overlay_menu.labels["control_time"].get_visible()
    app.overlay_radios["input"].set_active(0)
    assert not app.overlay_control_selector.axis.get_visible()
    assert not app.overlay_menu.labels["control_time"].get_visible()
    assert not app.time_slider.active
    assert not any(button.active for button in app.file_menu.buttons)
    assert not any(slider.active for slider in app.sliders.values())
    old_time = app.time_slider.val
    app._running = True
    marker = next(iter(app._control_markers))
    app._on_pick(SimpleNamespace(artist=marker))
    assert app.time_slider.val == old_time
    app._running = False
    app._on_key_press(SimpleNamespace(key="right"))
    assert app.time_slider.val == old_time

    file_actions = []
    app._run_modal_action = lambda action: file_actions.append(action)
    point = app.overlay_radios["image_mode"]._buttons.get_offsets()[1]
    x, y = app.overlay_radios["image_mode"].ax.transAxes.transform(point)
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, app.fig.canvas, x, y, button=1)
        app.fig.canvas.callbacks.process(event_name, event)
    assert app.current_image_mode == "deformation"
    assert file_actions == []

    app.overlay_radios["current_field"].set_active(0)
    app.overlay_radios["target_mode"].set_active(1)
    app._on_key_press(SimpleNamespace(key="escape"))
    assert not app.active_modal == "view"
    assert app.time_slider.active
    assert app.source_ax.get_visible()
    assert app.current_image_mode == "deformation"
    assert app.current_field is None
    assert app.target_mode == "Absolute error"
    assert app.current_ax.get_title().startswith("Deformation only")
    assert "$I_{D," in app.current_ax.get_title()
    assert r"I_D(t)" in app.target_ax.get_title()
    assert app.source_ax.title.get_color() == INK_COLOR
    assert app.current_ax.title.get_color() == INK_COLOR

    app.overlay_radios["image_mode"].set_active(2)
    assert app.current_image_mode == "photometric"
    assert app.current_image.get_cmap().name == "gray"
    torch.testing.assert_close(
        app.renderer.current_image_tensor(
            app.series.source,
            app.cache,
            app.current_image_mode,
            app._time_index(),
        ),
        app.cache.photometric_only[app._time_index()],
    )
    assert "Photometric only" in app.current_ax.get_title()
    assert r"|I_{\mathrm{phot}}(t)-I_{1}|" in (app.target_ax.get_title())
    assert r"I_{\mathrm{phot}}(t)" in app.target_footer.get_text()

    app.overlay_radios["current_field"].set_active(6)
    assert app.current_field == "vector_momentum"
    assert r"\Vert m(t)\Vert_{V^*}^2" in app.current_footer.get_text()
    app.sliders["spacing"].set_val(9)
    assert app.renderer.vector_spacing == 9
    quivers = [
        artist
        for artist in app._dynamic_artists[app.current_ax]
        if artist.__class__.__name__ == "Quiver"
    ]
    assert len(quivers) == 1
    assert not app.current_colorbar_ax.get_visible()
    assert quivers[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(DUAL_COLOR)
    )
    displayed = app.cache.vector_momentum[app._time_index()]
    values, x, y, factor = prepare_vector_display(displayed, spacing=9)
    assert quivers[0].X.tolist() == x.tolist()
    assert quivers[0].Y.tolist() == y.tolist()
    torch.testing.assert_close(
        torch.as_tensor(quivers[0].U),
        values[0, 0, y, x] * factor,
        check_dtype=False,
    )
    torch.testing.assert_close(
        torch.as_tensor(quivers[0].V),
        values[0, 1, y, x] * factor,
        check_dtype=False,
    )
    index = app._time_index()
    title = rf"Photometric only $I_{{\mathrm{{phot}},{index}}}$ + $m$"
    assert app.current_ax.get_title() == scaled_field_title(title, factor)

    app._on_key_press(SimpleNamespace(key="l"))
    assert app.active_modal == "files"
    assert app.file_menu.backdrop_ax.get_visible()
    app._on_key_press(SimpleNamespace(key="l"))
    assert not app.active_modal == "files"
    plt.close(app.fig)


@pytest.mark.parametrize(
    ("key", "model", "modal"),
    (
        ("p", "splines", "parameters"),
        ("v", "splines", "view"),
        ("i", "classic", "images"),
        ("i", "splines", "observations"),
        ("l", "splines", "files"),
    ),
)
def test_menu_shortcuts_preserve_image_panel_geometry(key, model, modal):
    source = torch.zeros(1, 1, 64, 96)
    app = SplinePlayground(
        zero_setup(source, parameters=make_parameters(steps=2, model=model)),
        device="cpu",
    )
    app.fig.canvas.draw()
    x, y = app.current_ax.transAxes.transform((0.5, 0.5))

    def geometry(axis):
        return (
            axis.get_position().bounds,
            axis.get_xlim(),
            axis.get_ylim(),
            axis.get_aspect(),
            axis.get_adjustable(),
            axis.get_box_aspect(),
            axis.get_xscale(),
            axis.get_yscale(),
            axis.get_autoscalex_on(),
            axis.get_autoscaley_on(),
        )

    expected = tuple(geometry(axis) for axis in app.axes)
    for _ in range(3):
        app.fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", app.fig.canvas, key, x=x, y=y),
        )
        assert app.active_modal == modal
        app.fig.canvas.draw()
        app.fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", app.fig.canvas, key, x=x, y=y),
        )
        assert app.active_modal is None
        app.fig.canvas.draw()
        for actual, wanted in zip(map(geometry, app.axes), expected):
            assert actual[0] == pytest.approx(wanted[0])
            assert actual[1] == pytest.approx(wanted[1])
            assert actual[2] == pytest.approx(wanted[2])
            assert actual[3:] == wanted[3:]
    plt.close(app.fig)


def test_input_and_current_image_switches_are_independent():
    source = torch.zeros(1, 1, 10, 12)
    source[..., 2:8, 3:9] = 0.5
    setup = zero_setup(
        source,
        torch.roll(source, shifts=1, dims=-1),
        make_parameters(rho=0.5, steps=2),
    )
    setup.variables.initial_momentum.fill_(0.05)
    app = SplinePlayground(setup, device="cpu")
    app.run_classic()
    assert app.cache is not None

    choices = (app.current_image_mode, app.current_field, app.target_mode)
    app.set_modal("view")
    app.overlay_checks["input_image"].set_active(0)
    app.overlay_checks["current_image"].set_active(0)
    assert not app.show_input_image
    assert not app.show_current_image
    app.set_modal(None)

    assert not np.asarray(app.source_image.get_array()).any()
    assert not np.asarray(app.current_image.get_array()).any()
    assert np.asarray(app.target_image.get_array()).any()
    assert not app.source_ax.get_title().startswith("Source +")
    assert not app.current_ax.get_title().startswith("Current image")
    assert (app.current_image_mode, app.current_field, app.target_mode) == choices
    assert app._dynamic_artists[app.source_ax]
    assert app._dynamic_artists[app.current_ax]

    app.set_modal("view")
    app.overlay_checks["input_image"].set_active(0)
    app.set_modal(None)
    assert app.show_input_image
    assert not app.show_current_image
    assert np.asarray(app.source_image.get_array()).any()
    assert not np.asarray(app.current_image.get_array()).any()

    app.set_modal("view")
    app.overlay_checks["current_image"].set_active(0)
    app.set_modal(None)
    assert app.show_current_image
    assert np.asarray(app.current_image.get_array()).any()
    assert (app.current_image_mode, app.current_field, app.target_mode) == choices
    plt.close(app.fig)


def test_drawing_amplitude_is_remembered_per_editable_field():
    setup = zero_setup(
        torch.zeros(1, 1, 10, 12),
        parameters=make_parameters(steps=5, control_times=(0.2, 0.6)),
    )
    app = SplinePlayground(setup, device="cpu")

    assert app.sliders["amplitude"].val == pytest.approx(0.5)
    assert "[x0.5]" in app.source_ax.get_title()
    app.sliders["amplitude"].set_val(0.75)
    assert "[x0.75]" in app.source_ax.get_title()

    app.overlay_radios["input"].set_active(2)
    assert app.editor.active_key == "initial_jerk"
    assert app.sliders["amplitude"].val == pytest.approx(0.5)
    app.sliders["amplitude"].set_val(3.0)
    app.editor.on_press(
        SimpleNamespace(
            inaxes=app.source_ax,
            xdata=4.0,
            ydata=5.0,
            button=1,
            key=None,
        )
    )
    assert app.editor.stroke is not None
    assert app.editor.stroke.amplitude == pytest.approx(3.0)
    app.editor.cancel()
    assert "[x3]" in app.source_ax.get_title()

    app.overlay_radios["input"].set_active(0)
    assert app.sliders["amplitude"].val == pytest.approx(0.75)
    assert "[x0.75]" in app.source_ax.get_title()

    app.overlay_radios["input"].set_active(3)
    app.sliders["amplitude"].set_val(2.0)
    app._select_control_time(1)
    assert app.sliders["amplitude"].val == pytest.approx(0.5)
    app.sliders["amplitude"].set_val(3.0)
    app._select_control_time(0)
    assert app.sliders["amplitude"].val == pytest.approx(2.0)

    app._add_control_time(0.5)
    assert app.sliders["amplitude"].val == pytest.approx(0.5)
    app._select_control_time(2)
    assert app.sliders["amplitude"].val == pytest.approx(3.0)
    app._select_control_time(0)
    assert app.sliders["amplitude"].val == pytest.approx(2.0)
    plt.close(app.fig)


def test_timed_targets_setup_round_trip(tmp_path):
    source = torch.zeros(1, 1, 5, 6)
    targets = torch.cat((torch.full_like(source, 0.25), torch.ones_like(source)))
    setup = zero_setup(
        source,
        targets,
        make_parameters(steps=4),
        target_times=(0.5, 1.0),
        target_paths=("half.png", "final.png"),
    )
    restored = load_setup(save_setup(setup, tmp_path / "timed_setup.pt"))
    assert restored.images.target.shape == (2, 1, 5, 6)
    assert restored.images.target_times == (0.5, 1.0)
    assert restored.target_steps == (2, 4)
    assert restored.images.target_paths == ("half.png", "final.png")


def test_selected_field_and_setup_only_project_round_trip(tmp_path):
    source = torch.zeros(1, 1, 5, 6)
    setup = zero_setup(
        source,
        torch.ones_like(source),
        make_parameters(steps=2),
    )
    app = SplinePlayground(setup, device="cpu")
    app.overlay_radios["input"].set_active(1)
    app.fields["initial_acceleration"].fill_(0.75)

    field_path = app.save_field(tmp_path / "acceleration")
    assert field_path.name == "acceleration.pt"
    loaded_field = load_field_file(field_path)
    assert loaded_field.kind == "a"
    assert loaded_field.metadata["field_role"] == "initial_acceleration"
    app.fields["initial_acceleration"].zero_()
    app.load_field(field_path)
    torch.testing.assert_close(
        app.fields["initial_acceleration"],
        torch.full_like(source, 0.75),
    )

    destination = app.save_project(tmp_path / "setup_project")
    assert (destination / "spline_setup.pt").is_file()
    assert not (destination / "trajectory.pt").exists()
    assert not (destination / "optimization.pt").exists()

    app.fields["initial_acceleration"].zero_()
    app.load_project(destination / "images.csv")
    assert app.cache is None
    assert app.last_registration is None
    torch.testing.assert_close(
        app.fields["initial_acceleration"],
        torch.full_like(source, 0.75),
    )
    assert "(setup)" in app.status_text.get_text()

    torch.save({}, destination / "optimization.pt")
    with pytest.raises(ValueError, match="RegistrationResult"):
        load_project(destination)
    plt.close(app.fig)


def test_model_actions_timed_target_selection_and_manual_placement(tmp_path):
    source = torch.zeros(1, 1, 6, 7)
    targets = torch.cat((torch.full_like(source, 0.25), torch.ones_like(source)))
    setup = zero_setup(
        source,
        targets,
        make_parameters(rho=0, steps=4, model="splines"),
        target_times=(0.5, 1.0),
        target_paths=("half.png", "final.png"),
    )
    app = SplinePlayground(setup, device="cpu")
    assert app.buttons["run"].label.get_text() == "RUN SPLINE"
    assert app.buttons["register"].label.get_text() == "REGISTER SPLINE"
    assert len(app._target_markers) == 4
    assert app.control_time_editor.image_steps == (2, 4)

    app.set_time_index(2)
    assert app.target_index == 0
    app.set_time_index(3)
    assert app.target_index == 1
    assert "Target 2/2" in app.target_ax.get_title()

    app.run_classic()
    assert len(app.series.targets) == 2
    assert app.series.times == [0.5, 1.0]

    app.radios["model"].set_active(0)
    assert app.buttons["run"].label.get_text() == "RUN CLASSIC"
    assert app.buttons["register"].label.get_text() == "REGISTER CLASSIC"
    saved_setup = load_setup(app.save(tmp_path / "classic_timed.pt"))
    assert saved_setup.images.target_times == (0.5, 1.0)
    app.radios["model"].set_active(1)

    extra_path = tmp_path / "early.png"
    plt.imsave(extra_path, np.full((6, 7), 0.5), cmap="gray", vmin=0, vmax=1)
    app.set_time_index(4)
    app.add_images((extra_path,))
    assert app.series.times[-1] is None
    assert app.series.selected == 3
    assert app.observation_menu.editor.selected == 3
    assert app.target_index == 1
    torch.testing.assert_close(app.target, targets[1:2])
    np.testing.assert_allclose(app.target_image.get_array(), targets[1, 0])
    assert "Target 2/3" in app.target_ax.get_title()
    app._place_image(3, 0.0)
    assert float(app.series.source.mean()) == pytest.approx(0.5, abs=3e-3)
    assert app.series.times[-1] is None
    app._place_image(3, 0.25)
    assert app.series.times[-1] == pytest.approx(0.25)
    assert "Placed target 1" in app.status_text.get_text()
    app.set_time_index(1)
    assert "Target 1/3" in app.target_ax.get_title()
    ordered = app.make_setup("splines")
    assert ordered.images.target_times == (0.25, 0.5, 1.0)

    app.run_spline()
    assert app.cache is not None
    destination = app.save_project(tmp_path / "saved_series")
    restored = load_timed_image_directory(destination)
    assert restored.target_times == (0.25, 0.5, 1.0)
    assert (destination / "spline_setup.pt").is_file()
    assert (destination / "trajectory.pt").is_file()
    assert not (destination / "optimization.pt").exists()

    loaded = SplinePlayground(zero_setup(source), device="cpu")
    loaded.load_project(destination)
    assert loaded.cache is not None
    assert loaded.last_registration is None
    assert loaded.parameters.model == "splines"
    assert loaded.series.times == [0.25, 0.5, 1.0]
    torch.testing.assert_close(loaded.cache.images, app.cache.images)
    assert "trajectory" in loaded.status_text.get_text()
    plt.close(loaded.fig)
    plt.close(app.fig)


def test_register_actions_load_optimized_fields_and_trajectory(tmp_path):
    source = torch.zeros(1, 1, 3, 3)
    target = torch.ones_like(source)

    classic = SplinePlayground(
        zero_setup(
            source,
            target,
            make_parameters(rho=0, steps=2, model="classic", iterations=2),
        ),
        device="cpu",
    )
    classic.register()
    assert classic.last_error is None
    assert classic.cache is not None
    assert classic.last_registration is not None
    assert classic.last_registration.trajectory is classic.cache
    assert len(classic.last_registration.loss_stock) == 2
    classic_curves = classic.last_registration.loss_curves()
    classic_losses = torch.as_tensor(classic.last_registration.loss_stock)
    torch.testing.assert_close(classic_curves["data"], classic_losses[:, 0])
    torch.testing.assert_close(
        classic_curves["regularized"],
        classic.parameters.spline.cost * classic_losses[:, 1:].sum(dim=1),
    )
    torch.testing.assert_close(
        classic_curves["full"],
        classic_curves["data"] + classic_curves["regularized"],
    )
    classic.overlay_radios["target_mode"].set_active(2)
    assert classic.target_mode == "Global loss"
    assert not classic.target_image.get_visible()
    assert not classic.target_ax.xaxis._major_tick_kw["gridOn"]
    assert classic.target_ax.get_box_aspect() == pytest.approx(1.0)
    assert classic.target_ax.get_xlim()[1] < len(classic_losses) + 1
    assert classic.target_ax.get_ylim()[1] < 2 * max(
        float(curve.max()) for curve in classic_curves.values()
    )
    assert [line.get_label() for line in classic.target_ax.lines] == [
        "Full loss",
        "Data loss",
        "Regularized momentum cost",
    ]
    assert classic.overlay_checks["target_loss"].get_status() == [True, True, True]
    classic.overlay_checks["target_loss"].set_active(1)
    assert [line.get_label() for line in classic.target_ax.lines] == [
        "Full loss",
        "Regularized momentum cost",
    ]
    assert torch.count_nonzero(classic.fields["initial_momentum"]) > 0
    assert torch.count_nonzero(classic.fields["initial_acceleration"]) == 0
    classic_destination = classic.save_project(tmp_path / "classic_project")
    loaded_classic = SplinePlayground(zero_setup(source), device="cpu")
    loaded_classic.load_project(classic_destination)
    assert loaded_classic.parameters.model == "classic"
    assert loaded_classic.cache is not None
    assert loaded_classic.last_registration is not None
    assert loaded_classic.last_registration.trajectory is loaded_classic.cache
    assert loaded_classic.last_registration.setup.parameters.model == "classic"
    plt.close(loaded_classic.fig)
    plt.close(classic.fig)

    splines = SplinePlayground(
        zero_setup(
            source,
            target,
            make_parameters(rho=0, steps=3, model="splines", iterations=2),
        ),
        device="cpu",
    )
    splines.register()
    assert splines.last_error is None
    assert splines.cache is not None
    assert splines.last_registration is not None
    assert splines.last_registration.trajectory is splines.cache
    assert all(len(loss) == 2 for loss in splines.last_registration.loss_stock.values())
    spline_curves = splines.last_registration.loss_curves()
    spline_losses = splines.last_registration.loss_stock
    torch.testing.assert_close(spline_curves["data"], spline_losses["data_loss"])
    torch.testing.assert_close(
        spline_curves["regularized"],
        splines.parameters.spline.cost * spline_losses["acceleration_energy"],
    )
    torch.testing.assert_close(spline_curves["full"], spline_losses["total_cost"])
    splines.overlay_radios["target_mode"].set_active(2)
    assert splines.target_ax.get_box_aspect() == pytest.approx(1.0)
    assert splines.target_ax.get_xlim()[1] < len(spline_losses["data_loss"]) + 1
    assert [line.get_label() for line in splines.target_ax.lines] == [
        "Full loss",
        "Data loss",
        "Regularized acceleration cost",
    ]
    assert torch.count_nonzero(splines.fields["initial_momentum"]) > 0
    assert "Optimized fields loaded" in splines.status_text.get_text()
    destination = splines.save_project(tmp_path / "optimized_project")
    assert (destination / "optimization.pt").is_file()
    loaded_splines = SplinePlayground(zero_setup(source), device="cpu")
    loaded_splines.load_project(destination)
    assert loaded_splines.cache is not None
    assert loaded_splines.last_registration is not None
    assert loaded_splines.last_registration.trajectory is loaded_splines.cache
    for name, losses in splines.last_registration.loss_stock.items():
        torch.testing.assert_close(
            loaded_splines.last_registration.loss_stock[name],
            losses,
        )
    copied = loaded_splines.save_project(tmp_path / "optimized_project_copy")
    assert (copied / "optimization.pt").is_file()
    plt.close(loaded_splines.fig)
    splines._invalidate("field changed")
    assert splines.last_registration is None
    plt.close(splines.fig)


def test_loss_plot_limits_ignore_large_image_extent():
    source = torch.zeros(1, 1, 512, 512)
    app = SplinePlayground(zero_setup(source), device="cpu")
    app.last_registration = RegistrationResult(
        setup=app.make_setup("splines"),
        trajectory=None,  # type: ignore[arg-type]
        loss_stock={
            "data_loss": torch.tensor([300.0, 200.0, 180.0]),
            "acceleration_energy": torch.tensor([0.0, 20.0, 10.0]),
            "total_cost": torch.tensor([300.0, 200.2, 180.1]),
        },
        elapsed_seconds=0.0,
    )

    app.overlay_radios["target_mode"].set_active(2)

    assert app.target_ax.get_xlim()[1] < 3
    assert app.target_ax.get_ylim()[1] < 350
    assert all(float(tick).is_integer() for tick in app.target_ax.get_xticks())
    colors = {line.get_label(): line.get_color() for line in app.target_ax.lines}
    app.set_time_index(1)
    assert {
        line.get_label(): line.get_color() for line in app.target_ax.lines
    } == colors
    app.overlay_checks["target_loss"].set_active(1)
    assert {line.get_label(): line.get_color() for line in app.target_ax.lines} == {
        "Full loss": colors["Full loss"],
        "Regularized acceleration cost": colors["Regularized acceleration cost"],
    }
    app.overlay_radios["target_mode"].set_active(0)
    assert app.target_image.get_visible()
    assert app.target_ax.get_xlim() == pytest.approx((-0.5, 511.5))
    assert app.target_ax.get_ylim() == pytest.approx((-0.5, 511.5))
    assert not app.target_ax.get_autoscalex_on()
    assert not app.target_ax.get_autoscaley_on()
    plt.close(app.fig)


def test_register_spline_materializes_final_diagnostics_without_replay(monkeypatch):
    saved_runs = 0
    retained_runs = 0
    original_forward = MetamorphosisSplineIntegrator.forward

    def counting_forward(self, *args, **kwargs):
        nonlocal saved_runs, retained_runs
        if kwargs.get("save", True):
            saved_runs += 1
        if kwargs.get("retain_diagnostics", False):
            retained_runs += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(MetamorphosisSplineIntegrator, "forward", counting_forward)
    source = torch.zeros(1, 1, 3, 3)
    result = register_spline(
        zero_setup(
            source,
            torch.ones_like(source),
            make_parameters(rho=0, steps=3, iterations=1),
        ),
        device="cpu",
    )

    assert saved_runs == 0
    assert retained_runs == 1
    assert result.trajectory.images.shape == (4, 1, 3, 3)


def test_registration_forwards_setup_lbfgs_learning_rate(monkeypatch):
    from demeter.metamorphosis.var_classes import Momenta
    from draft.playground.splines import registration as registration_module

    source = torch.zeros(1, 1, 3, 3)
    setup = zero_setup(
        source,
        torch.ones_like(source),
        make_parameters(
            rho=0,
            steps=3,
            iterations=1,
            learning_rate=0.025,
        ),
    )
    captured = {}
    trajectory = object()

    class FakeOptimizer:
        def __init__(self, **kwargs):
            self.mp = kwargs["geodesic"]
            self.loss_stock = {}
            captured["optimizer_init"] = kwargs

        def forward(self, variables_ini, **kwargs):
            captured.update(kwargs)
            self.optimized_variables = variables_ini

    monkeypatch.setattr(
        registration_module,
        "MetamorphosisSplineOptimizer",
        FakeOptimizer,
    )
    monkeypatch.setattr(
        registration_module,
        "_trajectory_from_final_spline_integration",
        lambda *_args, **_kwargs: trajectory,
    )

    result = registration_module.register_spline(setup, device="cpu")

    assert captured["grad_coef"] == pytest.approx(0.025)
    assert result.trajectory is trajectory

    classic_setup = zero_setup(
        source,
        torch.ones_like(source),
        make_parameters(
            model="classic",
            steps=3,
            iterations=1,
            learning_rate=0.4,
        ),
    )
    classic_captured = {}

    def fake_classic_optimizer(**kwargs):
        classic_captured.update(kwargs)
        return SimpleNamespace(
            optimized_momenta=Momenta(momentum_I=torch.zeros_like(source)),
            loss_stock=[],
        )

    monkeypatch.setattr(registration_module, "metamorphosis", fake_classic_optimizer)
    monkeypatch.setattr(
        registration_module,
        "run_classic",
        lambda *_args, **_kwargs: trajectory,
    )

    result = registration_module.register_classic(classic_setup, device="cpu")

    assert classic_captured["grad_coef"] == pytest.approx(0.4)
    assert result.trajectory is trajectory


def test_register_spline_uses_geodesic_regression_for_warm_start(monkeypatch):
    from demeter.metamorphosis.var_classes import Momenta
    from draft.playground.splines import registration as registration_module

    source = torch.zeros(1, 1, 3, 3)
    target = torch.cat((torch.ones_like(source), 2 * torch.ones_like(source)))
    seed = torch.full_like(source, 0.25)
    setup = zero_setup(
        source,
        target,
        make_parameters(
            alpha=0.4,
            beta=0.3,
            gamma=0.2,
            rho=0,
            steps=4,
            iterations=1,
            cost=0.05,
            learning_rate=0.025,
            optimized_fields=("initial_acceleration",),
            initialization="warm",
            regression=SolverSettings(0.07, 8, 3, 0.4),
        ),
        target_times=(0.5, 1.0),
    )
    captured = {"optimizer_calls": 0}
    trajectory = object()

    def fake_regression(**kwargs):
        captured["regression"] = kwargs
        return SimpleNamespace(optimized_momenta=Momenta(momentum_I=seed.clone()))

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["optimizer_calls"] += 1
            captured["optimizer_init"] = kwargs
            self.mp = kwargs["geodesic"]
            self.loss_stock = {}

        def forward(self, variables_ini, **kwargs):
            captured["variables"] = variables_ini
            captured["spline_forward"] = kwargs
            self.optimized_variables = variables_ini

    monkeypatch.setattr(
        registration_module, "metamorphosis_regression", fake_regression
    )
    monkeypatch.setattr(
        registration_module, "MetamorphosisSplineOptimizer", FakeOptimizer
    )
    monkeypatch.setattr(
        registration_module,
        "_trajectory_from_final_spline_integration",
        lambda *_args, **_kwargs: trajectory,
    )

    result = registration_module.register_spline(setup, device="cpu")

    regression = captured["regression"]
    assert regression["target_times"] == (0.5, 1.0)
    torch.testing.assert_close(regression["target"], setup.images.target)
    assert regression["rho"] == 0
    assert regression["integration_steps"] == 8
    assert regression["n_iter"] == 3
    assert regression["cost_cst"] == pytest.approx(0.07)
    assert regression["grad_coef"] == pytest.approx(0.4)
    assert regression["lbfgs_max_iter"] == registration_module.LBFGS_MAX_ITER
    assert regression["lbfgs_history_size"] == registration_module.LBFGS_HISTORY_SIZE
    assert regression["boundary"] == "periodic"
    assert regression["kernelOperator"].alpha == pytest.approx(0.4)
    assert regression["kernelOperator"].beta == pytest.approx(0.3)
    assert regression["kernelOperator"].gamma == pytest.approx(0.2)
    variables = captured["variables"]
    torch.testing.assert_close(variables.initial_momentum, seed)
    assert not variables.initial_momentum.requires_grad
    assert variables.initial_acceleration.requires_grad
    assert not variables.initial_jerk.requires_grad
    assert torch.count_nonzero(variables.initial_acceleration) == 0
    assert torch.count_nonzero(variables.initial_jerk) == 0
    torch.testing.assert_close(result.setup.variables.initial_momentum, seed)
    assert result.trajectory is trajectory

    no_fields = replace(
        setup,
        parameters=replace(setup.parameters, optimized_fields=()),
    )
    captured["replay_setup"] = None
    monkeypatch.setattr(
        registration_module,
        "run_spline",
        lambda replay_setup, **_kwargs: (
            captured.update(replay_setup=replay_setup) or trajectory
        ),
    )

    result = registration_module.register_spline(no_fields, device="cpu")

    assert captured["optimizer_calls"] == 1
    torch.testing.assert_close(result.setup.variables.initial_momentum, seed)
    torch.testing.assert_close(
        captured["replay_setup"].variables.initial_momentum,
        seed,
    )


def test_register_spline_optimizes_only_selected_fields_and_always_controls():
    source = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
    target = torch.ones_like(source)
    acceleration_only = zero_setup(
        source,
        target,
        make_parameters(
            rho=0,
            steps=3,
            iterations=2,
            optimized_fields=("initial_acceleration",),
        ),
    )

    result = register_spline(acceleration_only, device="cpu")

    assert torch.count_nonzero(result.setup.variables.initial_momentum) == 0
    assert torch.count_nonzero(result.setup.variables.initial_acceleration) > 0
    assert torch.count_nonzero(result.setup.variables.initial_jerk) == 0

    control_only = zero_setup(
        source,
        target,
        make_parameters(
            rho=0,
            steps=4,
            control_times=(0.25,),
            iterations=2,
            optimized_fields=(),
        ),
    )

    result = register_spline(control_only, device="cpu")

    assert torch.count_nonzero(result.setup.variables.initial_momentum) == 0
    assert torch.count_nonzero(result.setup.variables.initial_acceleration) == 0
    assert torch.count_nonzero(result.setup.variables.initial_jerk) == 0
    assert torch.count_nonzero(result.setup.variables.control_jerks) > 0

    no_fields = zero_setup(
        source,
        target,
        make_parameters(
            rho=0,
            steps=3,
            iterations=2,
            optimized_fields=(),
        ),
    )

    result = register_spline(no_fields, device="cpu")

    assert torch.count_nonzero(result.setup.variables.initial_momentum) == 0
    assert torch.count_nonzero(result.setup.variables.initial_acceleration) == 0
    assert torch.count_nonzero(result.setup.variables.initial_jerk) == 0
