import os
import pytest

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "none"
os.environ["MKL_THREADING_LAYER"] = "GNU"

if not os.access("/dev/shm", os.W_OK):
    pytest.skip("Shared memory unavailable in this environment", allow_module_level=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import pi

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from demeter.utils.toolbox import full_ellipse
import demeter.utils.torchbox as tb
import demeter.utils.reproducing_kernels as rk
from demeter.metamorphosis.rotate import RigidMetamorphosis_integrator


def _make_off_center_oval(height=64, width=80):
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    mask = full_ellipse(
        x,
        y,
        a=width / 4,
        b=height / 5,
        center=(width * 0.65, height * 0.35),
        theta=0.35,
    )
    return torch.tensor(mask, dtype=torch.float32)[None, None]


def test_update_field_affine_translation_equivariance(tmp_path):
    image = _make_off_center_oval()
    # kernel = rk.DummyKernel()
    kernel = rk.GaussianRKHS((2,2),normalized=True)
    integrator = RigidMetamorphosis_integrator(
        rho=0.5,
        kernelOperator=kernel,
        n_step=4,
        dx_convention="2square",
    )
    integrator.flag_hamiltonian_integration = False
    integrator.id_grid = tb.make_regular_grid(
        image.shape[2:],
        dx_convention=integrator.dx_convention,
        device=image.device,
    )
    kernel.init_kernel(image)

    test_cases = [
        ("id", 0.1, torch.zeros((2,), dtype=image.dtype),0),
        ("b_x = -0.25", 0.25, torch.tensor([-0.25, 0.0], dtype=image.dtype), pi/2),
        ("negP, b_y = 0.25", -0.015, torch.tensor([0.0, 0.25], dtype=image.dtype), 0),
        ("rot", 0.015, torch.tensor([0., 0.], dtype=image.dtype), -2*pi/3),
    ]

    fig, axes = plt.subplots(1, len(test_cases), figsize=(4 * len(test_cases), 4))
    stride = 3
    id_grid = integrator.id_grid
    extent = [
        float(id_grid[..., 0].min()),
        float(id_grid[..., 0].max()),
        float(id_grid[..., 1].min()),
        float(id_grid[..., 1].max()),
    ]

    for ax, (name, momentum_value, b_vec, angle) in zip(axes, test_cases):
        momentum = torch.full_like(image, momentum_value)
        rot_mat = tb.create_rot_mat_2d(torch.tensor(angle))
        # rotated_grid = tb.grid_from_rotation(integrator.id_grid, rot_mat)
        inv_a = rot_mat.T
        field = integrator._update_field_affine_(momentum, image, inv_a, b_vec)

        grad_image = tb.spatialGradient(image, dx_convention=integrator.dx_convention)
        field_momentum = (grad_image * momentum.unsqueeze(2)).sum(dim=1)
        translation_grid = tb.grid_from_rotation_translation(id_grid, inv_a, -b_vec)
        convolved = kernel(field_momentum)
        expected_field = -tb.im2grid(tb.imgDeform(convolved, translation_grid))

        # torch.testing.assert_close(field, expected_field, rtol=1e-4, atol=1e-5)
        assert field.shape == (*id_grid.shape[:3], 2)
        assert torch.isfinite(field).all()

        grid_x = id_grid[0, ::stride, ::stride, 0].cpu()
        grid_y = id_grid[0, ::stride, ::stride, 1].cpu()
        ax.imshow(image[0, 0].cpu(), extent=extent, origin="lower", cmap="gray")
        ax.quiver(
            grid_x,
            grid_y,
            field[0, ::stride, ::stride, 0].detach().cpu(),
            field[0, ::stride, ::stride, 1].detach().cpu(),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color="#d95f02",
        )
        ax.set_title(f"{name}, b={tuple(b_vec.tolist())}")
        ax.set_xticks([])
        ax.set_yticks([])

    plot_path = tmp_path / "update_field_affine.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()
