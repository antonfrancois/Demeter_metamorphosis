import pytest
import torch
import math

import src.demeter.utils.torchbox as tb


def test_grid_from_matrix_2d():
    grid = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]],
          [[-1.0, 0.5], [0.0, -2.0]]]]
    )  # [B, H, W, 2]
    rot_mat = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])

    out = tb.matrix_time_grid(grid, rot_mat)

    x = grid[..., 0]
    y = grid[..., 1]
    expected = torch.stack((2.0 * x + y, -x + 3.0 * y), dim=-1)
    assert torch.allclose(out, expected)


def test_grid_from_matrix_3d():
    grid = torch.tensor(
        [[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
          [[[0.0, -1.0, 2.0], [-2.0, 1.0, 0.0]]]]]
    )  # [B, D, H, W, 3]
    rot_mat = torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0], [4.0, 0.0, 1.0]])

    out = tb.matrix_time_grid(grid, rot_mat)

    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]
    expected = torch.stack((x + 2.0 * y, -y + 3.0 * z, 4.0 * x + z), dim=-1)
    assert torch.allclose(out, expected)


def test_grid_from_matrix_casts_rot_mat_dtype_to_grid_dtype():
    grid = torch.randn(1, 3, 4, 2, dtype=torch.float32)
    rot_mat = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)

    out = tb.matrix_time_grid(grid, rot_mat)

    assert out.dtype == grid.dtype
    assert out.device == grid.device


def test_grid_from_matrix_raises_on_bad_rot_mat_shape():
    grid = torch.randn(1, 3, 4, 2)
    bad_rot_mat = torch.eye(3)

    with pytest.raises(ValueError, match=r"Expected rot_mat shape"):
        tb.matrix_time_grid(grid, bad_rot_mat)


@pytest.mark.parametrize(
    "angle,expected",
    [
        (math.pi / 4, (math.sqrt(2) / 2, math.sqrt(2) / 2)),
        (math.pi / 2, (0.0, 1.0)),
        (math.pi, (-1.0, 0.0)),
        (-math.pi / 3, (0.5, -math.sqrt(3) / 2)),
    ],
)
def test_create_rot_mat_2d_rotates_grid_with_expected_angle(angle, expected):
    # Single grid vector along +x; after rotation by theta it becomes (cos(theta), sin(theta)).
    grid = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    rot_mat = tb.create_rot_mat_2d(torch.tensor(angle, dtype=torch.float32))

    out = tb.matrix_time_grid(grid, rot_mat)
    expected_vec = torch.tensor(expected, dtype=torch.float32)

    assert torch.allclose(out[0, 0, 0], expected_vec, atol=1e-6)


@pytest.mark.parametrize(
    "axis,angle,grid_vec,expected",
    [
        ("gamma", math.pi / 4, (1.0, 0.0, 0.0), (math.sqrt(2) / 2, math.sqrt(2) / 2, 0.0)),
        ("gamma", math.pi / 2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("beta", math.pi / 4, (1.0, 0.0, 0.0), (math.sqrt(2) / 2, 0.0, -math.sqrt(2) / 2)),
        ("beta", -math.pi / 2, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("alpha", math.pi / 3, (0.0, 1.0, 0.0), (0.0, 0.5, math.sqrt(3) / 2)),
        ("alpha", -math.pi / 2, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ],
)
def test_create_rot_mat_3d_rotates_grid_with_expected_angle(axis, angle, grid_vec, expected):
    gamma = torch.tensor(0.0, dtype=torch.float32)
    beta = torch.tensor(0.0, dtype=torch.float32)
    alpha = torch.tensor(0.0, dtype=torch.float32)
    if axis == "gamma":
        gamma = torch.tensor(angle, dtype=torch.float32)
    elif axis == "beta":
        beta = torch.tensor(angle, dtype=torch.float32)
    elif axis == "alpha":
        alpha = torch.tensor(angle, dtype=torch.float32)
    else:
        raise AssertionError(f"Unknown axis '{axis}'")

    grid = torch.tensor([[[[[*grid_vec]]]]], dtype=torch.float32)
    rot_mat = tb.create_rot_mat_3d(
        (gamma, beta, alpha)
    )

    out = tb.matrix_time_grid(grid, rot_mat)
    expected_vec = torch.tensor(expected, dtype=torch.float32)

    assert torch.allclose(out[0, 0, 0, 0], expected_vec, atol=1e-6)


def _apply_affine_homogeneous_2d(grid: torch.Tensor, affine_3x3: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(grid[..., :1])
    grid_h = torch.cat((grid, ones), dim=-1)  # [B, H, W, 3]
    out_h = grid_h @ affine_3x3.T
    return out_h[..., :2]


def test_rotation_then_transpose_inverse_recovers_grid():
    grid = tb.make_regular_grid((1, 9, 11, 2), dx_convention="2square").to(torch.float32)
    angle = torch.tensor(math.pi / 7, dtype=torch.float32)
    rot = tb.create_rot_mat_2d(angle)

    rotated = tb.matrix_time_grid(grid, rot)
    recovered = tb.matrix_time_grid(rotated, rot.T)

    assert torch.allclose(recovered, grid, atol=1e-6)


def test_affine_then_matrix_inverse_recovers_grid():
    grid = tb.make_regular_grid((1, 9, 11, 2), dx_convention="2square").to(torch.float32)
    params = torch.tensor([math.pi / 6, 0.2, -0.15, 1.2, 0.8], dtype=torch.float32)
    affine = tb.create_affine_mat_2d(params)
    affine_inv = torch.linalg.inv(affine)

    transformed = _apply_affine_homogeneous_2d(grid, affine)
    recovered = _apply_affine_homogeneous_2d(transformed, affine_inv)

    assert torch.allclose(recovered, grid, atol=1e-5)
