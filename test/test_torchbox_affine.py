import math
import pytest

torch = pytest.importorskip("torch")

import demeter.utils.torchbox as tb


@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, - 5 *math.pi / 6])
def test_grid_from_rotation_matches_manual_rotation(theta):
    grid = tb.make_regular_grid((5, 5), dx_convention="2square")
    rot_mat = tb.create_rot_mat_2d(torch.tensor(theta))

    rotated_grid = tb.grid_from_rotation(grid, rot_mat)

    assert rotated_grid.shape == grid.shape

    sample_indices = [(0, 0), (2, 2), (4, 1), (1, 4)]
    for i, j in sample_indices:
        coord = grid[0, i, j]
        expected = torch.matmul(rot_mat, coord)
        torch.testing.assert_close(rotated_grid[0, i, j], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "theta,translation",
    [
        (0.0, (0.0, 0.0)),
        (math.pi / 6, (0.2, -0.1)),
        (-math.pi / 4, (0.5, 0.3)),
    ],
)
def test_grid_from_rotation_translation(theta, translation):
    grid = tb.make_regular_grid((4, 6), dx_convention="2square")
    rot_mat = tb.create_rot_mat_2d(torch.tensor(theta))
    translation_vec = torch.tensor(translation, dtype=grid.dtype)

    rotated_translated = tb.grid_from_rotation_translation(grid, rot_mat, translation_vec)

    sample_indices = [(0, 0), (1, 3), (3, 5)]
    for i, j in sample_indices:
        coord = grid[0, i, j] + translation_vec
        expected = torch.matmul(rot_mat, coord)
        torch.testing.assert_close(
            rotated_translated[0, i, j], expected, rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize(
    "theta,translation,scale",
    [
        (math.pi / 8, (0.1, -0.2), 0.5),
        (-math.pi / 3, (-0.25, 0.15), 1.4),
    ],
)
def test_grid_from_rotation_translation_scaling(theta, translation, scale):
    grid = tb.make_regular_grid((5, 5), dx_convention="2square")
    rot_mat = tb.create_rot_mat_2d(torch.tensor(theta))
    translation_vec = torch.tensor(translation, dtype=grid.dtype)
    scale_val = torch.tensor(scale, dtype=grid.dtype)

    transformed = tb.grid_from_rotation_translation_scaling(grid, rot_mat, translation_vec, scale_val)

    sample_indices = [(0, 0), (2, 2), (4, 4)]
    for i, j in sample_indices:
        coord = grid[0, i, j] + translation_vec
        expected = torch.matmul(rot_mat, coord) * scale_val
        torch.testing.assert_close(transformed[0, i, j], expected, rtol=1e-6, atol=1e-6)
