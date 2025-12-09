import math
import pytest
import torch

import demeter.utils.reproducing_kernels as rk


def _manual_sigma(img_shape, subdiv, c=0.1):
    return tuple(math.sqrt(-((size / s) ** 2) / 2 * math.log(c)) for size, s in zip(img_shape, subdiv))


def test_int_subdiv_with_batched_3d_shape():
    img = torch.zeros(1, 1, 60, 80, 100)
    result = rk.get_sigma_from_img_ratio(img.shape, 5)
    expected = _manual_sigma((60, 80, 100), (5, 5, 5))
    assert result == expected


def test_tuple_subdiv_matches_dimensions():
    shape = (75, 100)
    result = rk.get_sigma_from_img_ratio(shape, (3, 5), c=0.01)
    expected = _manual_sigma(shape, (3, 5), c=0.01)
    assert result == expected


def test_list_of_tuples_returns_list_of_sigma_tuples():
    shape = (50, 60, 70)
    subdiv = [(5, 6, 7), (10, 12, 14)]
    result = rk.get_sigma_from_img_ratio(shape, subdiv)
    expected = [_manual_sigma(shape, s) for s in subdiv]
    assert result == expected


def test_list_of_singletons_broadcasts_to_all_dims():
    shape = (193, 229, 193)
    subdiv = [[50], [40]]
    result = rk.get_sigma_from_img_ratio(shape, subdiv)
    expected = [
        _manual_sigma(shape, (50, 50, 50)),
        _manual_sigma(shape, (40, 40, 40)),
    ]
    assert result == expected


def test_list_of_ints_broadcasts_each_to_all_dims():
    shape = (120, 160)
    subdiv = [12, 24]
    result = rk.get_sigma_from_img_ratio(shape, subdiv)
    expected = [
        _manual_sigma(shape, (12, 12)),
        _manual_sigma(shape, (24, 24)),
    ]
    assert result == expected


def test_invalid_tuple_length_raises():
    with pytest.raises(ValueError):
        rk.get_sigma_from_img_ratio((100, 150), (5, 10, 4))


def test_invalid_list_length_raises():
    with pytest.raises(ValueError):
        rk.get_sigma_from_img_ratio((100, 150, 200), [[5, 10]])
