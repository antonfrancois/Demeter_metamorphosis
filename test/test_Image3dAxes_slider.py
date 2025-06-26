import os
import torch
import pytest
import matplotlib.pyplot as plt

from demeter.utils.torchbox import temporal_img_cmp
from draft.visualiseGeo_refactor import Image3dAxes_slider


# Path to your test file
DATA_PATH = "/home/turtlefox/Documents/11_metamorphoses/data/pixyl/aligned/PSL_001/"
DATA_FILE = "PSL_001_longitudinal_rigid.pt"


@pytest.fixture(scope="module")
def loaded_data():
    data = torch.load(os.path.join(DATA_PATH, DATA_FILE))
    data["flair_longitudinal"] /= data["flair_longitudinal"].max()
    data["t1ce_longitudinal"] /= data["t1ce_longitudinal"].max()
    return data


def test_slider_with_5d_tensor(loaded_data):
    flair = loaded_data["flair_longitudinal"]  # (T, 1, D, H, W)
    viewer = Image3dAxes_slider(flair)
    assert viewer.image.shape[0] == flair.shape[0]


def test_slider_with_temporal_img_cmp(loaded_data):
    flair = loaded_data["flair_longitudinal"]
    t1ce = loaded_data["t1ce_longitudinal"]
    cmp_img = temporal_img_cmp(flair, t1ce)  # (T, D, H, W)
    viewer = Image3dAxes_slider(cmp_img)
    assert viewer.image.shape[:2] == cmp_img.shape[:2]  # check (T, D)


def test_slider_with_pred_clipped(loaded_data):
    pred = torch.clip(loaded_data["pred_longitudinal"], 0, 10)
    viewer = Image3dAxes_slider(pred)
    assert viewer.image.shape[0] == pred.shape[0]


def test_slider_with_single_3d_image(loaded_data):
    flair = loaded_data["flair_longitudinal"]
    single_3d = flair[-1, 0]  # (D, H, W)
    viewer = Image3dAxes_slider(single_3d)
    assert viewer.image[0,...,0].shape == single_3d.shape


@pytest.mark.parametrize("cmap", ["tab10", "viridis" ])
def test_cmap_switching(loaded_data, cmap):
    pred = torch.clip(loaded_data["pred_longitudinal"], 0, 10)
    viewer = Image3dAxes_slider(pred)
    viewer.change_image(pred, cmap=cmap)
    assert viewer.plt_img_x.get_cmap().name == cmap


def test_go_on_slice_validates_silently(loaded_data):
    flair = loaded_data["flair_longitudinal"]
    viewer = Image3dAxes_slider(flair)
    try:
        viewer.go_on_slice(x=100, y=120, z=80)
    except Exception as e:
        pytest.fail(f"go_on_slice should not fail, but got {e}")
