import numpy as np
import torch
import pytest
from draft.visualiseGeo_refactor import img_torch_to_plt  # replace with actual module name

# ---------- 2D Torch Tensor Tests ----------

def test_2d_tensor_rgb():
    img = torch.randn(2, 3, 64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (2, 64, 64, 3)
    assert isinstance(out, np.ndarray)

def test_2d_tensor_gray():
    img = torch.randn(4, 1, 32, 32)
    out = img_torch_to_plt(img)
    assert out.shape == (4, 32, 32, 1)

def test_2d_tensor_invalid_channels():
    img = torch.randn(1, 5, 32, 32)
    with pytest.raises(ValueError, match="Unsupported number of channels for 2D"):
        img_torch_to_plt(img)

def test_2d_tensor_no_batch():
    img = torch.randn(64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (1, 64, 64, 1)

def test_2d_numpy_rgb():
    img = np.random.randn(64, 64, 3)
    out = img_torch_to_plt(img)
    assert out.shape == (1, 64, 64, 3)

def test_2d_numpy_invalid_channels():
    img = np.random.randn(64, 64, 5)
    with pytest.raises(ValueError):
        img_torch_to_plt(img)

# ---------- 3D Torch Tensor Tests ----------

def test_3d_tensor_rgb():
    img = torch.randn(2, 3, 10, 64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (2, 10, 64, 64, 3)

def test_3d_tensor_gray():
    img = torch.randn(2, 1, 10, 64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (2, 10, 64, 64, 1)

def test_3d_tensor_scalar():
    img = torch.randn(10, 64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (1, 10, 64, 64, 1)

def test_3d_tensor_batch_scalar():
    img = torch.randn(2, 10, 64, 64)
    out = img_torch_to_plt(img)
    assert out.shape == (2, 10, 64, 64, 1), f"expected (2, 10, 64, 64, 1) got {out.shape}"

def test_3d_tensor_batch_scalar_invalid_w3():
    img = torch.randn(2, 10, 64, 3)
    with pytest.raises(ValueError):
        img_torch_to_plt(img)

# ---------- 3D Numpy Array Tests ----------

def test_3d_numpy_rgb():
    img = np.random.randn(10, 64, 64, 3)
    out = img_torch_to_plt(img)
    assert out.shape == (1, 10, 64, 64, 3), f"expected  (1, 10, 64, 64, 3) got {out.shape}"

def test_3d_numpy_rgb_batched():
    img = np.random.randn(2, 10, 64, 64, 3)
    out = img_torch_to_plt(img)
    assert out.shape == (2, 10, 64, 64, 3)

def test_3d_numpy_invalid_channels():
    img = np.random.randn(2, 10, 64, 64, 5)
    with pytest.raises(ValueError):
        img_torch_to_plt(img)

# ---------- Invalid Input Tests ----------

def test_invalid_input_type():
    with pytest.raises(TypeError):
        img_torch_to_plt("not an array or tensor")

def test_unsupported_numpy_shape():
    img = np.random.randn(5, 5, 5, 5, 5, 5)
    with pytest.raises(ValueError):
        img_torch_to_plt(img)

def test_unsupported_tensor_shape():
    img = torch.randn(1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        img_torch_to_plt(img)