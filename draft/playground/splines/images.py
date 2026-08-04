"""Image discovery and loading for the spline playground."""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F

from ..field_playground_core import coerce_image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_BANK = PROJECT_ROOT / "examples" / "im2Dbank"
DEFAULT_SOURCE = IMAGE_BANK / "reg_test_m0t.png"
DEFAULT_TARGET = IMAGE_BANK / "reg_test_m1c.png"


def resolve_image_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    for path in (
        candidate,
        IMAGE_BANK / candidate,
        IMAGE_BANK / f"reg_test_{value}.png",
    ):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"could not find image {value!r}; pass a path, im2Dbank filename, "
        "or shorthand such as '01'"
    )


def load_image(
    value: str | Path,
    size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, Path]:
    path = resolve_image_path(value)
    array = mpimg.imread(path)
    if array.ndim == 3:
        array = np.dot(array[..., :3], [0.2989, 0.5870, 0.1140])
    if not np.isfinite(array).all():
        raise ValueError(f"image {path} contains non-finite values")
    # Raster rows run top-to-bottom; Demeter tensors use a lower-left origin.
    image = coerce_image(np.flip(np.asarray(array), axis=0).copy())
    if size is not None and tuple(image.shape[-2:]) != tuple(size):
        image = F.interpolate(
            image,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return image, path
