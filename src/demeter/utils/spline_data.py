"""Timed image-series I/O for metamorphosis spline regression."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import shutil
import tempfile

import matplotlib.image as mpimg
import numpy as np
import torch
from torch import Tensor


MANIFEST_NAME = "images.csv"


def _load_grayscale(path: Path) -> Tensor:
    array = mpimg.imread(path)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / np.iinfo(array.dtype).max
    else:
        array = array.astype(np.float32, copy=False)
    if array.ndim == 3:
        array = np.dot(array[..., :3], (0.2989, 0.5870, 0.1140))
    if (
        array.ndim != 2
        or not np.isfinite(array).all()
        or array.min() < 0
        or array.max() > 1
    ):
        raise ValueError(f"image {path} must be a finite grayscale image")
    image = torch.as_tensor(np.asarray(array).copy()).float()
    return image[None, None].contiguous()


@dataclass(frozen=True)
class TimedImageBatch:
    """One source at time zero and an ordered batch of later observations."""

    source: Tensor
    target: Tensor
    target_times: tuple[float, ...]
    source_path: str = ""
    target_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source = self.source.detach().cpu().contiguous()
        target = self.target.detach().cpu().contiguous()
        times = tuple(float(time) for time in self.target_times)
        if source.ndim != 4 or source.shape[:2] != (1, 1):
            raise ValueError("source must have shape [1, 1, H, W]")
        if target.ndim != 4 or target.shape[1:] != source.shape[1:]:
            raise ValueError("target must have shape [N, 1, H, W] matching source")
        if not target.shape[0] or len(times) != target.shape[0]:
            raise ValueError("target_times must contain one time per target")
        if not torch.is_floating_point(source) or not torch.is_floating_point(target):
            raise TypeError("source and target images must be floating point")
        if source.dtype != target.dtype:
            raise ValueError("source and target images must share one dtype")
        if any(not isfinite(time) or not 0 < time <= 1 for time in times):
            raise ValueError("target times must be finite and lie in (0, 1]")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("target times must be strictly increasing")
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError("images must contain only finite values")
        if source.min() < 0 or source.max() > 1 or target.min() < 0 or target.max() > 1:
            raise ValueError("images must have values in [0, 1]")
        paths = tuple(str(path) for path in self.target_paths)
        if paths and len(paths) != len(times):
            raise ValueError("target_paths must contain one path per target")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "target_times", times)
        object.__setattr__(self, "source_path", str(self.source_path))
        object.__setattr__(self, "target_paths", paths)


def load_timed_image_directory(directory: str | Path) -> TimedImageBatch:
    """Load ``images.csv`` and its source/target image files."""
    root = Path(directory).expanduser().resolve()
    manifest = root / MANIFEST_NAME
    if not root.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"expected {manifest}")

    rows = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filename", "time"]:
            raise ValueError("images.csv must have exactly: filename,time")
        for row in reader:
            relative = Path(row["filename"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe image path {relative}")
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise FileNotFoundError(f"missing image {relative}")
            try:
                time = float(row["time"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid time for {relative}") from error
            if not isfinite(time) or not 0 <= time <= 1:
                raise ValueError("image times must be finite and lie in [0, 1]")
            rows.append((time, path))

    sources = [(time, path) for time, path in rows if time == 0]
    targets = sorted((time, path) for time, path in rows if time > 0)
    if len(sources) != 1 or not targets:
        raise ValueError("images.csv must contain one source at time 0 and targets")
    if len({time for time, _ in rows}) != len(rows):
        raise ValueError("image times must be unique")

    source = _load_grayscale(sources[0][1])
    target_images = [_load_grayscale(path) for _, path in targets]
    if any(image.shape != source.shape for image in target_images):
        raise ValueError("all images in a timed series must have the same shape")
    return TimedImageBatch(
        source=source,
        target=torch.cat(target_images),
        target_times=tuple(time for time, _ in targets),
        source_path=str(sources[0][1]),
        target_paths=tuple(str(path) for _, path in targets),
    )


def save_timed_image_directory(
    batch: TimedImageBatch,
    directory: str | Path,
) -> Path:
    """Atomically save a timed image batch as images plus ``images.csv``."""
    destination = Path(directory).expanduser()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        records = [("source.png", 0.0, batch.source)]
        records.extend(
            (f"target_{index:03d}.png", time, batch.target[index - 1:index])
            for index, time in enumerate(batch.target_times, start=1)
        )
        for filename, _time, image in records:
            mpimg.imsave(
                temporary / filename,
                image[0, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        with (temporary / MANIFEST_NAME).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("filename", "time"))
            writer.writerows(
                (filename, format(time, ".17g"))
                for filename, time, _image in records
            )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
