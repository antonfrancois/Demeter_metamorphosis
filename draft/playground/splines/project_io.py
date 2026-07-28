"""Atomic timed-image project persistence for the spline lab."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import torch

from demeter.utils.spline_data import TimedImageBatch, save_timed_image_directory

from .core import SplineSetup, SplineTrajectory

if TYPE_CHECKING:
    from .registration import RegistrationResult


def save_timed_project(
    setup: SplineSetup,
    destination: str | Path,
    *,
    trajectory: SplineTrajectory | None = None,
    registration: RegistrationResult | None = None,
) -> Path:
    """Save images, setup, trajectory, and optimization as one atomic directory."""
    destination = Path(destination).expanduser()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    staged = staging_root / destination.name
    try:
        save_timed_image_directory(
            TimedImageBatch(
                setup.source,
                setup.target,
                setup.target_times,
                setup.source_path,
                setup.target_paths,
            ),
            staged,
        )
        torch.save(setup.payload(), staged / "spline_setup.pt")
        if trajectory is not None:
            torch.save(trajectory.payload(), staged / "trajectory.pt")
        if registration is not None:
            torch.save(registration.payload(), staged / "optimization.pt")
        staged.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination
