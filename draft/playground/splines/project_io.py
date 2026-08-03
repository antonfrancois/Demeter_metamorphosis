"""Atomic project persistence for the spline lab."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from demeter.utils.spline_data import (
    TimedImageBatch,
    load_timed_image_directory,
    save_timed_image_directory,
)

from .core import (
    TRAJECTORY_FIELDS,
    SplineSetup,
    SplineTrajectory,
    load_setup,
)
from .registration import RegistrationResult


SETUP_FILENAME = "spline_setup.pt"
TRAJECTORY_FILENAME = "trajectory.pt"
OPTIMIZATION_FILENAME = "optimization.pt"


@dataclass(frozen=True)
class LoadedProject:
    setup: SplineSetup
    trajectory: SplineTrajectory | None
    registration: RegistrationResult | None


def save_project(
    setup: SplineSetup,
    destination: str | Path,
    *,
    trajectory: SplineTrajectory | None = None,
    registration: RegistrationResult | None = None,
) -> Path:
    """Save the setup and every available computed artifact atomically."""
    if registration is not None and trajectory is None:
        raise ValueError("an optimization cannot be saved without its trajectory")
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
        torch.save(setup.payload(), staged / SETUP_FILENAME)
        if trajectory is not None:
            torch.save(trajectory.payload(), staged / TRAJECTORY_FILENAME)
        if registration is not None:
            torch.save(registration.payload(), staged / OPTIMIZATION_FILENAME)
        staged.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def load_project(directory: str | Path) -> LoadedProject:
    """Load every artifact present in one spline project directory."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"project directory does not exist: {directory}")

    setup_path = directory / SETUP_FILENAME
    if not setup_path.is_file():
        raise FileNotFoundError(f"project is missing {SETUP_FILENAME}: {directory}")
    setup = load_setup(setup_path)
    image_batch = load_timed_image_directory(directory)
    if (
        image_batch.source.shape != setup.source.shape
        or image_batch.target.shape != setup.target.shape
        or any(
            abs(saved - expected) > 1e-12
            for saved, expected in zip(
                image_batch.target_times,
                setup.target_times,
                strict=True,
            )
        )
    ):
        raise ValueError("project images.csv does not match spline_setup.pt")

    trajectory_path = directory / TRAJECTORY_FILENAME
    optimization_path = directory / OPTIMIZATION_FILENAME
    trajectory = (
        _load_trajectory(trajectory_path, setup)
        if trajectory_path.is_file()
        else None
    )
    if optimization_path.exists() and trajectory is None:
        raise ValueError("project optimization.pt requires trajectory.pt")
    registration = (
        _load_registration(optimization_path, setup, trajectory)
        if optimization_path.is_file() and trajectory is not None
        else None
    )
    return LoadedProject(setup, trajectory, registration)


def _load_payload(path: Path, label: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a valid {label} payload")
    return payload


def _load_trajectory(path: Path, setup: SplineSetup) -> SplineTrajectory:
    payload = _load_payload(path, "trajectory")
    nodes = setup.parameters.n_steps + 1
    height, width = setup.size
    scalar_shape = (nodes, 1, height, width)
    vector_shape = (nodes, 2, height, width)
    values = {
        name: _trajectory_tensor(payload, name, scalar_shape, path)
        for name in (
            "images",
            "deformed_source",
            "photometric_only",
            "momentum",
            "force",
            "acceleration",
            "jerk",
        )
    }
    values.update(
        {
            name: _trajectory_tensor(payload, name, vector_shape, path)
            for name in ("velocity", "vector_momentum")
        }
    )
    values["target_mse"] = _trajectory_tensor(
        payload,
        "target_mse",
        (setup.target.shape[0], nodes),
        path,
    )

    energy_payload = payload.get("field_energies")
    if not isinstance(energy_payload, dict):
        raise ValueError(f"{path} is missing field_energies")
    energies = {
        name: _trajectory_tensor(energy_payload, name, (nodes,), path)
        for name in TRAJECTORY_FIELDS
    }
    elapsed = _elapsed_seconds(payload, path)
    return SplineTrajectory(
        **values,
        field_energies=energies,
        elapsed_seconds=elapsed,
    )


def _trajectory_tensor(
    payload: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    path: Path,
) -> torch.Tensor:
    value = payload.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"{path} has no tensor {name!r}")
    tensor = value
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{path} tensor {name!r} is not floating point")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{path} tensor {name!r} has shape {tuple(tensor.shape)}, expected {shape}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} tensor {name!r} contains non-finite values")
    return tensor.detach().cpu().contiguous()


def _elapsed_seconds(payload: dict[str, Any], path: Path) -> float:
    try:
        elapsed = float(payload["elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has no valid elapsed_seconds") from error
    if not isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{path} elapsed_seconds must be finite and nonnegative")
    return elapsed


def _load_registration(
    path: Path,
    setup: SplineSetup,
    trajectory: SplineTrajectory,
) -> RegistrationResult:
    payload = _load_payload(path, "optimization")
    model = payload.get("model")
    if model not in ("classic", "splines"):
        raise ValueError(f"{path} has an invalid optimization model")
    if model != setup.parameters.model:
        raise ValueError("optimization.pt model does not match spline_setup.pt")
    if "loss_stock" not in payload:
        raise ValueError(f"{path} is missing loss_stock")
    return RegistrationResult(
        setup=setup,
        trajectory=trajectory,
        loss_stock=payload["loss_stock"],
        elapsed_seconds=_elapsed_seconds(payload, path),
        model=model,
    )
