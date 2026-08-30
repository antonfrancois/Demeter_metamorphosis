"""Project persistence for the spline playground."""

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import tempfile

import torch

from demeter.utils.spline_data import save_timed_image_directory

from .core import SplineSetup, SplineTrajectory, load_setup, save_setup
from .registration import RegistrationResult


SETUP_FILENAME = "spline_setup.pt"
TRAJECTORY_FILENAME = "trajectory.pt"
OPTIMIZATION_FILENAME = "optimization.pt"


@dataclass(frozen=True)
class LoadedProject:
    setup: SplineSetup
    trajectory: SplineTrajectory | None = None
    registration: RegistrationResult | None = None


def project_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    if directory.is_file():
        directory = directory.parent
    if not (directory / SETUP_FILENAME).is_file():
        raise FileNotFoundError(f"expected {directory / SETUP_FILENAME}")
    return directory


def save_project(
    setup: SplineSetup,
    destination: str | Path,
    *,
    trajectory: SplineTrajectory | None = None,
    registration: RegistrationResult | None = None,
) -> Path:
    destination = Path(destination).expanduser()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    project = staging / destination.name
    try:
        save_timed_image_directory(setup.images, project)
        save_setup(setup, project / SETUP_FILENAME)
        if trajectory is not None:
            torch.save(trajectory, project / TRAJECTORY_FILENAME)
        if registration is not None:
            torch.save(registration, project / OPTIMIZATION_FILENAME)
        project.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _load(path: Path, expected_type):
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, expected_type):
        raise ValueError(f"{path} does not contain {expected_type.__name__}")
    return value


def load_project(path: str | Path) -> LoadedProject:
    directory = project_directory(path)
    setup = load_setup(directory / SETUP_FILENAME)
    trajectory = _load(directory / TRAJECTORY_FILENAME, SplineTrajectory)
    registration = _load(directory / OPTIMIZATION_FILENAME, RegistrationResult)
    if registration is not None:
        if trajectory is None:
            raise ValueError("optimization requires a saved trajectory")
        registration = replace(registration, setup=setup, trajectory=trajectory)
    return LoadedProject(setup, trajectory, registration)
