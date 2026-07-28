"""Interactive tools for exploring 2D metamorphosis splines."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = str(PROJECT_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from .core import (
    SplineParameters,
    SplineSetup,
    SplineTrajectory,
    cometric_squared_norm,
    load_scalar_field,
    load_setup,
    resolve_device,
    run_classic,
    run_spline,
    save_setup,
    zero_setup,
)
from .registration import RegistrationResult, register_classic, register_spline
from demeter.utils.spline_data import (
    TimedImageBatch,
    load_timed_image_directory,
    save_timed_image_directory,
)

__all__ = [
    "SplineParameters",
    "SplineSetup",
    "SplineTrajectory",
    "RegistrationResult",
    "TimedImageBatch",
    "cometric_squared_norm",
    "load_scalar_field",
    "load_setup",
    "load_timed_image_directory",
    "register_classic",
    "register_spline",
    "resolve_device",
    "run_classic",
    "run_spline",
    "save_setup",
    "save_timed_image_directory",
    "zero_setup",
]
