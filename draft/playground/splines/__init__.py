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

__all__ = [
    "SplineParameters",
    "SplineSetup",
    "SplineTrajectory",
    "cometric_squared_norm",
    "load_scalar_field",
    "load_setup",
    "resolve_device",
    "run_classic",
    "run_spline",
    "save_setup",
    "zero_setup",
]
