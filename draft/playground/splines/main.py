"""Command-line entry point for the metamorphosis spline playground."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = str(PROJECT_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from draft.playground.splines.app import SplinePlayground
from draft.playground.splines.images import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    load_image,
)
import matplotlib.pyplot as plt

from demeter.utils.spline_data import load_timed_image_directory
from draft.playground.splines.core import (
    SplineParameters,
    SplineSetup,
    load_scalar_field,
    load_setup,
    zero_setup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="Source image path or im2Dbank shorthand")
    parser.add_argument("target", nargs="?", help="Target image path or im2Dbank shorthand")
    parser.add_argument("--setup", help="Saved spline playground setup")
    parser.add_argument("--timed-images", help="Directory containing images.csv")
    parser.add_argument("--size", nargs=2, type=int, metavar=("H", "W"))
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device (default: auto, preferring CUDA when available)",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--control-steps",
        nargs="*",
        type=int,
        help="Interior mesh nodes; pass the flag without values for no controls",
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--kernel", choices=("sobolev", "gaussian"))
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--cg-eps", type=float)
    parser.add_argument("--model", choices=("classic", "splines"))
    parser.add_argument("--cost-cst", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", help="Path used by Ctrl+S")
    parser.add_argument("--field", help="Scalar field loaded before launch")
    parser.add_argument(
        "--field-kind",
        choices=("momentum", "force", "jerk", "control"),
        default="momentum",
    )
    parser.add_argument("--control-index", type=int, default=0)
    parser.add_argument("--run", action="store_true", help="Run immediately")
    parser.add_argument("--register", action="store_true", help="Optimize immediately")
    parser.add_argument("--screenshot", help="Save the rendered application")
    parser.add_argument("--no-show", action="store_true")
    return parser


def _parameter_overrides(
    args: argparse.Namespace,
    parameters: SplineParameters,
) -> SplineParameters:
    values = parameters.as_dict()
    for argument, name in (
        (args.alpha, "alpha"),
        (args.beta, "beta"),
        (args.gamma, "gamma"),
        (args.rho, "rho"),
        (args.cg_eps, "cg_eps"),
        (getattr(args, "kernel", None), "kernel"),
        (getattr(args, "sigma", None), "sigma"),
        (getattr(args, "model", None), "model"),
        (getattr(args, "cost_cst", None), "cost_cst"),
        (getattr(args, "iterations", None), "iterations"),
    ):
        if argument is not None:
            values[name] = argument
    if args.steps is not None:
        values["n_steps"] = args.steps
    if args.control_steps is not None:
        control_steps = tuple(args.control_steps)
        values["control_steps"] = control_steps
        values["control_times"] = tuple(
            step / values["n_steps"] for step in control_steps
        )
    if values.get("kernel") == "gaussian" and getattr(args, "model", None) is None:
        values["model"] = "classic"
    return SplineParameters.from_dict(values)


def _replace_parameters(
    setup: SplineSetup,
    parameters: SplineParameters,
) -> SplineSetup:
    controls = setup.source.new_zeros(
        (len(parameters.control_steps),) + tuple(setup.source.shape)
    )
    for index, time in enumerate(parameters.control_times):
        old_index = next(
            (
                candidate
                for candidate, old_time in enumerate(setup.parameters.control_times)
                if abs(old_time - time) <= 1e-12
            ),
            None,
        )
        if old_index is not None:
            controls[index] = setup.control_jerks[old_index]
    return replace(setup, parameters=parameters, control_jerks=controls)


def main(argv: list[str] | None = None) -> SplinePlayground:
    parser = build_parser()
    args = parser.parse_args(argv)
    size = tuple(args.size) if args.size else None

    if args.setup:
        if args.source or args.target or size is not None:
            parser.error("source, target, and --size cannot be combined with --setup")
        setup = load_setup(args.setup)
        parameters = _parameter_overrides(args, setup.parameters)
        if parameters != setup.parameters:
            setup = _replace_parameters(setup, parameters)
    elif args.timed_images:
        if args.source or args.target or size is not None:
            parser.error("source, target, and --size cannot be combined with --timed-images")
        batch = load_timed_image_directory(args.timed_images)
        parameters = _parameter_overrides(args, SplineParameters(model="splines"))
        setup = zero_setup(
            batch.source,
            batch.target,
            parameters,
            source_path=batch.source_path,
            target_path=batch.target_paths[-1],
            target_times=batch.target_times,
            target_paths=batch.target_paths,
        )
    else:
        source, source_path = load_image(args.source or DEFAULT_SOURCE, size)
        target, target_path = load_image(
            args.target or DEFAULT_TARGET,
            tuple(source.shape[-2:]),
        )
        n_steps = args.steps if args.steps is not None else 16
        if args.control_steps is None:
            control_steps = (n_steps // 2,) if n_steps > 1 else ()
        else:
            control_steps = tuple(args.control_steps)
        parameters = SplineParameters(
            alpha=args.alpha if args.alpha is not None else 0.2,
            beta=args.beta if args.beta is not None else 0.2,
            gamma=args.gamma if args.gamma is not None else 0.001,
            kernel=args.kernel if args.kernel is not None else "sobolev",
            sigma=args.sigma if args.sigma is not None else 3.0,
            rho=args.rho if args.rho is not None else 0.5,
            cg_eps=args.cg_eps if args.cg_eps is not None else 1e-5,
            n_steps=n_steps,
            control_steps=control_steps,
            model=(
                args.model
                if args.model is not None
                else ("classic" if args.kernel == "gaussian" else "splines")
            ),
            cost_cst=args.cost_cst if args.cost_cst is not None else 0.01,
            iterations=args.iterations if args.iterations is not None else 10,
        )
        setup = zero_setup(
            source,
            target,
            parameters,
            source_path=source_path,
            target_path=target_path,
        )

    if args.field:
        field = load_scalar_field(
            args.field,
            setup.size,
            dtype=setup.source.dtype,
        )
        if args.field_kind == "momentum":
            setup.initial_momentum = field
        elif args.field_kind == "force":
            setup.initial_force = field
        elif args.field_kind == "jerk":
            setup.initial_jerk = field
        else:
            if setup.n_controls == 0:
                parser.error("--field-kind control requires at least one control node")
            if not 0 <= args.control_index < setup.n_controls:
                parser.error(
                    f"--control-index must be between 0 and {setup.n_controls - 1}"
                )
            setup.control_jerks[args.control_index] = field

    app = SplinePlayground(
        setup,
        device=args.device,
        output_path=args.output,
    )
    if args.run or args.screenshot:
        app.run()
        if app.last_error is not None:
            raise RuntimeError("spline integration failed") from app.last_error
    if args.register:
        app.register()
        if app.last_error is not None:
            raise RuntimeError("registration failed") from app.last_error
    if args.screenshot:
        app.fig.savefig(
            args.screenshot,
            dpi=140,
            facecolor=app.fig.get_facecolor(),
        )
    if not args.no_show:
        plt.show()
    return app


if __name__ == "__main__":
    main()
