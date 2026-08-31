"""Command-line entry point for the metamorphosis spline playground."""

from __future__ import annotations

import argparse
from dataclasses import replace
from .app import SplinePlayground
from .images import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    load_image,
)
import matplotlib.pyplot as plt

from demeter.utils.spline_data import load_timed_image_directory
from .core import (
    SplineParameters,
    SplineSetup,
    load_scalar_field,
    load_setup,
    zero_setup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", help="Source image path or im2Dbank shorthand"
    )
    parser.add_argument(
        "target", nargs="?", help="Target image path or im2Dbank shorthand"
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--setup", help="Saved spline playground setup")
    inputs.add_argument("--timed-images", help="Directory containing images.csv")
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
        help="Interior mesh nodes (default: none)",
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--kernel", choices=("sobolev", "gaussian"))
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--cg-tolerance", type=float)
    parser.add_argument("--model", choices=("classic", "splines"))
    parser.add_argument("--cost-cst", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--lbfgs-lr", type=float)
    parser.add_argument("--output", help="Path used by Ctrl+S")
    parser.add_argument("--field", help="Scalar field loaded before launch")
    parser.add_argument(
        "--field-kind",
        choices=("momentum", "acceleration", "jerk", "control"),
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
    physical = {
        name: value
        for name in (
            "alpha",
            "beta",
            "gamma",
            "rho",
            "cg_tolerance",
            "kernel",
            "sigma",
            "model",
        )
        if (value := getattr(args, name)) is not None
    }
    solver = replace(
        parameters.spline,
        **{
            name: value
            for name, value in (
                ("steps", args.steps),
                ("cost", args.cost_cst),
                ("iterations", args.iterations),
                ("learning_rate", args.lbfgs_lr),
            )
            if value is not None
        },
    )
    if physical.get("kernel") == "gaussian" and "model" not in physical:
        physical["model"] = "classic"
    control_times = parameters.control_times
    if args.control_steps is not None:
        control_times = tuple(step / solver.steps for step in args.control_steps)
    return replace(
        parameters,
        **physical,
        spline=solver,
        control_times=control_times,
    )


def _replace_parameters(
    setup: SplineSetup,
    parameters: SplineParameters,
) -> SplineSetup:
    controls = setup.images.source.new_zeros(
        (len(parameters.control_times),) + tuple(setup.images.source.shape)
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
            controls[index] = setup.variables.control_jerks[old_index]
    return replace(
        setup,
        parameters=parameters,
        variables=replace(setup.variables, control_jerks=controls),
    )


def _initial_setup(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> SplineSetup:
    size = tuple(args.size) if args.size else None
    if (args.setup or args.timed_images) and any((args.source, args.target, size)):
        parser.error("source, target, and --size cannot accompany a saved input")
    if args.setup:
        setup = load_setup(args.setup)
        parameters = _parameter_overrides(args, setup.parameters)
        return _replace_parameters(setup, parameters)
    if args.timed_images:
        batch = load_timed_image_directory(args.timed_images)
        parameters = _parameter_overrides(args, SplineParameters(model="splines"))
        return zero_setup(
            batch.source,
            batch.target,
            parameters,
            source_path=batch.source_path,
            target_times=batch.target_times,
            target_paths=batch.target_paths,
        )
    source, source_path = load_image(args.source or DEFAULT_SOURCE, size)
    target, target_path = load_image(
        args.target or DEFAULT_TARGET,
        tuple(source.shape[-2:]),
    )
    return zero_setup(
        source,
        target,
        _parameter_overrides(args, SplineParameters()),
        source_path=source_path,
        target_paths=(target_path,),
    )


def _apply_initial_field(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    setup: SplineSetup,
) -> None:
    if not args.field:
        return
    field = load_scalar_field(
        args.field,
        setup.size,
        dtype=setup.images.source.dtype,
    )
    field_attribute = {
        "momentum": "initial_momentum",
        "acceleration": "initial_acceleration",
        "jerk": "initial_jerk",
    }.get(args.field_kind)
    if field_attribute is not None:
        setattr(setup.variables, field_attribute, field)
        return
    if setup.n_controls == 0:
        parser.error("--field-kind control requires at least one control node")
    if not 0 <= args.control_index < setup.n_controls:
        parser.error(f"--control-index must be between 0 and {setup.n_controls - 1}")
    setup.variables.control_jerks[args.control_index] = field


def main(argv: list[str] | None = None) -> SplinePlayground:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup = _initial_setup(args, parser)
    _apply_initial_field(args, parser, setup)

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
