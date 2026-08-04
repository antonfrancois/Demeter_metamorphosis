"""MP4 export for the spline playground's current trajectory image."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from matplotlib.transforms import Bbox
import torch

from .core import SplineTrajectory
from .rendering import SplineRenderer


DEFAULT_VIDEO_FPS = 30
VIDEO_SECONDS = 4
ENDPOINT_PADDING_SECONDS = 1
VIDEO_LONG_EDGE_PIXELS = 720


def trajectory_video_filename(
    current_field: str | None,
    show_image: bool,
) -> str:
    """Return a filename describing exactly what is displayed in the video."""
    parts = ["images"] if show_image else []
    if current_field is not None:
        parts.append(current_field)
    if not parts:
        parts.append("empty")
    return f"trajectory_{'_'.join(parts)}.mp4"


def trajectory_frame_indices(
    n_steps: int,
    fps: int = DEFAULT_VIDEO_FPS,
) -> tuple[int, ...]:
    """Return a four-second node schedule with one-second endpoint freezes."""
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a strictly positive integer")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps < 1:
        raise ValueError("fps must be a strictly positive integer")

    padding_frames = ENDPOINT_PADDING_SECONDS * fps
    moving_frames = (VIDEO_SECONDS - 2 * ENDPOINT_PADDING_SECONDS) * fps
    moving = tuple(
        round(frame * n_steps / (moving_frames - 1))
        for frame in range(moving_frames)
    )
    return (0,) * padding_frames + moving + (n_steps,) * padding_frames


def save_current_panel_video(
    destination: str | Path,
    *,
    figure: Any,
    renderer: SplineRenderer,
    source: torch.Tensor,
    trajectory: SplineTrajectory,
    image_mode: str,
    current_field: str | None,
    show_image: bool,
    restore_index: int,
    fps: int = DEFAULT_VIDEO_FPS,
) -> Path:
    """Save only the current image canvas as a four-second H.264 MP4."""
    destination = Path(destination).expanduser()
    if not destination.suffix:
        destination = destination.with_suffix(".mp4")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("video output must use the .mp4 extension")
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise RuntimeError("ffmpeg is required to save MP4 videos")

    n_steps = trajectory.images.shape[0] - 1
    indices = trajectory_frame_indices(n_steps, fps)
    if not 0 <= restore_index <= n_steps:
        raise ValueError(f"restore_index must lie between 0 and {n_steps}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        crop = _current_image_crop(figure, renderer)
        with tempfile.TemporaryDirectory(
            dir=destination.parent,
            prefix=f".{destination.stem}.video.",
        ) as temporary_name:
            temporary = Path(temporary_name)
            node_paths = _save_node_frames(
                temporary,
                figure,
                renderer,
                source,
                trajectory,
                image_mode,
                current_field,
                show_image,
                crop,
            )
            for frame, index in enumerate(indices):
                shutil.copyfile(
                    node_paths[index],
                    temporary / f"frame_{frame:04d}.png",
                )

            encoded = temporary / "video.mp4"
            command = [
                encoder,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(temporary / "frame_%04d.png"),
                "-frames:v",
                str(len(indices)),
                "-an",
                "-c:v",
                "libx264",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or "").strip() or "unknown encoding error"
                raise RuntimeError(
                    f"ffmpeg failed to encode the video: {detail}"
                ) from error
            encoded.replace(destination)
    finally:
        renderer.render_current(
            source,
            trajectory,
            image_mode,
            current_field,
            restore_index,
            show_image,
        )
        figure.canvas.draw_idle()
    return destination


def _current_image_crop(
    figure: Any,
    renderer: SplineRenderer,
) -> Bbox:
    figure.canvas.draw()
    return renderer.current_ax.get_window_extent().transformed(
        figure.dpi_scale_trans.inverted()
    )


def _save_node_frames(
    directory: Path,
    figure: Any,
    renderer: SplineRenderer,
    source: torch.Tensor,
    trajectory: SplineTrajectory,
    image_mode: str,
    current_field: str | None,
    show_image: bool,
    crop: Bbox,
) -> tuple[Path, ...]:
    paths = []
    frame_dpi = VIDEO_LONG_EDGE_PIXELS / max(crop.width, crop.height)
    for index in range(trajectory.images.shape[0]):
        renderer.render_current(
            source,
            trajectory,
            image_mode,
            current_field,
            index,
            show_image,
        )
        path = directory / f"node_{index:04d}.png"
        figure.savefig(
            path,
            format="png",
            dpi=frame_dpi,
            bbox_inches=crop,
            pad_inches=0,
            facecolor=figure.get_facecolor(),
            transparent=False,
        )
        paths.append(path)
    return tuple(paths)
