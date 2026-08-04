import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from draft.playground.splines import video_export
from draft.playground.splines.app import SplinePlayground
from draft.playground.splines.core import SplineParameters, zero_setup


def test_trajectory_video_schedule_is_always_four_seconds():
    assert video_export.DEFAULT_VIDEO_FPS == 30
    assert video_export.trajectory_video_filename(None, True) == "trajectory_images.mp4"
    assert (
        video_export.trajectory_video_filename("velocity", True)
        == "trajectory_images_velocity.mp4"
    )
    assert (
        video_export.trajectory_video_filename("velocity", False)
        == "trajectory_velocity.mp4"
    )

    for fps in (1, 10, 25, 60):
        indices = video_export.trajectory_frame_indices(7, fps)
        assert len(indices) == 4 * fps
        assert indices[:fps] == (0,) * fps
        assert indices[-fps:] == (7,) * fps
        assert indices[fps] == 0
        assert indices[-fps - 1] == 7
        assert all(left <= right for left, right in zip(indices, indices[1:]))

    with pytest.raises(ValueError, match="n_steps"):
        video_export.trajectory_frame_indices(0)
    with pytest.raises(ValueError, match="fps"):
        video_export.trajectory_frame_indices(2, 0)


def test_save_video_uses_image_canvas_and_restores_current_panel(tmp_path, monkeypatch):
    source = torch.zeros(1, 1, 6, 6)
    source[..., 2:5, 1:5] = 0.7
    setup = zero_setup(
        source,
        torch.roll(source, shifts=1, dims=-1),
        SplineParameters(rho=0.5, n_steps=2),
    )
    setup.initial_momentum[..., 2:5, 2:6] = 0.02
    app = SplinePlayground(setup, device="cpu")
    with pytest.raises(ValueError, match="run or load"):
        app.save_video(tmp_path / "missing.mp4")

    dialog_request = []
    monkeypatch.setattr(
        app,
        "_choose_file",
        lambda purpose, **options: dialog_request.append((purpose, options)),
    )
    app.current_field = "velocity"
    app.save_video_dialog()
    assert dialog_request == [
        ("save_video", {"initial_name": "trajectory_images_velocity.mp4"})
    ]

    app.run_spline()
    assert app.cache is not None
    app.current_image_mode = "deformation"
    app.current_field = "velocity"
    app.show_current_image = True
    app.renderer.vector_spacing = 3
    app.set_time_index(1)
    app._render_current()
    original_title = app.current_ax.get_title()

    monkeypatch.setattr(video_export.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        video_export.save_current_panel_video(
            tmp_path / "unavailable.mp4",
            figure=app.fig,
            renderer=app.renderer,
            source=app.source,
            trajectory=app.cache,
            image_mode=app.current_image_mode,
            current_field=app.current_field,
            show_image=app.show_current_image,
            restore_index=app._time_index(),
        )

    calls = []
    render_current = app.renderer.render_current

    def recording_render(
        source_arg,
        trajectory,
        image_mode,
        current_field,
        index,
        show_image,
    ):
        calls.append((image_mode, current_field, index, show_image))
        return render_current(
            source_arg,
            trajectory,
            image_mode,
            current_field,
            index,
            show_image,
        )

    monkeypatch.setattr(app.renderer, "render_current", recording_render)
    monkeypatch.setattr(video_export.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fake_ffmpeg(command, *, check, capture_output, text):
        assert check and capture_output and text
        fps = int(command[command.index("-framerate") + 1])
        frame_count = int(command[command.index("-frames:v") + 1])
        assert fps == video_export.DEFAULT_VIDEO_FPS
        assert frame_count == 4 * fps
        assert command[command.index("-c:v") + 1] == "libx264"
        assert command[command.index("-pix_fmt") + 1] == "yuv420p"
        frame_directory = Path(command[command.index("-i") + 1]).parent
        frames = sorted(frame_directory.glob("frame_*.png"))
        assert len(frames) == frame_count
        assert plt.imread(frames[fps]).shape[:2] == (720, 720)
        assert all(frame.read_bytes() == frames[0].read_bytes() for frame in frames[:fps])
        assert all(frame.read_bytes() == frames[-1].read_bytes() for frame in frames[-fps:])
        Path(command[-1]).write_bytes(b"mock mp4")

    monkeypatch.setattr(video_export.subprocess, "run", fake_ffmpeg)
    destination = app.save_video(tmp_path / "trajectory")

    assert destination == tmp_path / "trajectory.mp4"
    assert destination.read_bytes() == b"mock mp4"
    assert int(app.time_slider.val) == 1
    assert app.current_ax.get_title() == original_title
    assert calls[-1] == ("deformation", "velocity", 1, True)
    assert {index for _mode, _field, index, _show in calls} == {0, 1, 2}
    assert all(
        mode == "deformation" and field == "velocity" and show
        for mode, field, _index, show in calls
    )
    assert "Saved video" in app.status_text.get_text()
    plt.close(app.fig)
