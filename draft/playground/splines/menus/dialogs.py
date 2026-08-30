"""Tk file dialogs used by the TkAgg playground."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from ..images import IMAGE_BANK, PROJECT_ROOT


IMAGE_TYPES = (
    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
    ("All files", "*"),
)
FILE_DIALOGS = {
    "load_setup": (
        "Load spline setup",
        PROJECT_ROOT / "draft",
        (("Spline setup", "*.pt"),),
        "",
    ),
    "load_field": (
        "Load scalar field",
        PROJECT_ROOT / "draft",
        (("Scalar field", "*.pt *.npy *.npz"),),
        "",
    ),
    "save_setup": (
        "Save spline setup",
        PROJECT_ROOT / "draft" / "spline_setup.pt",
        (("Spline setup", "*.pt"),),
        ".pt",
    ),
    "save_field": (
        "Save scalar field",
        PROJECT_ROOT / "draft" / "field.pt",
        (("Scalar field", "*.pt"),),
        ".pt",
    ),
    "save_video": (
        "Save trajectory video",
        PROJECT_ROOT / "draft" / "trajectory.mp4",
        (("MP4 video", "*.mp4"),),
        ".mp4",
    ),
}


def _choose(action) -> str:
    root = tk.Tk()
    root.withdraw()
    try:
        return action()
    finally:
        root.destroy()


def choose_file(
    purpose: str,
    *,
    output_path: Path | None,
    initial_name: str | None = None,
) -> Path | None:
    save = purpose.startswith("save_")
    title, initial, filetypes, extension = FILE_DIALOGS.get(
        purpose,
        (f"Load {purpose}", IMAGE_BANK, IMAGE_TYPES, ""),
    )
    if purpose == "save_setup" and output_path is not None:
        initial = output_path
    if initial_name is not None:
        initial = initial.parent / initial_name
    options = {
        "title": title,
        "initialdir": str(initial.parent if initial.suffix else initial),
        "filetypes": filetypes,
    }
    if save:
        options.update(initialfile=initial.name, defaultextension=extension)
        action = lambda: filedialog.asksaveasfilename(**options)
    else:
        action = lambda: filedialog.askopenfilename(**options)
    selected = _choose(action)
    return Path(selected) if selected else None


def choose_files() -> tuple[Path, ...]:
    selected = _choose(
        lambda: filedialog.askopenfilenames(
            title="Add spline images",
            initialdir=str(IMAGE_BANK),
            filetypes=IMAGE_TYPES,
        )
    )
    return tuple(map(Path, selected))


def choose_directory(purpose: str) -> Path | None:
    title = {
        "load_timed_images": "Load timed image directory",
        "load_project": "Load spline project",
        "save_project": "Save spline project",
    }[purpose]
    initial = PROJECT_ROOT / "draft"
    if purpose == "save_project":
        action = lambda: filedialog.asksaveasfilename(
            title=title,
            initialdir=str(initial),
            initialfile="spline_project",
        )
    else:
        action = lambda: filedialog.askdirectory(title=title, initialdir=str(initial))
    selected = _choose(action)
    return Path(selected) if selected else None
