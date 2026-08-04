"""Backend-specific native file dialogs."""

from pathlib import Path

import matplotlib.pyplot as plt

from ..images import IMAGE_BANK, PROJECT_ROOT


def choose_file(
    purpose: str,
    *,
    output_path: Path | None,
    initial_name: str | None = None,
) -> Path | None:
    save = purpose in ("save_setup", "save_field", "save_video")
    default_extension = ""
    if purpose == "save_setup":
        title = "Save spline setup"
        initial = output_path or PROJECT_ROOT / "draft" / "spline_setup.pt"
        qt_filter = "PyTorch setup (*.pt)"
        tk_types = (("PyTorch setup", "*.pt"),)
        default_extension = ".pt"
    elif purpose == "load_setup":
        title = "Load spline setup"
        initial = PROJECT_ROOT / "draft"
        qt_filter = "PyTorch setup (*.pt *.pth);;All files (*)"
        tk_types = (("PyTorch setup", "*.pt *.pth"), ("All files", "*"))
    elif purpose == "load_field":
        title = "Load scalar field"
        initial = PROJECT_ROOT / "draft"
        qt_filter = "Field files (*.pt *.pth *.npy *.npz);;All files (*)"
        tk_types = (("Field files", "*.pt *.pth *.npy *.npz"), ("All files", "*"))
    elif purpose == "save_field":
        title = "Save scalar field"
        initial = PROJECT_ROOT / "draft" / "field.pt"
        qt_filter = "PyTorch field (*.pt *.pth)"
        tk_types = (("PyTorch field", "*.pt *.pth"),)
        default_extension = ".pt"
    elif purpose == "save_video":
        title = "Save trajectory video"
        initial = PROJECT_ROOT / "draft" / (initial_name or "trajectory_images.mp4")
        qt_filter = "MP4 video (*.mp4)"
        tk_types = (("MP4 video", "*.mp4"),)
        default_extension = ".mp4"
    else:
        title = f"Load {purpose}"
        initial = IMAGE_BANK
        qt_filter = "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)"
        tk_types = (("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*"))

    backend = plt.get_backend().lower()
    if "qt" in backend:
        try:
            from matplotlib.backends.qt_compat import QtWidgets

            dialog = QtWidgets.QFileDialog
            if save:
                filename, _ = dialog.getSaveFileName(None, title, str(initial), qt_filter)
            else:
                filename, _ = dialog.getOpenFileName(None, title, str(initial), qt_filter)
            return Path(filename) if filename else None
        except Exception:
            pass

    if "tk" in backend:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                options = {
                    "title": title,
                    "initialdir": str(initial.parent if Path(initial).suffix else initial),
                    "filetypes": tk_types,
                }
                if save:
                    options["initialfile"] = Path(initial).name
                    options["defaultextension"] = default_extension
                    filename = filedialog.asksaveasfilename(**options)
                else:
                    filename = filedialog.askopenfilename(**options)
            finally:
                root.destroy()
            return Path(filename) if filename else None
        except Exception:
            pass

    raise RuntimeError(
        "No file dialog is available for this backend; use a path or CLI option."
    )


def choose_files() -> tuple[Path, ...]:
    """Choose multiple spline images with the active GUI backend."""
    title = "Add spline images"
    image_filter = "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)"
    backend = plt.get_backend().lower()
    if "qt" in backend:
        try:
            from matplotlib.backends.qt_compat import QtWidgets

            filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
                None, title, str(IMAGE_BANK), image_filter
            )
            return tuple(Path(filename) for filename in filenames)
        except Exception:
            pass
    if "tk" in backend:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                filenames = filedialog.askopenfilenames(
                    title=title,
                    initialdir=str(IMAGE_BANK),
                    filetypes=(
                        ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                        ("All files", "*"),
                    ),
                )
            finally:
                root.destroy()
            return tuple(Path(filename) for filename in filenames)
        except Exception:
            pass
    raise RuntimeError("No multiple-file dialog is available for this backend.")


def choose_directory(purpose: str = "load_timed_images") -> Path | None:
    """Choose an existing input directory or a new project directory path."""
    if purpose not in ("load_timed_images", "load_project", "save_project"):
        raise ValueError(f"unknown directory dialog purpose {purpose!r}")
    save = purpose == "save_project"
    title = {
        "load_timed_images": "Load timed image directory",
        "load_project": "Load spline project",
        "save_project": "Save spline project",
    }[purpose]
    initial = PROJECT_ROOT / "draft"
    backend = plt.get_backend().lower()
    if "qt" in backend:
        try:
            from matplotlib.backends.qt_compat import QtWidgets

            if save:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    None, title, str(initial / "spline_project"), "Directory name (*)"
                )
            else:
                filename = QtWidgets.QFileDialog.getExistingDirectory(
                    None, title, str(initial)
                )
            return Path(filename) if filename else None
        except Exception:
            pass
    if "tk" in backend:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                if save:
                    filename = filedialog.asksaveasfilename(
                        title=title,
                        initialdir=str(initial),
                        initialfile="spline_project",
                    )
                else:
                    filename = filedialog.askdirectory(
                        title=title,
                        initialdir=str(initial),
                    )
            finally:
                root.destroy()
            return Path(filename) if filename else None
        except Exception:
            pass
    raise RuntimeError("No directory dialog is available for this backend.")
