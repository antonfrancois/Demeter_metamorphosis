"""Backend-specific native file dialogs."""

from pathlib import Path

import matplotlib.pyplot as plt

from ..images import IMAGE_BANK, PROJECT_ROOT


def choose_file(
    purpose: str,
    *,
    output_path: Path | None,
) -> Path | None:
    save = purpose == "save_setup"
    if save:
        title = "Save spline setup"
        initial = output_path or PROJECT_ROOT / "draft" / "spline_setup.pt"
        qt_filter = "PyTorch setup (*.pt)"
        tk_types = (("PyTorch setup", "*.pt"),)
    elif purpose == "load_setup":
        title = "Load spline setup"
        initial = PROJECT_ROOT / "draft"
        qt_filter = "PyTorch setup (*.pt *.pth);;All files (*)"
        tk_types = (("PyTorch setup", "*.pt *.pth"), ("All files", "*"))
    elif purpose == "field":
        title = "Load scalar field"
        initial = PROJECT_ROOT / "draft"
        qt_filter = "Field files (*.pt *.pth *.npy *.npz);;All files (*)"
        tk_types = (("Field files", "*.pt *.pth *.npy *.npz"), ("All files", "*"))
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
                    options["defaultextension"] = ".pt"
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
