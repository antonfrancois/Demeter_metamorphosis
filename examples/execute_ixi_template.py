import os, re
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Iterator, Optional, Callable, Tuple, List, Dict, Union
from pathlib import Path
from nibabel.processing import resample_from_to

import SimpleITK as sitk
import itk

# data base management
import sqlite3, json, time, datetime
from contextlib import contextmanager

import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
import demeter.utils.rigid_exploration as rg
import demeter.utils.reproducing_kernels as rk


def to_torch(arr: np.ndarray) -> torch.Tensor:
    """
    (X,Y,Z) numpy -> torch (1,1,Z,Y,X) float32.
    Nib canonical volumes are (X,Y,Z), so we permute to (Z,Y,X) for PyTorch 3D.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 1, 0)  # Z,Y,X
    return t.unsqueeze(0).unsqueeze(0).contiguous()  # 1,1,Z,Y,X


def normalize(img):
    quant = np.quantile(img, 0.99)
    img = np.clip(img, 0, quant)
    img /= img.max()
    return img


def simplify_segs(seg):
    # Create new label map (e.g. 0-5)
    new_seg = np.zeros_like(seg)

    # Define label sets
    CSF = [4, 43, 15, 14]
    GM = [3, 42, *range(1000, 1036), *range(2000, 2036)]
    WM = [2, 41, 77]
    SCGM = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
    BRAINSTEM = [16]

    # Map new labels
    # plt.imshow(np.isin(seg,GM))
    new_seg[np.isin(seg, CSF)] = 1
    new_seg[np.isin(seg, GM)] = 2
    new_seg[np.isin(seg, SCGM)] = 3
    new_seg[np.isin(seg, WM)] = 4
    new_seg[np.isin(seg, BRAINSTEM)] = 5

    return new_seg


def load_canonical(img_path: str) -> nib.spatialimages.SpatialImage:
    """Load image and convert to RAS+ canonical orientation (safer for affine math)."""
    img = nib.load(img_path)
    return nib.as_closest_canonical(img)


def save_like(target_img: nib.spatialimages.SpatialImage, new_data: np.ndarray, out_path: str, dtype=np.float32):
    """Save `new_data` in the same space/affine/header class as target_img."""
    new_img = target_img.__class__(new_data.astype(dtype), target_img.affine, target_img.header)
    nib.save(new_img, out_path)
    return out_path


# def load_row_template_data(template_folder):
#     template_name = "mni_icbm152_t1_tal_nlin_asym_09c.nii"
#     template_mask_name = "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"
#     template_segs_name = "mni_icbm152.auto_noCCseg.mgz"
#
#     tpl_img = load_canonical(os.path.join(template_folder, template_name))
#     tpl_msk_img = load_canonical(os.path.join(template_folder, template_mask_name))
#     tpl_segs_img = load_canonical(os.path.join(template_folder, template_seg_path, template_segs_name))
#
#     # Zero template outside its mask (optional)
#     tpl_data = tpl_img.get_fdata()
#     tpl_msk = tpl_msk_img.get_fdata() > 0.5
#     tpl_data_masked = np.where(tpl_msk, tpl_data, 0.0)
#
#     return tpl_data_masked, tpl_segs_img

def _ixi_number_from_folder(folder_name: str) -> Optional[int]:
    m = re.match(r"^IXI(\d+)-", folder_name)
    return int(m.group(1)) if m else None


def find_ixi_folder(base_path: str, number: Optional[int] = None) -> Iterator[str]:
    """
    Yield folder names in `base_path` matching the IXI format.

    Parameters
    ----------
    base_path : str
        Path where the folders are located.
    number : int, optional
        IXI subject number (e.g., 40 will match "IXI040-...").
        If None, all IXI folders are returned.

    Yields
    ------
    str
        Matching folder names.

    Usage:
    -----------
    folder = next(find_ixi_folder(ixi_folder, 1), None)
    print("First match:", folder)
    # or iterate through all folders
    for f in find_ixi_folder(ixi_folder):
        print("Folder:", f)
    """
    if number is not None:
        num_str = f"{number:03d}"
        pattern = re.compile(rf"^IXI{num_str}-")
    else:
        # match any IXI folder with 3-digit number
        pattern = re.compile(r"^IXI\d{3}-")

    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)) and pattern.match(folder):
            yield folder


def ensure_nifti(path: str | Path) -> Path:
    """
    Ensure the file is in NIfTI (.nii.gz) format.

    If input is already .nii.gz → return the path.
    If input is .mgz → convert to .nii.gz using nibabel, save next to original,
    and return the new path.

    Parameters
    ----------
    path : str | Path
        Path to input file (.nii.gz or .mgz).

    Returns
    -------
    Path
        Path to the .nii.gz file.
    """
    path = Path(path)
    # print("DEBUG:",path)
    # Case 1: Already .nii.gz
    if path.suffixes == [".nii", ".gz"] or path.suffix == ".nii":
        return path

    # Case 2: .mgz → convert
    if path.suffix == ".mgz":
        # print(os.listdir(path.parent))
        # print(path.with_suffix('.nii.gz'))
        # print(">> ",path.with_suffix('.nii.gz').name in  os.listdir(path.parent))
        if path.with_suffix('.nii.gz').name in os.listdir(path.parent):
            print("><", path.with_suffix('.nii.gz'))
            return path.with_suffix('.nii.gz')

        print(".mgz found and .nii.gz not found, converting to .nii.gz")
        out_path = path.with_suffix("")  # strip .mgz
        out_path = out_path.with_suffix(".nii.gz")
        img = nib.load(str(path))
        nib.save(img, str(out_path))
        return out_path

    raise ValueError(f"Unsupported file extension: {path.suffixes or path.suffix} in {path.name}")


def _affine_to_sitk(aff: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Convert a 4x4 RAS affine to (spacing, origin, direction) for SimpleITK."""
    # Extract spacing as norm of columns 0..2
    R = aff[:3, :3]
    spacing = np.linalg.norm(R, axis=0)
    spacing = np.where(spacing == 0, 1.0, spacing)
    # Normalize to get direction cosines (row-major flatten)
    Rn = (R / spacing).astype(float)
    # ITK direction is a flat tuple length 9
    direction = tuple(Rn.flatten(order="F"))  # column-major to match ITK’s convention
    origin = tuple(aff[:3, 3])
    return tuple(spacing.tolist()), origin, direction


def ensure_nrrd(in_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """
    Ensure we have a .nrrd on disk for a given .nii.gz or .mgz (or .nrrd already).
    Returns the .nrrd Path. Uses nibabel to read, SimpleITK to write with geometry.
    """
    in_path = Path(in_path)
    if in_path.suffix.lower() == ".nrrd":
        return in_path

    if out_dir is None:
        out_dir = in_path.parent
    out_dir = Path(out_dir)

    # Build target .nrrd name next to input
    stem = in_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]  # strip .nii.gz
    elif stem.endswith(".mgz"):
        stem = stem[:-4]  # strip .mgz
    out_path = out_dir / f"{stem}.nrrd"

    # Load with nibabel (supports .nii.gz and .mgz)
    img = nib.load(str(in_path))
    data = img.get_fdata(dtype=np.float32)  # float32 for intensities; ok for masks too
    # Note: nibabel data is (X,Y,Z) in RAS after as_closest_canonical
    img_can = nib.as_closest_canonical(img)
    aff = img_can.affine

    spacing, origin, direction = _affine_to_sitk(aff)

    # SimpleITK expects (Z,Y,X) numpy when using GetImageFromArray (unless we set geometry after)
    # We’ll set geometry explicitly, so the array axis order is fine; sitk will map from geometry.
    sitk_img = sitk.GetImageFromArray(np.asarray(img_can.get_fdata(dtype=np.float32)))
    sitk_img.SetSpacing(spacing[::-1])  # spacing per axis order of the array (Z,Y,X) vs (X,Y,Z)
    sitk_img.SetOrigin(origin)  # origin is in physical space (X,Y,Z)
    # Direction needs to match the array axis order; flip to Z,Y,X:
    # We formed direction in (X,Y,Z); for a quick and robust route, let’s just rely on spacing+origin,
    # and leave direction as identity if needed. If you want strict orientation, uncomment below:
    # sitk_img.SetDirection(direction)   # Use with care if axis conventions differ

    sitk.WriteImage(sitk_img, str(out_path))
    return out_path


NumberArg = Optional[Union[int, List[int]]]


class IXIToTemplatePreprocessor:
    """
    Align IXI subjects (orig_nu, mask, aseg) to a template (T1, mask, segs).

    - Initialize with explicit roots (no environment detection).
    - Iterate subject file paths via get_subjects_paths(...).
    - Compute aligned tensors via get_subjects_aligned(...).
    - Access template paths via get_template_paths().

    Usage:
    -------
    pp = IXIToTemplatePreprocessor(
        ixi_root="[...path...]/data/IXI-T1_fastsurfer",
        template_root="[...path...]/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c",
        do_plot=False,
    )

    # 1) Just paths (iterator of dicts)
    for p in pp.get_subjects_paths(numbers=40):
        print(p["mri_dir"], p["image"].name, p["mask"].name, p["aseg"].name)
        break

    # 2) Aligned tensors for the first match
    source, target, seg_source, seg_target = pp.get_subjects_aligned(numbers=40, resize_factor=0.3, first_only=True)

    # 3) Iterate all aligned subjects
    # 3.a) Several subjects (list[int]) with progress bar and total
    for paths, src, tgt, sseg, tseg in pp.get_subjects_aligned(
        numbers=[2, 40, 63, 22], resize_factor=0.25, first_only=False, progress=True
    ):
        print(f"Subject: {paths['subject_dir'].name} → {src.shape}")

    # 3.b) All subjects with progress
    for paths, src, tgt, sseg, tseg in pp.get_subjects_aligned(
        numbers=None, resize_factor=0.25, first_only=False, progress=True, tqdm_kwargs={"leave": True}
    ):
        print("--")

    """

    def __init__(
            self,
            ixi_root: str | Path,
            template_root: str | Path,
            *,
            template_name: str = "mni_icbm152_t1_tal_nlin_asym_09c.nii",
            template_mask_name: str = "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
            template_segs_name: str = "mni_icbm152.auto_noCCseg.mgz",
            template_seg_path: str = "fastsurfer_seg/mri/",

            # IXI filenames inside each subject directory (or its /mri subdir)
            ixi_image_name: str = "orig_nu.mgz",
            ixi_segs_name: str = "aseg.auto_noCCseg.mgz",
            ixi_mask_name: str = "mask.mgz",
            ixi_mri_subdir: str = "mri",  # if present, files are under <subject>/mri/

            simplify_segs_fn: Callable[[np.ndarray], np.ndarray] = simplify_segs,
            do_plot: bool = False,
    ):
        self.ixi_root = Path(ixi_root)
        self.template_root = Path(template_root)

        # filenames
        self.template_name = template_name
        self.template_mask_name = template_mask_name
        self.template_segs_name = template_segs_name
        self.template_seg_path = template_seg_path

        self.ixi_image_name = ixi_image_name
        self.ixi_segs_name = ixi_segs_name
        self.ixi_mask_name = ixi_mask_name
        self.ixi_mri_subdir = ixi_mri_subdir

        self.simplify_segs_fn = simplify_segs_fn
        self.do_plot = do_plot

        # Load template once
        self._tpl_img = load_canonical(self.template_root / self.template_name)
        self._tpl_mask_img = load_canonical(self.template_root / self.template_mask_name)

        tpl_segs_full = self.template_root / self.template_seg_path / self.template_segs_name
        self._tpl_segs_img = load_canonical(tpl_segs_full)

        # Mask template intensities
        tpl = self._tpl_img.get_fdata().astype(np.float32)
        tpl_m = (self._tpl_mask_img.get_fdata() > 0.5)
        self._tpl_data_masked = np.where(tpl_m, tpl, 0.0).astype(np.float32)

        # target grid spec for resampling
        self._target_spec = (self._tpl_img.shape, self._tpl_img.affine)

    # ---------------------- paths APIs ----------------------
    def get_template_paths(self) -> Dict[str, Path]:
        """
        Return template paths as a dict:
        { 'root', 'image', 'mask', 'aseg' }
        """
        tpl_img = self.template_root / self.template_name
        tpl_mask = self.template_root / self.template_mask_name
        tpl_segs = self.template_root / self.template_seg_path / self.template_segs_name

        missing = [p for p in (tpl_img, tpl_mask, tpl_segs) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing template files:\n" + "\n".join(map(str, missing)))

        return {"root": self.template_root, "image": tpl_img, "mask": tpl_mask, "aseg": tpl_segs}

    def get_subjects_paths(
            self,
            numbers: NumberArg = None,
            *,
            require_all: bool = True,
    ) -> Iterator[Dict[str, Path]]:
        """
        Iterate over subjects and yield dicts of paths:
        {
          'subject_dir': <Path>,
          'mri_dir':     <Path>,
          'image':       <Path to orig_nu>,
          'mask':        <Path to mask.mgz>,
          'aseg':        <Path to aseg.auto_noCCseg.mgz>
        }

        numbers: int | list[int] | None
          - None: all IXI subjects
          - int: only that IXI number
          - list[int]: only those IXI numbers
        """
        # normalize numbers to a set (or None)
        if numbers is None:
            wanted = None
        elif isinstance(numbers, int):
            wanted = {numbers}
        else:
            wanted = set(numbers)

        for folder in sorted(p.name for p in self.ixi_root.iterdir() if p.is_dir()):
            n = _ixi_number_from_folder(folder)
            if n is None:
                continue
            if (wanted is not None) and (n not in wanted):
                continue

            subj_dir = self.ixi_root / folder
            mri_dir = subj_dir / self.ixi_mri_subdir
            mri_dir = mri_dir if mri_dir.exists() else subj_dir

            image = mri_dir / self.ixi_image_name
            mask = mri_dir / self.ixi_mask_name
            aseg = mri_dir / self.ixi_segs_name

            if require_all and not (image.exists() and mask.exists() and aseg.exists()):
                continue

            yield {
                "subject_dir": subj_dir,
                "mri_dir": mri_dir,
                "image": image,
                "mask": mask,
                "aseg": aseg,
            }

    # ---------------------- alignment API ----------------------
    def get_subjects_aligned(
            self,
            numbers: NumberArg = None,
            *,
            resize_factor: float = 1.0,
            first_only: bool = True,
            progress: bool = False,
            tqdm_kwargs: Optional[dict] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Iterator[Tuple[Dict[str, Path], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """
        Align subjects to template.

        If first_only=True:
            returns (subject_dir name, source, target, seg_source, seg_target)
        Else:
            yields (paths_dict, source, target, seg_source, seg_target) for each subject.

        numbers: int | list[int] | None
        progress: show tqdm progress bar (requires `tqdm` installed) when first_only=False
        """
        paths_list = list(self.get_subjects_paths(numbers, require_all=True))
        if not paths_list:
            raise FileNotFoundError(f"No matching subjects under {self.ixi_root} for numbers={numbers}")

        if first_only:
            return (paths_list[0]["subject_dir"].name,) + self._process_one(paths_list[0], resize_factor=resize_factor)

        # multi-subject: yield with optional tqdm progress
        iterator = paths_list
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    paths_list,
                    total=len(paths_list),
                    desc=f"Processing subjects :",
                    **(tqdm_kwargs or {})
                )
            except Exception:
                # tqdm not available; silently fall back
                pass

        def _gen():
            for paths in iterator:
                iterator.set_description(f"Processing subjects : {paths["subject_dir"].name}")
                yield (paths, *self._process_one(paths, resize_factor=resize_factor))

        return _gen()

    # ---------------------- internal: one subject ----------------------
    def _process_one(
            self,
            paths: Dict[str, Path],
            *,
            resize_factor: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load IXI
        ixi_img = load_canonical(paths["image"])
        ixi_mask = load_canonical(paths["mask"])
        ixi_seg = load_canonical(paths["aseg"])

        # Mask intensities in native space
        x = ixi_img.get_fdata().astype(np.float32)
        m = (ixi_mask.get_fdata() > 0.5)
        xM = np.where(m, x, 0.0).astype(np.float32)

        # Resample all to template grid
        x_img = resample_from_to(ixi_img.__class__(xM, ixi_img.affine, ixi_img.header), self._target_spec, order=3)
        s_lab = resample_from_to(ixi_seg, self._target_spec, order=0)
        m_lab = resample_from_to(ixi_mask, self._target_spec, order=0)

        # Simplify labels
        tpl_segs_np = self.simplify_segs_fn(self._tpl_segs_img.get_fdata())
        src_segs_np = self.simplify_segs_fn(s_lab.get_fdata())

        # Re-mask after resample
        m_tpl = (m_lab.get_fdata() > 0.5)
        x_tpl = np.where(m_tpl, x_img.get_fdata().astype(np.float32), 0.0).astype(np.float32)

        # To torch
        source = normalize(to_torch(x_tpl))
        target = normalize(to_torch(self._tpl_data_masked))
        seg_source = to_torch(src_segs_np.astype(np.float32))
        seg_target = to_torch(tpl_segs_np.astype(np.float32))

        # Resize if needed
        if resize_factor != 1.0:
            source = tb.resize_image(source, resize_factor)
            target = tb.resize_image(target, resize_factor)
            seg_source = tb.resize_image(seg_source, resize_factor, mode="nearest")
            seg_target = tb.resize_image(seg_target, resize_factor, mode="nearest")

        # quick sanity plot
        if self.do_plot:
            self._quick_plot(source, target, seg_source, seg_target, name=paths["subject_dir"].name)

        return source, target, seg_source, seg_target

    # ---------------------- quick figure ----------------------

    def _quick_plot(self, source, target, seg_source, seg_target, name=None):
        w = source.shape[-1] // 2
        fig, ax = plt.subplots(2, 2)
        fig.suptitle(name)
        ax[0, 0].imshow(source[0, 0, ..., w], cmap="gray")
        ax[0, 0].set_title("Source")

        ax[0, 1].imshow(target[0, 0, ..., w], cmap="gray")
        ax[0, 1].set_title("Target")

        ax[1, 0].imshow(seg_source[0, 0, ..., w], cmap="tab10", vmin=seg_source.min(), vmax=seg_source.max())
        ax[1, 0].set_title("Segment source")
        ax[1, 1].imshow(seg_target[0, 0, ..., w], cmap="tab10", vmin=seg_source.min(), vmax=seg_source.max())
        ax[1, 1].set_title("Segment target")

        plt.show()

    def _debug_plot(
            self,
            ixi_native: np.ndarray,
            tpl_masked: np.ndarray,
            ixi_on_tpl: np.ndarray,
            ixi_segs_on_tpl: np.ndarray,
            tpl_segs: np.ndarray,
            name: Optional[str] = None
    ):
        z = ixi_native.shape[-1] // 2
        fig, ax = plt.subplots(2, 3, figsize=(10, 7))
        fig.suptitle(name or "IXI→Template sanity", fontsize=12)

        ax[0, 0].imshow(ixi_native[..., z], cmap="gray")
        ax[0, 0].set_title("IXI native (masked)")

        ax[0, 1].imshow(tpl_masked[..., z], cmap="gray")
        ax[0, 1].set_title("Template (masked)")

        ax[0, 2].imshow(tpl_segs[..., z], cmap="tab20",
                        vmin=np.min(tpl_segs), vmax=np.max(tpl_segs))
        ax[0, 2].set_title("Template segs")

        ax[1, 0].imshow(ixi_on_tpl[..., z], cmap="gray")
        ax[1, 0].set_title("IXI on template")

        # Simple composite: average
        comp = 0.5 * (ixi_on_tpl[..., z] / (ixi_on_tpl.max() + 1e-8)) + \
               0.5 * (tpl_masked[..., z] / (tpl_masked.max() + 1e-8))
        ax[1, 1].imshow(comp, cmap="gray")
        ax[1, 1].set_title("Composite (IXI on tpl vs tpl)")

        ax[1, 2].imshow(ixi_segs_on_tpl[..., z], cmap="tab20",
                        vmin=np.min(ixi_segs_on_tpl), vmax=np.max(ixi_segs_on_tpl))
        ax[1, 2].set_title("IXI segs on template")

        for a in ax.ravel(): a.axis("off")
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------
# Start of the executing function:

def execute_rigid_along_metamorphosis(pp, subjects_numbers):
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
            numbers=subjects_numbers, resize_factor=RESIZE_FACTOR, first_only=False, progress=True,
            tqdm_kwargs={"leave": True}
    ):
        sigma = [1, 3, 7]
        sigma = [(s,) * 3 for s in sigma]
        gamma = .5
        rho = 1
        cost_cst = 1e3
        cst_field = 1
        cost_affine = 1
        cost_field = 1
        adam_dt_step_field = 1e-6
        adam_dt_step_affine = 1e-2
        integration_steps = 10

        if FLAG_DECOUPLED:
            affine_optim = mt.affine_decoupled_along_metamorphosis
            aff_kwargs = {"rotation": True, "scaling": True, "translation": True}
            rotation = True
            scaling = True
            translation = True
            affine = False
        else:
            affine_optim = mt.affine_along_metamorphosis
            aff_kwargs = {"affine": True}
            rotation = False
            scaling = False
            translation = True
            affine = True

        print(f"\nPatient : {paths["subject_dir"].name}")
        # 2) Rigid search
        # 2.a  Align barycenters
        source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
        id_grid = tb.make_regular_grid(source_b.shape[2:], dx_convention="2square")
        seg_target_b = tb.imgDeform(seg_target, (id_grid + trans_t), mode="nearest")
        seg_source_b = tb.imgDeform(seg_source, (id_grid + trans_s), mode="nearest")

        search = False
        # 2.b Intial exploration:
        kernelOperator = rk.DummyKernel()
        datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'), alpha=1)
        datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)

        mr = mt.rigid_along_metamorphosis(
            source_b, target_b, momenta_ini=0,
            kernelOperator=kernelOperator,
            rho=1,
            data_term=datacost,
            integration_steps=integration_steps,
            cost_cst=.1,
        )
        top_params = rg.initial_exploration(mr, r_step=10, max_output=15, verbose=True)
        print(top_params)

        # 2.c Optimize on best finds
        best_loss, best_momenta, best_rot = rg.optimize_on_rigid(mr, top_params, n_iter=5, verbose=False)
        print(f"\nPatient : {paths["subject_dir"].name}")
        print("best_momenta = ", best_momenta)
        search = True

        # 3) [Optionnal] Check rigid search
        # rot_def = mr.mp.get_rigidor()
        # rotated_source = tb.imgDeform(mr.mp.image,rot_def,dx_convention='2square')
        # img = rotated_source[0,0,..., mr.source.shape[-1]//2].detach().cpu()
        # img_target = tb.imCmp(rotated_source[..., source.shape[-1]//2].detach().cpu(), mr.target[..., source.shape[-1]//2].detach().cpu(), "compose")[0]
        # img_source = tb.imCmp(rotated_source[..., source.shape[-1]//2].detach().cpu(), mr.source[..., source.shape[-1]//2].detach().cpu(), "compose")[0]
        # fig,ax = plt.subplots(1,3)
        # ax[0].imshow(img, cmap="gray")
        # ax[0].set_title("Final image")
        # ax[1].imshow(img_target, cmap="gray")
        # ax[1].set_title("img vs target")
        # ax[2].imshow(img_source, cmap="gray")
        # ax[2].set_title("img vs source")
        # fig.suptitle(f"rigid search {paths["subject_dir"].name}, {best_loss}")
        # if location == "meso":
        #     fig.savefig(os.path.join(result_folder, f'checkrigid_{paths["subject_dir"].name}.png'))
        # else:
        #     plt.show()

        # 4) Apply LDDMM
        # for cost_cst in [1e5, 5e5, 1e6]:
        #     for cst_field in [1e-2, 5e-2, 1e-1 ]:
        kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)

        # D(I,T) =  alpha *| S \cdot A.T  - T |^2 + (1 - alpha) * | I_1 \cdot A.T - T|^2
        datacost = mt.Rotation_Ssd_Cost(
            target_b.to("cuda:0"),
            gamma=gamma,
            # sigmoid_a=sigmoid_a,sigmoid_b=sigmoid_b,sigmoid_c=sigmoid_c,
            normalize_ssd=False,
            verbose=True,
            plot=False,
            save_plot=None
        )

        momenta = mt.prepare_momenta(
            source_b.shape,
            diffeo=True,
            device="cuda:0",
            **aff_kwargs,
            # **best_priors
        )

        mr = affine_optim(
            source_b.to("cuda:0"), target_b.to("cuda:0"), momenta_ini=momenta,
            kernelOperator=kernelOperator,
            rho=rho,
            data_term=datacost,
            integration_steps=integration_steps,
            cost_cst=cost_cst,
            cost_affine_cst=cost_affine,
            cost_field_cst=cost_field,
            n_iter=100,
            save_gpu_memory=False,
            optimizer_method='Adam',
            adam_dt_step_field=adam_dt_step_field,
            adam_dt_step_affine=adam_dt_step_affine,
            # lbfgs_max_iter = 10,
            # lbfgs_history_size = 30,
            # hamiltonian_integration=True
        )

        name = 'affine_lddmm' if not FLAG_DECOUPLED else "decoupled_lddmm"
        dices, _ = mr.compute_DICE(seg_source_b, seg_target_b, verbose=True)
        file_save, path = mr.save(f"{paths["subject_dir"].name}_{name}",
                                  light_save=False,
                                  save_path=os.path.join(result_folder, name)
                                  )
        mt.free_GPU_memory(mr)

        def _strf_(valbool):
            return "T" if valbool else "F"

        modifier_str = (
                "_r" + _strf_(rotation) +
                "_s" + _strf_(scaling) +
                "_t" + _strf_(translation)
        ) if not affine else "aT"

        dice = dices[0] | dices[1]
        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=paths["subject_dir"].name,
            method=name + modifier_str,
            metrics={name + ' ' + k: v for k, v in dice.items()},
            run_id=str(now) + ' at ' + location,
            step=0,
            meta={
                "gpu": torch.cuda.get_device_name(),
                "gamma": gamma,
                "rho": rho,
                "cost_cst": cost_cst,
                "cst_field": cst_field,
                "sigma": sigma,
                "integration_steps": integration_steps,
                "diffeo": True,
                "rotation": rotation,
                "scaling": scaling,
                "translation": translation,
                "affine": affine,
                "prelim_search": search,
                "adam_dt_step_field": adam_dt_step_field,
                "adam_dt_step_affine": adam_dt_step_affine,
                "file": os.path.join(path, file_save)
            }
        )


def execute_affine_along_metamorphosis_succLddmm(pp, subjects_numbers):
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
            numbers=subjects_numbers, resize_factor=RESIZE_FACTOR, first_only=False, progress=True,
            tqdm_kwargs={"leave": True}
    ):
        sigma = [3, 7]
        sigma = [(s,) * 3 for s in sigma]
        gamma = .5
        rho = 1
        cost_cst = 1e3
        cst_field = 1
        cost_affine = 1
        cost_field = 1
        adam_dt_step_field = 1e-6
        adam_dt_step_affine = 1e-2
        integration_steps = 10

        if FLAG_DECOUPLED:
            affine_optim = mt.affine_decoupled_along_metamorphosis
            aff_kwargs = {"rotation": True, "scaling": True, "translation": True}
            rotation = True
            scaling = True
            translation = True
            affine = False
        else:
            affine_optim = mt.affine_along_metamorphosis
            aff_kwargs = {"affine": True}
            rotation = False
            scaling = False
            translation = True
            affine = True

        print(f"\nPatient : {paths["subject_dir"].name}")
        method_name = 'affine_lddmm_succ' if not FLAG_DECOUPLED else "decoupled_lddmm_succ"

        # 4) Apply LDDMM
        # for cost_cst in [1e5, 5e5, 1e6]:
        #     for cst_field in [1e-2, 5e-2, 1e-1 ]:
        kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)

        # D(I,T) =  alpha *| S \cdot A.T  - T |^2 + (1 - alpha) * | I_1 \cdot A.T - T|^2
        gamma_kwargs = {'c': 1.0, 'nu': 1e-2} # 1
        gamma_kwargs = {'c': 1.0, 'nu': 1e-3} # 2
        gamma_kwargs = {'c': 1.0, 'nu': 1e-1} # 3
        gamma_kwargs = {'c': 2.0, 'nu': 1e-1} # 4
        gamma_kwargs = {'c': 2.0, 'nu': 1e-2} # 5
        gamma_kwargs = {'c': 2.0, 'nu': 1e-3} # 6

        datacost = mt.Rotation_Ssd_Cost(
            target.to("cuda:0"),
            gamma_mode="variationnal",
            # gamma_mode="sigmoid",
            gamma_kwargs=gamma_kwargs,
            # sigmoid_a=sigmoid_a,sigmoid_b=sigmoid_b,sigmoid_c=sigmoid_c,
            normalize_ssd=False,
            verbose=True,
            plot=True,
            save_plot=os.path.join(result_folder, method_name, paths["subject_dir"].name + str(gamma_kwargs)),
        )


        momenta = mt.prepare_momenta(
            source.shape,
            diffeo=True,
            device="cuda:0",
            **aff_kwargs,
            # **best_priors
        )

        mr_d = affine_optim(
            source.to("cuda:0"), target.to("cuda:0"), momenta_ini=momenta,
            kernelOperator=kernelOperator,
            rho=rho,
            data_term=datacost,
            integration_steps=integration_steps,
            cost_cst=cost_cst,
            cost_affine_cst=cost_affine,
            cost_field_cst=cost_field,
            n_iter=500 if location != "local" else 20,
            convergence_tol=1e-4,
            convergence_patience=3,
            save_gpu_memory=False,
            optimizer_method='Adam',
            adam_dt_step_field=adam_dt_step_field,
            adam_dt_step_affine=adam_dt_step_affine,
            adam_scheduler="reduce_on_plateau",
            # lbfgs_max_iter = 10,
            # lbfgs_history_size = 30,
            # hamiltonian_integration=True
        )
        fig, ax = mr_d.data_term.plot_cost_data_term()


        dices, _ = mr_d.compute_DICE(seg_source, seg_target, verbose=True)
        mt.free_GPU_memory(mr_d)

        deformator = mr_d.mp.get_affine_deformator()
        source_2 = tb.imgDeform(
            source, deformator,
            dx_convention=mr_d.dx_convention,
            # mode = 'nearest'
        ).to(device)
        source_seg_rotated = tb.imgDeform(
            seg_source, deformator,
            dx_convention=mr_d.dx_convention,
            mode='nearest'
        ).to(device)
        # tb.average_dice(source_seg_rotated.cpu(),seg_target,message="(affine p only)",verbose=True)

        print("\n>> Starting Part 2 - Classical LDDMM")
        sigma = [(3, 3, 3), (7, 7, 7)]
        kernel_op = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)
        # data_cost = mt.Mutual_Information(target)
        data_cost = mt.Ssd(target)
        mr2 = mt.lddmm(source_2, target, 0, kernel_op,
                       cost_cst=.001,
                       grad_coef=1,
                       integration_steps=integration_steps,  # avant c'était 7
                       n_iter=60 if location != "local" else 2,
                       lbfgs_history_size=15,
                       convergence_tol=1e-3,
                        convergence_patience=3,
                       data_term=data_cost,
                       )
        dices2, _ = mr2.compute_DICE(source_seg_rotated.to("cpu"), seg_target, verbose=True)


        file_save1, path = mr_d.save(f"{paths["subject_dir"].name}_{method_name}_part1",
                                     light_save=False,
                                     # save_path=os.path.join(result_folder, method_name, paths["subject_dir"].name),
                                     save_path=os.path.join(result_folder, method_name)
                                     )
        # ic(file_save1, path,path +"/"+ file_save1[:-4] + "_gamma_cost.png")
        ax.set_title(file_save1 +"\n"+ str(gamma_kwargs))
        fig.savefig(path +"/" +file_save1[:-4] + "_gamma_cost.png")
        if location != "local":
            plt.show()
        file_save2, path = mr2.save(f"{paths["subject_dir"].name}_{method_name}_part2",
                                    light_save=False,
                                    save_path=os.path.join(result_folder, method_name)
                                    )

        def _strf_(valbool):
            return "T" if valbool else "F"

        modifier_str = (
                "_r" + _strf_(rotation) +
                "_s" + _strf_(scaling) +
                "_t" + _strf_(translation)
        ) if not affine else "aT"
        modifier_str += "-lddmm2-tol"
        dice = dices[0] | dices[1] | { '(succ) ' + k:v for k,v in dices2.items()}
        ic(dice,dices, dices2)
        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=paths["subject_dir"].name,
            method=method_name + modifier_str,
            metrics={method_name + ' ' + k: v for k, v in dice.items()},
            run_id=str(now) + ' at ' + location,
            step=0,
            meta={
                "gpu": torch.cuda.get_device_name(),
                "gamma": gamma_kwargs,
                "rho": rho,
                "cost_cst": cost_cst,
                "cst_field": cst_field,
                "sigma": sigma,
                "integration_steps": integration_steps,
                "diffeo": True,
                "rotation": rotation,
                "scaling": scaling,
                "translation": translation,
                "affine": affine,
                "prelim_search": "no",
                "adam_dt_step_field": adam_dt_step_field,
                "adam_dt_step_affine": adam_dt_step_affine,
                "file": [os.path.join(path, file_save1), os.path.join(path, file_save2)]
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FA-LDDMM 5-arm freeze ablation — brain (IXI)
# ─────────────────────────────────────────────────────────────────────────────
#  Arms:
#    1  joint FA-LDDMM, no intervention              (baseline)
#    2  + freeze affine at variational γ hand-off τ
#    3  + freeze affine, reset diffeo Adam state at τ
#    4  + freeze affine, reset diffeo Adam state + zero diffeo values at τ
#    5  two-stage: affine-only → fresh LDDMM         (reference)
#
#  Arms 1–4 share identical hyperparams; only the freeze/reset flags differ.
#  Use execute_all_ablation_arms() to run the full ablation; it extracts τ from
#  arm 1 and caps arm 5's LDDMM stage at max(100, N_ITER − τ) iterations.
# ─────────────────────────────────────────────────────────────────────────────

def _ablation_hyperparams():
    """Shared hyperparameter dict for the 5-arm brain ablation."""
    sigma = [(s,) * 3 for s in [3, 7]]
    return dict(
        sigma                 = sigma,
        gamma_kwargs          = {"c": 1.0, "nu": 100},
        rho                   = 1,
        cost_cst              = 1e3,
        cost_affine_cst       = 1,
        cost_field_cst        = 1,
        adam_dt_step_field    = 1e-6,
        adam_dt_step_affine   = 1e-2,
        integration_steps     = 10,
        n_iter                = 500,
        convergence_tol       = 1e-4,
        convergence_patience  = 3,
    )


def _run_joint_arm_subject(
    source, target, seg_source, seg_target, paths,
    freeze_affine=False,
    reset_diffeo_state=False,
    reset_diffeo_values=False,
    method_name="fa_arm1_baseline",
):
    """Run one subject for arms 1–4; save + log metrics; return τ."""
    hp = _ablation_hyperparams()
    n_iter = hp["n_iter"] if location != "local" else 20

    kernelOperator = rk.Multi_scale_GaussianRKHS(hp["sigma"], normalized=False)
    datacost = mt.Rotation_Ssd_Cost(
        target.to("cuda:0"),
        gamma_mode="variationnal",
        gamma_kwargs=hp["gamma_kwargs"],
        normalize_ssd=False,
        verbose=False,
        save_values=True,
        edges_computes=1e-2,
    )
    momenta = mt.prepare_momenta(
        source.shape, diffeo=True, device="cuda:0", affine=True,
    )

    mr = mt.affine_along_metamorphosis(
        source.to("cuda:0"), target.to("cuda:0"),
        momenta_ini=momenta,
        kernelOperator=kernelOperator,
        rho=hp["rho"],
        data_term=datacost,
        integration_steps=hp["integration_steps"],
        cost_cst=hp["cost_cst"],
        cost_affine_cst=hp["cost_affine_cst"],
        cost_field_cst=hp["cost_field_cst"],
        n_iter=n_iter,
        convergence_tol=hp["convergence_tol"],
        convergence_patience=hp["convergence_patience"],
        save_gpu_memory=False,
        optimizer_method="Adam",
        adam_dt_step_field=hp["adam_dt_step_field"],
        adam_dt_step_affine=hp["adam_dt_step_affine"],
        adam_scheduler="reduce_on_plateau",
        freeze_affine_at_handoff=freeze_affine,
        reset_diffeo_state_at_handoff=reset_diffeo_state,
        reset_diffeo_values_at_handoff=reset_diffeo_values,
        log_affine_drift=True,
    )

    tau = getattr(mr, "_tau_iter", None)
    dices, _ = mr.compute_DICE(seg_source, seg_target, verbose=True)
    file_save, path = mr.save(
        f"{paths['subject_dir'].name}_{method_name}",
        light_save=False,
        save_path=os.path.join(result_folder, method_name),
    )

    dice = dices[0] | dices[1]
    now = datetime.datetime.now()
    log_metrics(
        db_path,
        patient_id=paths["subject_dir"].name,
        method=method_name,
        metrics={method_name + " " + k: v for k, v in dice.items()},
        run_id=str(now) + " at " + location,
        step=0,
        meta={
            "gpu": torch.cuda.get_device_name(),
            "gamma_kwargs": hp["gamma_kwargs"],
            "rho": hp["rho"],
            "cost_cst": hp["cost_cst"],
            "sigma": hp["sigma"],
            "integration_steps": hp["integration_steps"],
            "freeze_affine_at_handoff": freeze_affine,
            "reset_diffeo_state_at_handoff": reset_diffeo_state,
            "reset_diffeo_values_at_handoff": reset_diffeo_values,
            "tau": tau,
            "adam_dt_step_field": hp["adam_dt_step_field"],
            "adam_dt_step_affine": hp["adam_dt_step_affine"],
            "file": os.path.join(path, file_save),
        },
    )
    mt.free_GPU_memory(mr)
    return tau


def execute_arm1_joint_baseline(pp, subjects_numbers):
    """Arm 1 — joint FA-LDDMM, no intervention (baseline)."""
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        _run_joint_arm_subject(
            source, target, seg_source, seg_target, paths,
            freeze_affine=False,
            reset_diffeo_state=False,
            reset_diffeo_values=False,
            method_name="fa_arm1_baseline",
        )


def execute_arm2_freeze_affine(pp, subjects_numbers):
    """Arm 2 — joint FA-LDDMM, freeze affine at τ."""
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        _run_joint_arm_subject(
            source, target, seg_source, seg_target, paths,
            freeze_affine=True,
            reset_diffeo_state=False,
            reset_diffeo_values=False,
            method_name="fa_arm2_freeze",
        )


def execute_arm3_freeze_reset_state(pp, subjects_numbers):
    """Arm 3 — joint FA-LDDMM, freeze affine + reset diffeo Adam state at τ."""
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        _run_joint_arm_subject(
            source, target, seg_source, seg_target, paths,
            freeze_affine=True,
            reset_diffeo_state=True,
            reset_diffeo_values=False,
            method_name="fa_arm3_freeze_reset_state",
        )


def execute_arm4_freeze_reset_all(pp, subjects_numbers):
    """Arm 4 — joint FA-LDDMM, freeze affine + reset diffeo Adam state + zero diffeo values at τ."""
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        _run_joint_arm_subject(
            source, target, seg_source, seg_target, paths,
            freeze_affine=True,
            reset_diffeo_state=True,
            reset_diffeo_values=True,
            method_name="fa_arm4_freeze_reset_all",
        )


def execute_arm5_two_stage(pp, subjects_numbers, lddmm_n_iter=None):
    """
    Arm 5 — two-stage reference: affine-only FA → fresh LDDMM.

    lddmm_n_iter: LDDMM stage iteration cap.
                  Pass max(100, N_ITER - tau) from arm 1 for a fair comparison.
                  Defaults to hp["n_iter"] if None.
    """
    hp = _ablation_hyperparams()
    n_iter_affine = hp["n_iter"] if location != "local" else 20
    n_iter_lddmm  = lddmm_n_iter if lddmm_n_iter is not None else n_iter_affine

    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        method_name = "fa_arm5_two_stage"

        # Stage 1: affine-only with variational γ (DummyKernel → no diffeo)
        print(f"  [arm5] stage 1: affine-only  (n_iter={n_iter_affine})")
        datacost_affine = mt.Rotation_Ssd_Cost(
            target.to("cuda:0"),
            gamma_mode="variationnal",
            gamma_kwargs=hp["gamma_kwargs"],
            normalize_ssd=False,
            verbose=False,
            save_values=True,
            edges_computes=1e-2,
        )
        momenta_affine = mt.prepare_momenta(
            source.shape, diffeo=False, device="cuda:0", affine=True,
        )
        mr_affine = mt.affine_along_metamorphosis(
            source.to("cuda:0"), target.to("cuda:0"),
            momenta_ini=momenta_affine,
            kernelOperator=rk.DummyKernel(),
            rho=hp["rho"],
            data_term=datacost_affine,
            integration_steps=hp["integration_steps"],
            cost_cst=hp["cost_cst"],
            cost_affine_cst=hp["cost_affine_cst"],
            n_iter=n_iter_affine,
            convergence_tol=hp["convergence_tol"],
            convergence_patience=hp["convergence_patience"],
            save_gpu_memory=False,
            optimizer_method="Adam",
            adam_dt_step_affine=hp["adam_dt_step_affine"],
            adam_scheduler="reduce_on_plateau",
        )
        dices_affine, _ = mr_affine.compute_DICE(seg_source, seg_target, verbose=True)

        # Warp source and segmentation by the learned affine
        affine_grid = mr_affine.mp.get_affine_deformator().cpu()
        source_warped = tb.imgDeform(
            source.cpu(), affine_grid, dx_convention="2square",
        ).to("cuda:0")
        seg_source_warped = tb.imgDeform(
            seg_source.cpu(), affine_grid, dx_convention="2square", mode="nearest",
        )

        # Stage 2: pure LDDMM on affine-warped source
        print(f"  [arm5] stage 2: pure LDDMM  (n_iter={n_iter_lddmm})")
        kernelOperator_lddmm = rk.Multi_scale_GaussianRKHS(hp["sigma"], normalized=False)
        data_cost_lddmm = mt.Ssd(target.to("cuda:0"))
        mr_lddmm = mt.lddmm(
            source_warped, target.to("cuda:0"),
            0,
            kernelOperator_lddmm,
            cost_cst=hp["cost_cst"],
            grad_coef=hp["adam_dt_step_field"],
            integration_steps=hp["integration_steps"],
            n_iter=n_iter_lddmm,
            convergence_tol=hp["convergence_tol"],
            convergence_patience=hp["convergence_patience"],
            optimizer_method="Adam",
            adam_scheduler="reduce_on_plateau",
            dx_convention="2square",
            safe_mode=True,
        )
        dices_lddmm, _ = mr_lddmm.compute_DICE(seg_source_warped, seg_target, verbose=True)

        file_save1, path = mr_affine.save(
            f"{paths['subject_dir'].name}_{method_name}_stage1",
            light_save=False,
            save_path=os.path.join(result_folder, method_name),
        )
        file_save2, _ = mr_lddmm.save(
            f"{paths['subject_dir'].name}_{method_name}_stage2",
            light_save=False,
            save_path=os.path.join(result_folder, method_name),
        )

        dice = (dices_affine[0] | dices_affine[1]
                | {"lddmm " + k: v for k, v in (dices_lddmm[0] | dices_lddmm[1]).items()})
        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=paths["subject_dir"].name,
            method=method_name,
            metrics={method_name + " " + k: v for k, v in dice.items()},
            run_id=str(now) + " at " + location,
            step=0,
            meta={
                "gpu": torch.cuda.get_device_name(),
                "gamma_kwargs": hp["gamma_kwargs"],
                "rho": hp["rho"],
                "cost_cst": hp["cost_cst"],
                "sigma": hp["sigma"],
                "integration_steps": hp["integration_steps"],
                "lddmm_n_iter": n_iter_lddmm,
                "adam_dt_step_field": hp["adam_dt_step_field"],
                "adam_dt_step_affine": hp["adam_dt_step_affine"],
                "file": [os.path.join(path, file_save1), os.path.join(path, file_save2)],
            },
        )
        mt.free_GPU_memory(mr_affine)
        mt.free_GPU_memory(mr_lddmm)


def execute_all_ablation_arms(pp, subjects_numbers):
    """
    Run the full 5-arm FA-LDDMM freeze ablation on brain data (arms 1 → 5).

    Arm 1's mean τ across subjects is used to cap arm 5's LDDMM stage at
    max(100, N_ITER − τ) iterations for a fair comparison.
    """
    hp = _ablation_hyperparams()
    n_iter = hp["n_iter"] if location != "local" else 20

    # Arm 1 — collect τ per subject for arm-5 capping
    tau_per_subject: dict = {}
    print("\n" + "=" * 60 + "\nArm 1 — joint baseline\n" + "=" * 60)
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR,
        first_only=False, progress=True, tqdm_kwargs={"leave": True},
    ):
        print(f"\nPatient : {paths['subject_dir'].name}")
        tau = _run_joint_arm_subject(
            source, target, seg_source, seg_target, paths,
            freeze_affine=False, reset_diffeo_state=False, reset_diffeo_values=False,
            method_name="fa_arm1_baseline",
        )
        tau_per_subject[paths["subject_dir"].name] = tau

    print("\n" + "=" * 60 + "\nArm 2 — freeze affine at τ\n" + "=" * 60)
    execute_arm2_freeze_affine(pp, subjects_numbers)

    print("\n" + "=" * 60 + "\nArm 3 — freeze affine + reset diffeo state\n" + "=" * 60)
    execute_arm3_freeze_reset_state(pp, subjects_numbers)

    print("\n" + "=" * 60 + "\nArm 4 — freeze affine + reset diffeo state + zero values\n" + "=" * 60)
    execute_arm4_freeze_reset_all(pp, subjects_numbers)

    # Derive lddmm_n_iter from the mean τ observed in arm 1
    tau_values = [t for t in tau_per_subject.values() if t is not None]
    tau_ref = int(round(sum(tau_values) / len(tau_values))) if tau_values else None
    lddmm_n_iter = max(100, n_iter - tau_ref) if tau_ref is not None else n_iter
    print(f"\n[arm5] mean τ from arm1 = {tau_ref}  →  LDDMM n_iter = {lddmm_n_iter}")

    print("\n" + "=" * 60 + f"\nArm 5 — two-stage reference  (LDDMM n_iter={lddmm_n_iter})\n" + "=" * 60)
    execute_arm5_two_stage(pp, subjects_numbers, lddmm_n_iter=lddmm_n_iter)


def execute_subcmd(cmd):
    print(">>> executing command:")
    for arg in cmd:
        print(f"  {arg}")
    try:
        result = subprocess.run(
            cmd,
            check=True,  # raises CalledProcessError if command fails
            capture_output=True,  # capture stdout & stderr
            text=True  # decode as str instead of bytes
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        # return result
    except subprocess.CalledProcessError as e:
        print("Error running register")
        print("Return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        # return e


# ====================================================
#     Begin unigradicon
def itk_to_torch(image: "itk.Image[itk.F,3]", seg=False) -> torch.Tensor:
    """
    Convert an itk.ImageF3 (3D float image) to a PyTorch tensor.

    Parameters
    ----------
    image : itk.Image[itk.F,3]
        3D ITK image of floats.

    Returns
    -------
    torch.Tensor
        Tensor with shape (D, H, W) in float32.
    """
    # Step 1: ITK → NumPy (ITK gives array in z,y,x order)
    np_array = itk.GetArrayFromImage(image)  # shape = (z, y, x)
    if seg:
        np_array = simplify_segs(np_array)

    # Step 2: NumPy → Torch
    tensor = torch.from_numpy(np_array.astype("float32"))
    return tensor


def _evaluate_unigradicon(transform_file, fixed_seg, moving_seg, fixed_img, moving_img, plot):
    fixed_seg = itk.imread(ensure_nrrd(fixed_seg))
    moving_seg = itk.imread(ensure_nrrd(moving_seg))
    transform = itk.transformread(transform_file)[0]

    import numpy as np
    min_max_filter = itk.MinimumMaximumImageCalculator.New(fixed_seg)
    min_max_filter.Compute()
    ic("Min:", min_max_filter.GetMinimum())
    ic("Max:", min_max_filter.GetMaximum())
    ic(np.unique(itk.array_from_image(fixed_seg)))

    dispfield_filter = itk.TransformToDisplacementFieldFilter[itk.Image[itk.Vector[itk.F, 3], 3], itk.D].New()

    fixed_image = itk.imread(ensure_nrrd(fixed_img))
    dispfield_filter.SetTransform(transform)
    dispfield_filter.SetReferenceImage(fixed_image)
    dispfield_filter.SetUseReferenceImage(True)

    dispfield_filter.Update()

    displacement_field = dispfield_filter.GetOutput()

    displacement_field.GetLargestPossibleRegion().GetSize()

    interpolator = itk.NearestNeighborInterpolateImageFunction[
        type(moving_seg), itk.D
    ].New()

    warped_moving_seg = itk.warp_image_filter(
        moving_seg,
        output_origin=fixed_seg.GetOrigin(),
        output_direction=fixed_seg.GetDirection(),
        output_spacing=fixed_seg.GetSpacing(),
        displacement_field=displacement_field,
        interpolator=interpolator
    )

    dice = tb.average_dice(
        itk_to_torch(fixed_seg, seg=True),
        itk_to_torch(warped_moving_seg, seg=True),
        verbose=True
    )

    if plot:
        moving_image = itk.imread(moving_img)
        warped_moving_image = itk.warp_image_filter(
            moving_image,
            output_origin=fixed_image.GetOrigin(),
            output_direction=fixed_image.GetDirection(),
            output_spacing=fixed_image.GetSpacing(),
            displacement_field=displacement_field)

        fig, ax = plt.subplots(2, 4)
        ax[0, 0].imshow(fixed_image[50], cmap='gray')
        ax[0, 0].set_title("fixed image")
        ax[0, 1].imshow(warped_moving_image[50], cmap='gray')
        ax[0, 1].set_title("warped moving image")
        ax[0, 2].imshow(moving_image[50], cmap='gray')
        ax[0, 2].set_title("moving image")
        ax[0, 3].imshow(itk.checker_board_image_filter(fixed_image, warped_moving_image)[50], cmap='gray')
        ax[1, 0].imshow(fixed_seg[50], cmap='tab10')
        ax[1, 0].set_title("fixed image")
        ax[1, 1].imshow(warped_moving_seg[50], cmap='tab10')
        ax[1, 1].set_title("warped moving image")
        ax[1, 2].imshow(moving_seg[50], cmap='tab10')
        ax[1, 2].set_title("moving image")
        ax[1, 3].imshow(itk.checker_board_image_filter(fixed_seg, warped_moving_seg)[50], cmap='tab10')
        plt.show()

    return dice


def execute_uniGradIcon(pp, subjects_numbers):
    temp_paths = pp.get_template_paths()
    ic(temp_paths)
    print(temp_paths["image"])

    output_folder = os.path.join(result_folder, "unigradicon")
    if subjects_numbers is None:
        lsn = len(list(pp.get_subjects_paths(subjects_numbers, require_all=True)))
    else:
        lsn = len(subjects_numbers)
    for i, p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniGradIcon on Subject {i + 1} on {lsn}]:")
        output_name = f"uGI_{os.path.basename(p["subject_dir"])}_to_template"
        print("output_name :", output_name)

        fixed = ensure_nrrd(temp_paths["image"])
        moving = ensure_nrrd(p["image"])
        transform_out = os.path.join(output_folder, output_name + '.hdf5')
        if not RECOMPUTE and not os.path.exists(transform_out):

            cmd = [
                "unigradicon-register",
                f"--fixed={fixed}",
                f"--fixed_modality=mri",
                f"--fixed_segmentation={ensure_nrrd(temp_paths["mask"])}",
                f"--moving={moving}",
                f"--moving_modality=mri",
                f"--moving_segmentation={ensure_nrrd(p["mask"])}",
                f"--transform_out={transform_out}",
                f"--warped_moving_out={os.path.join(output_folder, output_name + '.nii.gz')}",
                # f"--io_iterations None",
            ]
            print(cmd)
            execute_subcmd(cmd)

        else:
            print(f"File exists, computation skipped : {transform_out}")

        # aseg_out = os.path.join(output_folder, output_name + '_aseg.nii.gz' )
        # cmd_wrap = [
        #     "unigradicon-warp",
        #     f"--fixed {fixed}",
        #     f"--moving {ensure_nrrd(p["aseg"])}",
        #     f"--transform {transform_out}",
        #     f"--warped_moving_out {aseg_out}",
        #     "--nearest_neighbor"
        # ]
        # execute_subcmd

        dice = _evaluate_unigradicon(transform_out,
                                     fixed_seg=ensure_nrrd(temp_paths["aseg"]),
                                     moving_seg=ensure_nrrd(p["aseg"]),
                                     fixed_img=fixed,
                                     moving_img=moving,
                                     plot=True
                                     )
        now = datetime.datetime.now()
        # Example per patient/method
        # log_metrics(
        #     db_path,
        #     patient_id=p["subject_dir"].name,
        #     method="unigradicon",
        #     metrics={'unigradicon ' + k: v for k,v in dice.items()},
        #     run_id= str(now) + ' at ' + location,
        #     step=0,
        #     meta={"gpu":torch.cuda.get_device_name()}
        # )


def execute_uniCarl(pp, subjects_numbers):
    temp_paths = pp.get_template_paths()
    ic(temp_paths)
    print(temp_paths["image"])

    output_folder = os.path.join(result_folder, "unicarl")
    if subjects_numbers is None:
        lsn = len(list(pp.get_subjects_paths(subjects_numbers, require_all=True)))
    else:
        lsn = len(subjects_numbers)
    for i, p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniCarl on Subject {i + 1} on {lsn}]:")
        output_name = f"unicarl_{os.path.basename(p["subject_dir"])}_to_template"
        print("output_name :", output_name)

        fixed = ensure_nrrd(temp_paths["image"])
        moving = ensure_nrrd(p["image"])
        transform_out = os.path.join(output_folder, output_name + '.hdf5')
        if not RECOMPUTE and not os.path.exists(transform_out):
            cmd = [
                "unicarl-register",
                f"--fixed={fixed}",
                # f"--fixed_modality=mri",
                # f"--fixed_segmentation={ensure_nrrd(temp_paths["mask"])}",
                f"--moving={moving}",
                # f"--moving_modality=mri",
                # f"--moving_segmentation={ensure_nrrd(p["mask"])}",
                f"--transform_out={transform_out}",
                f"--warped_moving_out={os.path.join(output_folder, output_name + '.nii.gz')}",
                # f"--io_iterations None",
            ]
            print(cmd)
            execute_subcmd(cmd)

        else:
            print(f"File exists, computation skipped : {transform_out}")

        # aseg_out = os.path.join(output_folder, output_name + '_aseg.nii.gz' )
        # cmd_wrap = [
        #     "unigradicon-warp",
        #     f"--fixed {fixed}",
        #     f"--moving {ensure_nrrd(p["aseg"])}",
        #     f"--transform {transform_out}",
        #     f"--warped_moving_out {aseg_out}",
        #     "--nearest_neighbor"
        # ]
        # execute_subcmd

        dice = _evaluate_unicarl(transform_out,
                                 fixed_seg=ensure_nrrd(temp_paths["aseg"]),
                                 moving_seg=ensure_nrrd(p["aseg"]),
                                 fixed_img=fixed,
                                 moving_img=moving,
                                 plot=False
                                 )
        now = datetime.datetime.now()
        print(dice)
        # Example per patient/method
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method="unicarl",
            metrics={'unicarl ' + k: v for k,v in dice.items()},
            run_id= str(now) + ' at ' + location,
            step=0,
            meta={"gpu":torch.cuda.get_device_name()}
        )

def apply_transform(moving, fixed, transform, use_nearest_neighbor=True):
    """
    moving : itk image to warp
    fixed  : itk image that defines the output grid (size, spacing, origin, direction)
    """
    interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving) \
                   if use_nearest_neighbor \
                   else itk.LinearInterpolateImageFunction.New(moving)

    resampler = itk.ResampleImageFilter.New(
        Input=moving,
        Transform=transform,
        Interpolator=interpolator,
        UseReferenceImage=True,
        ReferenceImage=fixed,    # output grid = fixed image grid
        DefaultPixelValue=0,
    )
    resampler.Update()
    return resampler.GetOutput()


def _evaluate_unicarl(transform_out,
                      fixed_seg,
                      moving_seg,
                      fixed_img,
                      moving_img,
                      plot=False
                      ):
    fixed_seg = itk.imread(ensure_nrrd(fixed_seg))
    moving_seg = itk.imread(ensure_nrrd(moving_seg))
    transform = itk.transformread(transform_out)[0]

    # ── Build affine-only composite ──────────────────────────────────────────────
    affine_transform = itk.CompositeTransform[itk.D, 3].New()
    affine_transform.PrependTransform(transform.GetNthTransform(3))
    affine_transform.PrependTransform(transform.GetNthTransform(1))
    affine_transform.PrependTransform(transform.GetNthTransform(0))


    # ── Apply affine-only transform ───────────────────────────────────────────────
    moving_seg_affine = apply_transform(moving_seg, fixed_seg, affine_transform)

    # ── Apply full composite transform (affine + displacement field) ──────────────
    moving_seg_full = apply_transform(moving_seg, fixed_seg, transform)

    dice_a = tb.average_dice(
                itk_to_torch(fixed_seg, seg = True),
                itk_to_torch(moving_seg_affine, seg = True),
                message = "(affine only)",
                verbose = True
            )
    dice_d = tb.average_dice(
            itk_to_torch(fixed_seg, seg = True),
            itk_to_torch(moving_seg_full, seg = True),
            verbose = True
            )

    return dice_a | dice_d


# end unigradicon
# ===================================================

# ===================================================
# Begin flirt + lddmm

def compute_flirt(moving, fixed, output, interp="trilinear"):
    # if output.endswith('nii.gz'):
    #     output_mat = output[:-7] + ".mat"
    # else:
    #     output_mat = output + ".mat"
    #     output = output + ".nii.gz"
    if output.suffixes == [".nii", ".gz"]:
        output_mat = output.with_suffix('').with_suffix('.mat')
    else:
        output_mat = output.with_suffix('.mat')
    print("output_mat :", output_mat)

    cmd = [
        "flirt",
        "-in", moving,
        "-ref", fixed,
        "-out", output,
        "-omat", output_mat,
        "-bins", "256",
        "-cost", "corratio",
        "-searchrx", "-90", "90",
        "-searchry", "-90", "90",
        "-searchrz", "-90", "90",
        "-dof", "12",
        "-interp", interp
    ]
    execute_subcmd(cmd)
    return output_mat


def apply_affine_mat_fsl(input_nii, reference_nii, transform_mat, output_nii, interp='nearestneighbour'):
    """
    Applies an affine transformation to a NIfTI image using FSL's flirt.

    Parameters:
        input_nii (str): Path to the input NIfTI file.
        reference_nii (str): Path to the reference NIfTI file (defines output space).
        transform_mat (str): Path to the .mat affine transformation file.
        output_nii (str): Path to save the transformed output NIfTI file.
        interp (str, optional): Interpolation method ('trilinear', 'nearestneighbour', 'spline'). Default is 'trilinear'.

    Returns:
        bool: True if successful, False otherwise.
    """

    cmd = [
        "flirt",
        "-in", ensure_nifti(input_nii),
        "-ref", reference_nii,
        "-applyxfm",
        "-init", transform_mat,
        "-out", output_nii,
        "-interp", interp
    ]
    execute_subcmd(cmd)


def mask_mri(im_dict):
    output_mask = im_dict["image"].with_name(im_dict["image"].stem + '_masked' + '.nii.gz')
    print(f"Applying masks to produce : {output_mask}")

    cmd_mask_ixi = [
        "fslmaths",
        ensure_nifti(im_dict["image"]),
        "-mas",
        ensure_nifti(im_dict["mask"]),
        output_mask
    ]
    execute_subcmd(cmd_mask_ixi)
    return output_mask


def open_nib_to_torch(image, seg: bool, resize_factor):
    img = load_canonical(image).get_fdata()

    if seg:
        img = to_torch(simplify_segs(img))
        mode = "nearest"
    else:
        img = normalize(to_torch(img.astype(np.float32)))
        mode = "bilinear"

    if resize_factor != 1.0:
        img = tb.resize_image(img, resize_factor, mode=mode)
    return img


def execute_flirt_lddmm(pp, subjects_numbers):
    if subjects_numbers is None:
        lsn = len(list(pp.get_subjects_paths(subjects_numbers, require_all=True)))
    else:
        lsn = len(subjects_numbers)
    for i, p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[flirt + lddmm on Subject {p["subject_dir"].name} : {i + 1} on {lsn}]:")

        temp_paths = pp.get_template_paths()
        rigid_ixi = p["image"].with_name(f"flirt_img_to_template.nii.gz")
        rigid_seg = p["aseg"].with_name(f"flirt_aseg_to_template.nii.gz")

        if not RECOMPUTE and not os.path.exists(rigid_ixi):
            ixi_masked = mask_mri(p)
            temp_masked = mask_mri(temp_paths)

            output_mat = compute_flirt(ixi_masked, temp_masked, rigid_ixi)
            apply_affine_mat_fsl(p["aseg"], temp_masked, output_mat, rigid_seg)
        else:
            print(f"Rigid registration found, skipping computation : {rigid_ixi}")
            temp_masked = temp_paths["image"].with_name(temp_paths["image"].stem + '_masked' + '.nii.gz')

        # # load images
        source = open_nib_to_torch(rigid_ixi, seg=False, resize_factor=RESIZE_FACTOR)
        target = open_nib_to_torch(temp_masked, seg=False, resize_factor=RESIZE_FACTOR)
        source_seg = open_nib_to_torch(rigid_seg, seg=True, resize_factor=RESIZE_FACTOR)
        target_seg = open_nib_to_torch(temp_paths["aseg"], seg=True, resize_factor=RESIZE_FACTOR)
        print("image shape : ", source.shape)
        source = source.to(device)
        target = target.to(device)

        dice_flirt = tb.average_dice(source_seg, target_seg, "(rigid only)", verbose=True)

        sigma = [(3, 3, 3), (7, 7, 7)]
        kernel_op = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)
        # data_cost = mt.Mutual_Information(target)
        data_cost = mt.Ssd(target)
        mr = mt.lddmm(source, target, 0, kernel_op,
                      cost_cst=.001,
                      grad_coef=1,
                      integration_steps=7,
                      n_iter=20,
                      lbfgs_history_size=15,
                      data_term=data_cost,
                      )
        # source_seg_def= tb.imgDeform(target_seg, mr.mp.get_deformator(), dx_convention=mr.dx_convention) # TODO: Probablement pas target_seg ici ....
        dice_lddmm, _ = mr.compute_DICE(source_seg, target_seg)
        # mr.save(f"{p["subject_dir"].name}_flirt_lddmm",
        #         light_save=True,
        #         save_path=os.path.join(result_folder, "flirt_lddmm")
        #         )
        dice = dice_flirt | dice_lddmm
        mt.free_GPU_memory(mr)

        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method=f"flirt_lddmm",
            metrics={f'flirt_lddmm ' + k: v for k, v in dice.items()},
            run_id=str(now) + ' at ' + location,
            step=0,
            meta={"gpu": torch.cuda.get_device_name(),
                  "data_cost": mr.data_term.__class__.__name__,
                  "sigma": sigma,
                  "RESIZE FACTOR": RESIZE_FACTOR,
                  }
        )


#                   end flirt + lddmm
# ====================================================

def execute_control(pp, subjects_numbers):
    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
            numbers=subjects_numbers, resize_factor=1, first_only=False, progress=True, tqdm_kwargs={"leave": True}
    ):
        print(paths["subject_dir"].name)
        dice = tb.average_dice(seg_source, seg_target, "before reg", verbose=True)

        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=paths["subject_dir"].name,
            method="before reg",
            metrics=dice,
            run_id=str(now) + ' at ' + location,
            step=0,
            meta={"gpu": torch.cuda.get_device_name()}
        )


# %%
def execute_dummy(pp, subjects_numbers):
    import random, string
    if subjects_numbers is None:
        lsn = paths_list = list(pp.get_subjects_paths(subjects_numbers, require_all=True))
    else:
        lsn = len(subjects_numbers)
    for i, p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniGradIcon on Subject {i + 1} on {lsn}]:")
        now = datetime.datetime.now()
        dice = {''.join(random.choices(string.ascii_uppercase, k=4)): random.random() for _ in range(7)}
        # Example per patient/method
        print("Execute dummy on ", p["subject_dir"].name)
        run_id = str(now) + ' at ' + location
        print("\t", run_id)
        metric = {'dummy ' + k: v for k, v in dice.items()}
        print("\tmetrics :", metric)
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method="dummy",
            metrics=metric,
            run_id=run_id,
            step=0,
            meta={"gpu": torch.cuda.get_device_name()}
        )


@contextmanager
def get_conn(db_path):
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency & durability
    conn.execute("PRAGMA synchronous=NORMAL;")  # good balance safety/speed
    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        patient_id TEXT NOT NULL,
        method     TEXT NOT NULL,
        metric     TEXT NOT NULL,
        value      REAL,
        run_id     TEXT NOT NULL,
        step       INTEGER DEFAULT 0,
        meta_json  TEXT,               -- optional: store shapes, seeds, params
        ts         REAL NOT NULL,      -- time.time()
        PRIMARY KEY (patient_id, method, metric, run_id, step)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_ts ON results(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_patient ON results(patient_id);")
    try:
        yield conn
    finally:
        conn.close()


def clean_method(db_name, method_name):
    with sqlite3.connect(db_name) as conn:
        conn.execute("DELETE FROM results WHERE method = ?", (method_name,))
        conn.commit()


def log_metrics(db_path, patient_id, method, metrics: dict, run_id, step=0, meta: dict = None):
    """
    metrics: {"dice": 0.91, "hausdorff95": 3.2, ...}
    meta:    {"gpu_mem": 3.1, "seed": 42, "shape": [160,192,160]}  (optional)
    """
    ts = time.time()
    meta_json = json.dumps(meta) if meta else None
    with get_conn(db_path) as conn:
        # UPSERT (idempotent if you re-run)
        conn.executemany("""
        INSERT INTO results(patient_id, method, metric, value, run_id, step, meta_json, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id, method, metric, run_id, step) DO UPDATE SET
            value=excluded.value,
            meta_json=COALESCE(excluded.meta_json, results.meta_json),
            ts=excluded.ts;
        """, [
            (patient_id, method, k, float(v), run_id, step, meta_json, ts)
            for k, v in metrics.items()
        ])


if __name__ == '__main__':
    # %%
    import subprocess

    cwd = subprocess.check_output("pwd", text=True).strip()
    if "content" in cwd:
        template_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        ixi_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        template_seg_path = ""
        location = "colab"
        result_folder = "/content/drive/MyDrive/demeter_data/ixi_results/"
    elif "gpfs" in cwd:
        template_folder = "/gpfs/workdir/francoisa/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/gpfs/workdir/francoisa/data/IXI-T1_fastsurfer/"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/gpfs/workdir/francoisa/data/IXI_results/"
        location = 'meso'
        # OPTIM_SAVE_DIR = "/gpfs/workdir/francoisa/saved_optim/"
    elif "afrancois" in cwd:
        template_folder = "/home/afrancois/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/home/afrancois/data/IXI-T1_fastsurfer"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/home/afrancois/data/IXI_results/"
        location = 'spark'
    else:
        template_folder = "/home/turtlefox/Documents/11_metamorphoses/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI-T1_fastsurfer"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/"
        location = 'local'
    device = "cuda:0"

    pp = IXIToTemplatePreprocessor(
        ixi_root=ixi_folder,
        template_root=template_folder,
        template_seg_path=template_seg_path,
        do_plot=False,
    )
    # jgnrz = list(pp.get_subjects_paths(None, require_all=True))
    # print(len(jgnrz))
    # for i in range(29):
    #     sublist = [_ixi_number_from_folder(str(jgnrz[j]['subject_dir'].name)) for j in range(i*20, i*20 + 20)]
    #     print(sublist)
    #     # print(_ixi_number_from_folder(str(j['subject_dir'].name)),',')

    #%%
    subjects_numbers = [2,12,13,14,15,16,17,19] # 1
    # subjects_numbers = [20,21,22,23,24,25,26,27,28,29] # 2
    # subjects_numbers = [30,31,33,34,35,36,37,38,39] # 3
    # subjects_numbers = [40,41,42,43,44,45,46,48,49] # 4
    # subjects_numbers = [50,51,52,53,54,55,56,57,58,59] # 5
    # subjects_numbers = [60,61,62,63,64,65,66,67,68,69] # 6
    # subjects_numbers = [35, 37, 61, 66, 34, 49]

    # all
    #1
    # subjects_numbers = [2, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
	# 	34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54,
	# 	55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
	# 	75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]

    # 2
#     subjects_numbers = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,107, 108, 109, 110, 111, 112, 113, 114, 115,
# 		116, 117, 118, 119, 120, 121, 122, 123, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138,
# 		139, 140, 141, 142, 143, 144, 145, 146, 148, 150, 151, 153, 154, 156, 157, 158, 159, 160, 161, 162,
# 		163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
# 		184, 185, 186, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206]
#
#     #3
#     subjects_numbers = [207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228,
# 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 244, 246, 247, 248, 249, 250, 251,
# 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 274,
# 275, 276, 277, 278, 279, 280, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
# 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317]
# #
# #     #4
#     subjects_numbers = [318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
# 340, 341, 342, 344, 345, 347, 348, 350, 351, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364,
# 365, 367, 368, 369, 370, 371, 372, 373, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386,
# 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406]
# #
# #     #5
#     subjects_numbers = [407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 424, 425, 426, 427,
# 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
# 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 467, 468,
# 469, 470, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490,
# 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511]
# #
# #     #6
#     subjects_numbers = [512, 515, 516, 517, 518, 519, 521, 522, 523, 524, 525, 526, 527, 528, 531, 532, 533, 534, 535, 536,
# 537, 538, 539, 541, 542, 543, 544, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 558, 559,
# 560, 561, 562, 563, 565, 566, 567, 568, 569, 571, 572, 573, 574, 575, 576, 577, 578, 579, 582, 584,
# 585, 586, 587, 588, 589, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 603, 605, 606, 607,
# 608, 609, 610, 611, 612, 613, 614, 616, 617, 618, 619, 621, 622, 623, 625, 626, 627, 629, 630, 631,
# 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 648, 651, 652, 653, 661, 662]

    # subjects_numbers =[560, 561, 562, 563, 565, 566, 567, 568, 569, 571,
    #                     572, 573, 574, 575, 576, 577, 578, 579, 582, 584,
    #                     585, 586, 587, 588, 589, 591, 592, 593, 594, 595,
    #                     596, 597, 598, 599, 600, 601, 603, 605, 606, 607,
    #                     608, 609, 610, 611, 612, 613, 614, 616, 617, 618,
    #                     619, 621, 622, 623, 625, 626, 627, 629, 630, 631,
    #                     632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
    #                     642, 643, 644, 646, 648, 651, 652, 653, 661, 662]


    # = [35,36,37,38,39,41,42,43] Done
    # [44,45,46,48,49,50,51,52,53,54, Done
    # 55,56,57,58,59,60,61,62, Done
    #     # subjects_numbers = [63,64,65,66,67,68,69]
    # subjects_numbers = None
    # subjects_numbers = [2]#, 26, 50,2, 12]
    RECOMPUTE = False
    RESIZE_FACTOR = .5 if location == 'local' else 1
    FLAG_DECOUPLED = False

    # init_csv(result_folder)

    if location == "meso":  # don't touch this line
        file_db = "ixi_results_arms_2026.db"
    else:  # here you can sandbox what you need to do.
        # file_db = f"ixi_results_{location}.db"
        file_db = f"ixi_results_arms_{location}.db"
        # file_db = "ixi_results_meso_20250917.db"
    db_path = os.path.join(result_folder, file_db)
    # clean_method(db_path, "affine_lddmm_succ")

    # execute_dummy(pp, subjects_numbers)
    # execute_control(pp,subjects_numbers)
    # if location == 'meso':
    # execute_uniGradIcon(pp, subjects_numbers)
    # execute_uniCarl(pp, subjects_numbers)
    # execute_flirt_lddmm(pp, subjects_numbers)
    # elif location == 'local':
    # execute_rigid_along_metamorphosis(pp, subjects_numbers)
    # execute_affine_along_metamorphosis_succLddmm(pp, subjects_numbers)

    # ── 5-arm freeze ablation (Phase C) ──────────────────────────────────────
    execute_all_ablation_arms(pp, subjects_numbers)   # full ablation 1→5
    # execute_arm1_joint_baseline(pp, subjects_numbers)
    # execute_arm2_freeze_affine(pp, subjects_numbers)
    # execute_arm3_freeze_reset_state(pp, subjects_numbers)
    # execute_arm4_freeze_reset_all(pp, subjects_numbers)
    # execute_arm5_two_stage(pp, subjects_numbers)
#