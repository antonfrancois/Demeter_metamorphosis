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
    img = np.clip(img, 0,quant)
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
        if path.with_suffix('.nii.gz').name in  os.listdir(path.parent):
            print("><", path.with_suffix('.nii.gz'))
            return path.with_suffix('.nii.gz')

        print(".mgz found and .nii.gz not found, converting to .nii.gz")
        out_path = path.with_suffix("")  # strip .mgz
        out_path = out_path.with_suffix(".nii.gz")
        img = nib.load(str(path))
        nib.save(img, str(out_path))
        return out_path

    raise ValueError(f"Unsupported file extension: {path.suffixes or path.suffix} in {path.name}")

def _affine_to_sitk(aff: np.ndarray) -> tuple[tuple[float,...], tuple[float,...], tuple[float,...]]:
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
    sitk_img.SetSpacing(spacing[::-1])   # spacing per axis order of the array (Z,Y,X) vs (X,Y,Z)
    sitk_img.SetOrigin(origin)           # origin is in physical space (X,Y,Z)
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
            mask  = mri_dir / self.ixi_mask_name
            aseg  = mri_dir / self.ixi_segs_name

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
        ixi_img  = load_canonical(paths["image"])
        ixi_mask = load_canonical(paths["mask"])
        ixi_seg  = load_canonical(paths["aseg"])

        # Mask intensities in native space
        x  = ixi_img.get_fdata().astype(np.float32)
        m  = (ixi_mask.get_fdata() > 0.5)
        xM = np.where(m, x, 0.0).astype(np.float32)

        # Resample all to template grid
        x_img = resample_from_to(ixi_img.__class__(xM, ixi_img.affine, ixi_img.header), self._target_spec, order=3)
        s_lab = resample_from_to(ixi_seg,  self._target_spec, order=0)
        m_lab = resample_from_to(ixi_mask, self._target_spec, order=0)

        # Simplify labels
        tpl_segs_np = self.simplify_segs_fn(self._tpl_segs_img.get_fdata())
        src_segs_np = self.simplify_segs_fn(s_lab.get_fdata())

        # Re-mask after resample
        m_tpl = (m_lab.get_fdata() > 0.5)
        x_tpl = np.where(m_tpl, x_img.get_fdata().astype(np.float32), 0.0).astype(np.float32)

        # To torch
        source     = normalize(to_torch(x_tpl))
        target     = normalize(to_torch(self._tpl_data_masked))
        seg_source = to_torch(src_segs_np.astype(np.float32))
        seg_target = to_torch(tpl_segs_np.astype(np.float32))

        # Resize if needed
        if resize_factor != 1.0:
            source     = tb.resize_image(source,     resize_factor)
            target     = tb.resize_image(target,     resize_factor)
            seg_source = tb.resize_image(seg_source, resize_factor, mode = "nearest")
            seg_target = tb.resize_image(seg_target, resize_factor, mode = "nearest")

        # quick sanity plot
        if self.do_plot:
            self._quick_plot(source, target, seg_source, seg_target, name=paths["subject_dir"].name)

        return source, target, seg_source, seg_target

    # ---------------------- quick figure ----------------------

    def _quick_plot(self, source, target, seg_source, seg_target, name=None):
        w = source.shape[-1]//2
        fig, ax = plt.subplots(2,2)
        fig.suptitle(name)
        ax[0,0].imshow(source[0,0,..., w], cmap="gray")
        ax[0,0].set_title("Source")

        ax[0,1].imshow(target[0,0,..., w], cmap="gray")
        ax[0,1].set_title("Target")

        ax[1,0].imshow(seg_source[0,0,..., w], cmap="tab10", vmin= seg_source.min(), vmax= seg_source.max())
        ax[1,0].set_title("Segment source")
        ax[1,1].imshow(seg_target[0,0,..., w], cmap="tab10", vmin= seg_source.min(), vmax= seg_source.max())
        ax[1,1].set_title("Segment target")


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

        ax[0,0].imshow(ixi_native[..., z], cmap="gray")
        ax[0,0].set_title("IXI native (masked)")

        ax[0,1].imshow(tpl_masked[..., z], cmap="gray")
        ax[0,1].set_title("Template (masked)")

        ax[0,2].imshow(tpl_segs[..., z], cmap="tab20",
                       vmin=np.min(tpl_segs), vmax=np.max(tpl_segs))
        ax[0,2].set_title("Template segs")

        ax[1,0].imshow(ixi_on_tpl[..., z], cmap="gray")
        ax[1,0].set_title("IXI on template")

        # Simple composite: average
        comp = 0.5 * (ixi_on_tpl[..., z] / (ixi_on_tpl.max() + 1e-8)) + \
               0.5 * (tpl_masked[..., z] / (tpl_masked.max() + 1e-8))
        ax[1,1].imshow(comp, cmap="gray")
        ax[1,1].set_title("Composite (IXI on tpl vs tpl)")

        ax[1,2].imshow(ixi_segs_on_tpl[..., z], cmap="tab20",
                       vmin=np.min(ixi_segs_on_tpl), vmax=np.max(ixi_segs_on_tpl))
        ax[1,2].set_title("IXI segs on template")

        for a in ax.ravel(): a.axis("off")
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------
# Start of the executing function:

def execute_rigid_along_metamorphosis(pp, subjects_numbers):

    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=RESIZE_FACTOR, first_only=False, progress=True, tqdm_kwargs={"leave": True}
    ):
        sigma= [1, 3,  7]
        sigma = [(s,)*3 for s in sigma]
        # alpha = .2
        rho = 1
        cost_cst = .1
        integration_steps = 10
        print(f"\nPatient : {paths["subject_dir"].name}")
        # 2) Rigid search
        # 2.a  Align barycenters
        source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
        id_grid = tb.make_regular_grid(source_b.shape[2:],dx_convention="2square")
        seg_target_b = tb.imgDeform(seg_target, (id_grid + trans_t), mode="nearest")
        seg_source_b = tb.imgDeform(seg_source, (id_grid + trans_s), mode="nearest")

        # 2.b Intial exploration:
        # kernelOperator = rk.GaussianRKHS(sigma=(15,15,15),normalized=False)
        # datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'), alpha=1)
        # # datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)
        #
        #
        # mr = mt.rigid_along_metamorphosis(
        #     source_b, target_b, momenta_ini=0,
        #     kernelOperator= kernelOperator,
        #     rho = 1,
        #     data_term=datacost ,
        #     integration_steps = integration_steps,
        #     cost_cst=.1,
        # )
        # top_params = rg.initial_exploration(mr,r_step=10, max_output = 15, verbose=False)
        # print(top_params)
        #
        # # 2.c Optimize on best finds
        # best_loss, best_momentum_R, best_momentum_T, best_momentum_S, best_rot = rg.optimize_on_rigid(mr, top_params, n_iter=10,verbose=False)
        # print("best_momentum_R = torch.",best_momentum_R)
        # print("best_momentum_T = torch.",best_momentum_T)
        # print("best_momentum_S = torch.",best_momentum_S)
        #
        # # 3) [Optionnal] Check rigid search
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
        for alpha in [.5, .6, .7, .8, .9, 1]:
            kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)

            # D(I,T) =  alpha *| S \cdot A.T  - T |^2 + (1 - alpha) * | I_1 \cdot A.T - T|^2
            # datacost = mt.Rotation_MutualInformation_Cost(target_b, alpha=.5)

            datacost = mt.Rotation_Ssd_Cost(target_b.to("cuda:0"), alpha=alpha)
            momenta = mt.prepare_momenta(
                source_b.shape,
                # rot_prior=best_momentum_R.detach().clone(),trans_prior=best_momentum_T.detach().clone(),
                # scale_prior=best_momentum_S.detach().clone(),
            )



            mr = mt.rigid_along_metamorphosis(
              source_b, target_b, momenta_ini=momenta,
              kernelOperator= kernelOperator,
              rho = rho,
              data_term=datacost ,
              integration_steps = integration_steps,
              cost_cst=cost_cst,
              n_iter=20,
              save_gpu_memory=False,
              lbfgs_max_iter = 20,
              lbfgs_history_size = 20,
            )

            dices, _ =mr.compute_DICE(seg_source_b, seg_target_b, verbose=True)
            file_save, path = mr.save(f"{paths["subject_dir"].name}_rigid_along_lddmm",
                    light_save=True,
                    save_path = os.path.join(result_folder, "rigid_along_lddmm")
                    )
            mt.free_GPU_memory(mr)
            dice = dices[0] | dices[1]
            now = datetime.datetime.now()
            log_metrics(
                db_path,
                patient_id=paths["subject_dir"].name,
                method="rigid_along_lddmm (no search)",
                metrics={'rigid_along_lddmm (no search) ' + k: v for k,v in dice.items()},
                run_id= str(now) + ' at ' + location,
                step=0,
                meta={"gpu":torch.cuda.get_device_name(),
                      "alpha" : alpha,
                      "rho" : rho,
                      "cost_cst" : cost_cst,
                      "sigma" : sigma,
                      "integration_steps" : integration_steps,
                      "file": os.path.join(path, file_save)
                      }
                )


def execute_subcmd(cmd):
    print(">>> executing command:")
    for arg in cmd:
        print(f"  {arg}")
    try:
        result = subprocess.run(
            cmd,
            check=True,          # raises CalledProcessError if command fails
            capture_output=True, # capture stdout & stderr
            text=True            # decode as str instead of bytes
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        # return result
    except subprocess.CalledProcessError as e:
        print("Error running unigradicon-register")
        print("Return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        # return e

# ====================================================
#     Begin unigradicon
def itk_to_torch(image: "itk.Image[itk.F,3]", seg = False) -> torch.Tensor:
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

def _evalutate_unigradicon(transform_file, fixed_seg, moving_seg, fixed_img, moving_img, plot):
    fixed_seg = itk.imread(ensure_nrrd(fixed_seg))
    moving_seg = itk.imread(ensure_nrrd(moving_seg))
    transform = itk.transformread(transform_file)[0]

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
            itk_to_torch(fixed_seg, seg = True),
            itk_to_torch(warped_moving_seg, seg = True),
            verbose = True
        )

    if plot:
        moving_image = itk.imread(moving_img)
        warped_moving_image = itk.warp_image_filter(
            moving_image,
            output_origin=fixed_image.GetOrigin(),
            output_direction=fixed_image.GetDirection(),
            output_spacing=fixed_image.GetSpacing(),
            displacement_field=displacement_field)

        fig, ax = plt.subplots(2,4)
        ax[0,0].imshow(fixed_image[50], cmap='gray')
        ax[0,0].set_title("fixed image")
        ax[0,1].imshow(warped_moving_image[50], cmap='gray')
        ax[0,1].set_title("warped moving image")
        ax[0,2].imshow(moving_image[50], cmap='gray')
        ax[0,2].set_title("moving image")
        ax[0,3].imshow(itk.checker_board_image_filter(fixed_image, warped_moving_image)[50], cmap='gray')
        ax[1,0].imshow(fixed_seg[50], cmap='tab10')
        ax[1,0].set_title("fixed image")
        ax[1,1].imshow(warped_moving_seg[50], cmap='tab10')
        ax[1,1].set_title("warped moving image")
        ax[1,2].imshow(moving_seg[50], cmap='tab10')
        ax[1,2].set_title("moving image")
        ax[1,3].imshow(itk.checker_board_image_filter(fixed_seg, warped_moving_seg)[50], cmap= 'tab10')
        plt.show()

    return dice


def execute_uniGradIcon(pp, subjects_numbers):
    temp_paths  = pp.get_template_paths()
    ic(temp_paths)
    print(temp_paths["image"])

    output_folder = os.path.join(result_folder, "unigradicon")
    if subjects_numbers is None:
        lsn =  len(list(pp.get_subjects_paths(subjects_numbers, require_all=True)))
    else:
        lsn = len(subjects_numbers)
    for i,p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniGradIcon on Subject {i+1} on {lsn}]:")
        output_name = f"uGI_{os.path.basename(p["subject_dir"])}_to_template"
        print("output_name :", output_name)

        fixed = ensure_nrrd(temp_paths["image"])
        moving = ensure_nrrd(p["image"])
        transform_out = os.path.join(output_folder, output_name + '.hdf5')
        if not RECOMPUTE and  not os.path.exists(transform_out):

            cmd = [
                "unigradicon-register",
                f"--fixed={fixed}" ,
                f"--fixed_modality=mri",
                f"--fixed_segmentation={ensure_nrrd(temp_paths["mask"])}",
                f"--moving={moving}" ,
                f"--moving_modality=mri",
                f"--moving_segmentation={ensure_nrrd(p["mask"])}",
                f"--transform_out={transform_out}",
                f"--warped_moving_out={os.path.join(output_folder, output_name + '.nii.gz' )}",
                # f"--io_iterations None",
            ]
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

        dice  = _evalutate_unigradicon(transform_out,
                               fixed_seg =  ensure_nrrd(temp_paths["aseg"]),
                               moving_seg= ensure_nrrd(p["aseg"]),
                               fixed_img=fixed,
                               moving_img=moving,
                               plot= False
            )
        now = datetime.datetime.now()
        # Example per patient/method
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method="unigradicon",
            metrics={'unigradicon ' + k: v for k,v in dice.items()},
            run_id= str(now) + ' at ' + location,
            step=0,
            meta={"gpu":torch.cuda.get_device_name()}
        )


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

def apply_affine_mat_fsl(input_nii, reference_nii, transform_mat, output_nii, interp='nearestneighbour' ):
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
        output_mask = im_dict["image"].with_name(im_dict["image"].stem +'_masked' + '.nii.gz')
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

def open_nib_to_torch(image, seg : bool, resize_factor):
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
    for i,p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[flirt + lddmm on Subject {p["subject_dir"].name} : {i+1} on {lsn}]:")

        temp_paths = pp.get_template_paths()
        rigid_ixi = p["image"].with_name(f"flirt_img_to_template.nii.gz")
        rigid_seg = p["aseg"].with_name(f"flirt_aseg_to_template.nii.gz")

        if not RECOMPUTE and  not os.path.exists(rigid_ixi):
            ixi_masked = mask_mri(p)
            temp_masked = mask_mri(temp_paths)

            output_mat = compute_flirt(ixi_masked, temp_masked, rigid_ixi)
            apply_affine_mat_fsl(p["aseg"], temp_masked, output_mat, rigid_seg)
        else:
            print(f"Rigid registration found, skipping computation : {rigid_ixi}")
            temp_masked = temp_paths["image"].with_name(temp_paths["image"].stem +'_masked' + '.nii.gz')

        # # load images
        source = open_nib_to_torch(rigid_ixi, seg = False, resize_factor = RESIZE_FACTOR)
        target = open_nib_to_torch(temp_masked, seg = False, resize_factor = RESIZE_FACTOR)
        source_seg = open_nib_to_torch(rigid_seg, seg = True,resize_factor = RESIZE_FACTOR)
        target_seg = open_nib_to_torch(temp_paths["aseg"], seg = True, resize_factor = RESIZE_FACTOR)
        print("image shape : ", source.shape)
        source = source.to(device)
        target =target.to(device)


        dice_flirt = tb.average_dice(source_seg, target_seg, "(rigid only)", verbose = True)

        sigma = [(3,3,3), (7,7,7)]
        kernel_op = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)
        # data_cost = mt.Mutual_Information(target)
        data_cost = mt.Ssd(target)
        mr = mt.lddmm(source, target, 0, kernel_op,
                 cost_cst=.001,
                grad_coef=1,
                 integration_steps=7,
                 n_iter= 20,
                lbfgs_history_size=15,
              data_term=data_cost,
        )
        dice_lddmm, _ = mr.compute_DICE(source_seg, target_seg)
        mr.save(f"{p["subject_dir"].name}_flirt_lddmm",
                light_save=True,
                save_path = os.path.join(result_folder, "flirt_lddmm")
                )
        dice = dice_flirt | dice_lddmm
        mt.free_GPU_memory(mr)

        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method="flirt_lddmm",
            metrics={'flirt_lddmm ' + k: v for k,v in dice.items()},
            run_id= str(now) + ' at ' + location,
            step=0,
            meta={"gpu":torch.cuda.get_device_name(),
                  "data_cost": mr.data_term.__class__.__name__,
                  "sigma":sigma}
        )

#                   end flirt + lddmm
# ====================================================

def execute_control(pp, subjects_numbers):

    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=1, first_only=False, progress=True, tqdm_kwargs={"leave": True}
    ):
        print(paths["subject_dir"].name)
        dice = tb.average_dice(seg_source, seg_target, "before reg", verbose = True)

        now = datetime.datetime.now()
        log_metrics(
            db_path,
            patient_id=paths["subject_dir"].name,
            method="before reg",
            metrics=dice,
            run_id= str(now) + ' at ' + location,
            step=0,
            meta={"gpu":torch.cuda.get_device_name()}
        )

#%%
def execute_dummy(pp, subjects_numbers):
    import random, string
    if subjects_numbers is None:
        lsn = paths_list = list(pp.get_subjects_paths(subjects_numbers, require_all=True))
    else:
        lsn = len(subjects_numbers)
    for i,p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniGradIcon on Subject {i+1} on {lsn}]:")
        now = datetime.datetime.now()
        dice = {''.join(random.choices(string.ascii_uppercase, k=4)) : random.random() for _ in range(7) }
        # Example per patient/method
        print("Execute dummy on ",p["subject_dir"].name)
        run_id =  str(now) + ' at ' + location
        print("\t",run_id)
        metric = {'dummy ' + k: v for k,v in dice.items()}
        print("\tmetrics :", metric)
        log_metrics(
            db_path,
            patient_id=p["subject_dir"].name,
            method="dummy",
            metrics=metric,
            run_id= run_id,
            step=0,
            meta={"gpu":torch.cuda.get_device_name()}
        )

@contextmanager
def get_conn(db_path):
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")       # better concurrency & durability
    conn.execute("PRAGMA synchronous=NORMAL;")     # good balance safety/speed
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



def log_metrics(db_path, patient_id, method, metrics: dict, run_id, step=0, meta: dict=None):
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
    #%%
    import subprocess
    cwd = subprocess.check_output("pwd", text=True).strip()
    if "content" in cwd:
        template_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        ixi_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        template_seg_path = ""
        location = "colab"
    elif "gpfs" in cwd:
        template_folder = "/gpfs/workdir/francoisa/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/gpfs/workdir/francoisa/data/IXI-T1_fastsurfer/"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/gpfs/workdir/francoisa/data/IXI_results/"
        location = 'meso'
        # OPTIM_SAVE_DIR = "/gpfs/workdir/francoisa/saved_optim/"
    else:
        template_folder ="/home/turtlefox/Documents/11_metamorphoses/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI-T1_fastsurfer"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/"
        location = 'local'
    device = "cuda:0"
    #%%
    pp = IXIToTemplatePreprocessor(
        ixi_root=ixi_folder,
        template_root=template_folder,
        template_seg_path=template_seg_path,
        do_plot=False,
    )



    subjects_numbers = [14,25,27,28,29,30,31,33,34]# Done
    # subjects_numbers = [30,31,33,34]# Done

    # = [35,36,37,38,39,41,42,43] Done
    # [44,45,46,48,49,50,51,52,53,54, Done
    # 55,56,57,58,59,60,61,62, Done
    #     # subjects_numbers = [63,64,65,66,67,68,69]
    # subjects_numbers = None
    # subjects_numbers = [2]#, 40, 26, 50,2, 12]
    RECOMPUTE = False
    RESIZE_FACTOR = 1 if location == 'meso' else .8

    # init_csv(result_folder)

    if location == "meso": # don't touch this line
        file_db = "ixi_results.db"
    else: # here you can sandbox what you need to do.
        # file_db = "ixi_results.db"
        file_db = "ixi_results_meso_20250917.db"
    db_path = os.path.join(result_folder, file_db)
    # clean_method(db_path, "rigid_along_lddmm")

    # execute_dummy(pp, subjects_numbers)
    # execute_control(pp,subjects_numbers)
    # if location == 'meso':
    # #     execute_uniGradIcon(pp, subjects_numbers)
    # execute_flirt_lddmm(pp, subjects_numbers)
    # elif location == 'local':
    execute_rigid_along_metamorphosis(pp, subjects_numbers)


