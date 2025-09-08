import os, re
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Iterator, Optional, Callable, Tuple, List, Dict, Union
from pathlib import Path
from nibabel.processing import resample_from_to

import SimpleITK as sitk

import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
import demeter.utils.rigid_exploration as rg


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
    if path.suffixes == [".nii", ".gz"] or path.suffix == [".nii"]:
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

# def load_raw_ixi(folder_path):
#     path_img = "mri/orig_nu.mgz"
#     ixi_segs_name = "mri/aseg.auto_noCCseg.mgz"      # discrete labels
#     ixi_mask_name = "mri/mask.mgz"                   # brain mask
#     ixi_img = load_canonical(os.path.join(folder_path, path_img))
#     ixi_segs = load_canonical(os.path.join(folder_path, ixi_segs_name))
#     ixi_mask = load_canonical(os.path.join(folder_path, ixi_mask_name))
#
#     print("IXI image shape:", ixi_img.shape)
#     print("IXI image affine:\n", ixi_img.affine)
#     print("IXI segs unique: ", np.unique( ixi_segs.get_fdata()))
#
#     # ---------- optional: mask IXI intensities in native space ----------
#     ixi_img_data = ixi_img.get_fdata()
#     ixi_mask_data = ixi_mask.get_fdata() > 0.5
#     ixi_img_data = np.where(ixi_mask_data, ixi_img_data, 0.0)
#     ixi_img_masked = ixi_img.__class__(ixi_img_data.astype(np.float32), ixi_img.affine, ixi_img.header)

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
            returns (source, target, seg_source, seg_target)
        Else:
            yields (paths_dict, source, target, seg_source, seg_target) for each subject.

        numbers: int | list[int] | None
        progress: show tqdm progress bar (requires `tqdm` installed) when first_only=False
        """
        paths_list = list(self.get_subjects_paths(numbers, require_all=True))
        if not paths_list:
            raise FileNotFoundError(f"No matching subjects under {self.ixi_root} for numbers={numbers}")

        if first_only:
            return self._process_one(paths_list[0], resize_factor=resize_factor)

        # multi-subject: yield with optional tqdm progress
        iterator = paths_list
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    paths_list,
                    total=len(paths_list),
                    desc="Processing subjects",
                    **(tqdm_kwargs or {})
                )
            except Exception:
                # tqdm not available; silently fall back
                pass

        def _gen():
            for paths in iterator:
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

def execute_rigid_along_metamorphosis(pp, subjects_numbers, resize_factor):

    for paths, source, target, seg_source, seg_target in pp.get_subjects_aligned(
        numbers=subjects_numbers, resize_factor=resize_factor, first_only=False, progress=True, tqdm_kwargs={"leave": True}
    ):
        # 2) Rigid search
        # 2.a  Align barycenters
        source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
        id_grid = tb.make_regular_grid(source_b.shape[2:],dx_convention="2square")
        seg_target_b = tb.imgDeform(seg_target, (id_grid + trans_t), mode="nearest")
        seg_source_b = tb.imgDeform(seg_source, (id_grid + trans_s), mode="nearest")

        # 2.b Intial exploration:
        kernelOperator = rk.GaussianRKHS(sigma=(15,15,15),normalized=False)
        datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'), alpha=1)
        # datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)


        mr = mt.rigid_along_metamorphosis(
            source_b, target_b, momenta_ini=0,
            kernelOperator= kernelOperator,
            rho = 1,
            data_term=datacost ,
            integration_steps = 10,
            cost_cst=.1,
        )
        top_params = rg.initial_exploration(mr,r_step=10, max_output = 15, verbose=True)
        print(top_params)

        # 2.c Optimize on best finds
        best_loss, best_momentum_R, best_momentum_T, best_momentum_S, best_rot = rg.optimize_on_rigid(mr, top_params, n_iter=10,verbose=True)

        # 3) [Optionnal] Check rigid search

        # 4) Apply LDDMM


def execute_uniGradIcon(pp, subjects_numbers):
    temp_paths  = pp.get_template_paths()
    ic(temp_paths)
    print(temp_paths["image"])

    output_folder = os.path.join(result_folder, "unigradicon")

    for i,p in enumerate(pp.get_subjects_paths(numbers=subjects_numbers)):
        print(f"\n[uniGradIcon on Subject {i+1} on {len(subjects_numbers)}]:")
        output_name = f"uGI_{os.path.basename(p["subject_dir"])}_to_template"
        print("output_name :", output_name)
        # unigradicon-register
        # --fixed= temp_paths["image"].name
        # --fixed_modality=mri
        # --fixed_segmentation=[temp_paths["mask"]]
        # --moving=p["image"]
        # --moving_modality=mri
        # --moving_segmentation=[p["mask"]]
        # --transform_out= os.path.join(output_folder, output_name + '.hdf5' )
        # --warped_moving_out= os.path.join(output_folder, output_name + '.nii.gz' )
        # --io_iterations None

        cmd = [
            "unigradicon-register",
            f"--fixed={ensure_nrrd(temp_paths["image"])}" ,
            f"--fixed_modality=mri",
            f"--fixed_segmentation={ensure_nrrd(temp_paths["mask"])}",
            f"--moving={ensure_nrrd(p["image"])}",
            f"--moving_modality=mri",
            f"--moving_segmentation={ensure_nrrd(p["mask"])}",
            f"--transform_out={os.path.join(output_folder, output_name + '.hdf5' )}",
            f"--warped_moving_out={os.path.join(output_folder, output_name + '.nii.gz' )}",
            # f"--io_iterations None",
        ]


        try:
            result = subprocess.run(
                cmd,
                check=True,          # raises CalledProcessError if command fails
                capture_output=True, # capture stdout & stderr
                text=True            # decode as str instead of bytes
            )
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Error running unigradicon-register")
            print("Return code:", e.returncode)
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)


if __name__ == '__main__':
    import subprocess
    cwd = subprocess.check_output("pwd", text=True).strip()
    if "content" in cwd:
        template_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        ixi_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
        template_seg_path = ""
    else:
        template_folder ="/home/turtlefox/Documents/11_metamorphoses/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
        ixi_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI-T1_fastsurfer"
        template_seg_path = "fastsurfer_seg/mri/"
        result_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/"


    pp = IXIToTemplatePreprocessor(
        ixi_root=ixi_folder,
        template_root=template_folder,
        template_seg_path=template_seg_path,
        do_plot=False,
    )
    subjects_numbers = [40]

    execute_uniGradIcon(pp, subjects_numbers)



    # # Pour meta+rigid
    # print("Available IXI dirs for number=40:")
    #
    # for (source, target, source_seg, tseg) in pp(number=None, resize_factor=0.25, first_only=False):
    #     print(src.shape, tgt.shape)
    #
    #
    #
    # # 1) Open images
    # source, target, seg_source, seg_target = pp(number=40, resize_factor=0.3, first_only=True, name="IXI002_to_template")
    # out =  pp(number=40, resize_factor=0.3, first_only=True, name="IXI002_to_template")
    # print(out)


