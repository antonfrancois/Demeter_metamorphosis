import os, re
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Iterator, Optional, Callable, Tuple, List, Dict
from pathlib import Path
from nibabel.processing import resample_from_to

import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
import demeter.utils.axes3dsliders_plt as a3s
import demeter.utils.rigid_exploration as rg

def to_torch(img):
    return torch.from_numpy(img)[None, None]

def normalize(img):
    quant = np.quantile(img, 0.99)
    print(quant, img.max())
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

class IXIToTemplatePreprocessor:
    """
    End-to-end loader & resampler from IXI subject to a template space.

    On __call__, returns (source, target, seg_source, seg_target) as torch 5D tensors
    shaped (1,1,D,H,W), optionally resized and normalized (intensities only).

    Usage:
    -------
    # Instantiate (auto-detects Colab vs local from cwd)
    pp = IXIToTemplatePreprocessor(do_plot=True)

    # 1) Single subject by number → returns the four tensors
    source, target, seg_source, seg_target = pp(number=2, resize_factor=0.3, first_only=True, name="IXI002_to_template")

    # 2) Iterate over *all* subjects
    for (src, tgt, sseg, tseg) in pp(number=None, resize_factor=0.25, first_only=False):
        print(src.shape, tgt.shape)
    """

    def __init__(
        self,
        cwd: str | Path | None = None,
        # Template roots (auto-switch if "content" in cwd):
        colab_template_root: str | Path = "/content/drive/MyDrive/demeter_data/ixi-T1/",
        local_template_root: str | Path = "/home/turtlefox/Documents/11_metamorphoses/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c",
        # IXI roots:
        colab_ixi_root: str | Path = "/content/drive/MyDrive/demeter_data/ixi-T1/",
        local_ixi_root: str | Path = "/home/turtlefox/Documents/11_metamorphoses/data/IXI-T1_fastsurfer",
        local_ixi_mri_subdir: str = "mri",  # FastSurfer layout: <subject>/mri/
        # Template file names:
        template_name: str = "mni_icbm152_t1_tal_nlin_asym_09c.nii",
        template_mask_name: str = "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
        template_segs_name: str = "mni_icbm152.auto_noCCseg.mgz",
        # Template segs optional subpath (e.g. "fastsurfer_seg/mri/"):
        template_seg_path: str = "fastsurfer_seg/mri/",
        # IXI filenames (inside each subject mri folder):
        ixi_image_name: str = "orig_nu.mgz",
        ixi_segs_name: str = "aseg.auto_noCCseg.mgz",
        ixi_mask_name: str = "mask.mgz",
        # Label simplifier:
        simplify_segs_fn: Callable[[np.ndarray], np.ndarray] = simplify_segs,
        # Plotting:
        do_plot: bool = False,
    ):
        self.cwd = str(cwd or os.getcwd())

        # Decide environment
        in_colab = "content" in self.cwd

        # Roots
        self.template_root = Path(colab_template_root if in_colab else local_template_root)
        self.ixi_root = Path(colab_ixi_root if in_colab else local_ixi_root)
        self.ixi_mri_subdir = local_ixi_mri_subdir

        # Filenames
        self.template_name = template_name
        self.template_mask_name = template_mask_name
        self.template_segs_name = template_segs_name
        self.template_seg_path = template_seg_path

        self.ixi_image_name = ixi_image_name
        self.ixi_segs_name = ixi_segs_name
        self.ixi_mask_name = ixi_mask_name

        self.simplify_segs_fn = simplify_segs_fn
        self.do_plot = do_plot

        # Load template triplet once
        self._tpl_img = load_canonical(self.template_root / self.template_name)
        self._tpl_msk_img = load_canonical(self.template_root / self.template_mask_name)

        tpl_seg_full = (self.template_root / self.template_seg_path / self.template_segs_name)
        self._tpl_segs_img = load_canonical(tpl_seg_full)

        # Precompute masked template intensities
        tpl_data = self._tpl_img.get_fdata()
        tpl_msk = (self._tpl_msk_img.get_fdata() > 0.5)
        self._tpl_data_masked = np.where(tpl_msk, tpl_data, 0.0).astype(np.float32)

        # Cache target spec
        self._target_spec = (self._tpl_img.shape, self._tpl_img.affine)

    # ---------- iteration over subjects ----------
    def iter_subjects(self, number: Optional[int] = None) -> Iterator[Path]:
        """
        Iterate over subject directories (IXI###-...).
        For local FastSurfer layout, the usable MRI path is <root>/<subject>/mri/.
        """
        for folder in find_ixi_folder(self.ixi_root, number):
            subj_dir = self.ixi_root / folder
            mri_dir = subj_dir / self.ixi_mri_subdir
            yield mri_dir if mri_dir.exists() else subj_dir  # fallback if no mri subdir

    def iter_processed(
        self,
        number: int | None = None,
        resize_factor: float = 1.0,
        name: str | None = None,
    ):
        """Generator: yields (source, target, seg_source, seg_target) for all matches."""
        for ixi_dir in self.iter_subjects(number):
            yield self.align_subject_with_affines(
                ixi_dir, resize_factor=resize_factor, name=name or ixi_dir.parent.name
            )

    def iter_subject_paths(
        self,
        number: Optional[int] = None,
        require_all: bool = True,
    ) -> Iterator[Dict[str, Path]]:
        """
        Yield dicts of paths for each matching subject:
        {
            'mri_dir': <Path to subject MRI dir>,
            'image':   <Path to intensity MRI (e.g., orig_nu.mgz)>,
            'mask':    <Path to skull-extraction mask (e.g., mask.mgz)>,
            'aseg':    <Path to FreeSurfer aseg (e.g., aseg.auto_noCCseg.mgz)>,
        }

        Args
        ----
        number : int | None
            IXI number to filter (e.g., 40 -> IXI040-...). If None, iterate all.
        require_all : bool
            If True, only yield subjects where all three files exist.
            If False, yield whatever exists (missing keys will still be present but may not exist()).
        """
        for mri_dir in self.iter_subjects(number):
            img  = mri_dir / self.ixi_image_name
            mask = mri_dir / self.ixi_mask_name
            aseg = mri_dir / self.ixi_segs_name

            if require_all and not (img.exists() and mask.exists() and aseg.exists()):
                # Skip incomplete subjects
                continue

            yield {
                "mri_dir": mri_dir,
                "image":   img,
                "mask":    mask,
                "aseg":    aseg,
            }

    def subject_paths(
        self,
        number: Optional[int] = None,
        first_only: bool = True,
        require_all: bool = True,
    ) -> Tuple[Path, Path, Path] | List[Tuple[Path, Path, Path]]:
        """
        Return paths to (image, skull mask, FreeSurfer aseg).

        Args
        ----
        number : int | None
            Filter by IXI number or iterate all if None.
        first_only : bool
            If True, return a single (image, mask, aseg) triple for the first matching subject.
            If False, return a list of triples for all matching subjects.
        require_all : bool
            Require all three files to exist before returning (skip otherwise).

        Raises
        ------
        FileNotFoundError
            If first_only=True and no matching subject is found.
        """
        it = self.iter_subject_paths(number=number, require_all=require_all)
        if first_only:
            item = next(it, None)
            if item is None:
                raise FileNotFoundError(
                    f"No subject with required files found for number={number} under {self.ixi_root}"
                )
            return item["image"], item["mask"], item["aseg"]
        else:
            return [(d["image"], d["mask"], d["aseg"]) for d in it]

    def template_paths(self) -> dict[str, Path]:
        """
        Return paths to the template image, mask, and segmentation.

        Returns
        -------
        dict
            {
                "image": Path to template image,
                "mask":  Path to template brain mask,
                "aseg":  Path to template segmentation,
                "root":  Path to template root folder
            }
        """
        tpl_img = self.template_root / self.template_name
        tpl_mask = self.template_root / self.template_mask_name
        tpl_segs = self.template_root / self.template_seg_path / self.template_segs_name

        missing = [p for p in [tpl_img, tpl_mask, tpl_segs] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing template files:\n" + "\n".join(str(m) for m in missing)
            )

        return {
            "root": self.template_root,
            "image": tpl_img,
            "mask": tpl_mask,
            "aseg": tpl_segs,
        }

    # ---------- core processing ----------
    def align_subject_with_affines(
        self,
        ixi_dir: Path,
        resize_factor: float = 1.0,
        name: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load IXI triplet, resample to template, simplify segs, remask, normalize/resize.

        Returns:
            source, target, seg_source, seg_target  (each as torch (1,1,D,H,W))
        """
        # Load IXI (canonical)
        ixi_img = load_canonical(ixi_dir / self.ixi_image_name)
        ixi_segs = load_canonical(ixi_dir / self.ixi_segs_name)
        ixi_mask = load_canonical(ixi_dir / self.ixi_mask_name)

        # Optional: mask IXI intensities in native space
        ixi_img_data = ixi_img.get_fdata().astype(np.float32)
        ixi_mask_data = (ixi_mask.get_fdata() > 0.5)
        ixi_img_data = np.where(ixi_mask_data, ixi_img_data, 0.0).astype(np.float32)
        ixi_img_masked = ixi_img.__class__(ixi_img_data, ixi_img.affine, ixi_img.header)

        # Resample to template grid
        ixi_to_tpl_img = resample_from_to(ixi_img_masked, self._target_spec, order=3)  # cubic for intensities
        ixi_to_tpl_segs = resample_from_to(ixi_segs,      self._target_spec, order=0)  # nearest for labels
        ixi_to_tpl_mask = resample_from_to(ixi_mask,      self._target_spec, order=0)  # nearest for mask

        # Simplify segmentations
        tpl_segs_data = self.simplify_segs_fn(self._tpl_segs_img.get_fdata())
        ixi_segs_data = self.simplify_segs_fn(ixi_to_tpl_segs.get_fdata())

        # Re-mask intensities after resampling
        tpl_mask_resampled = (ixi_to_tpl_mask.get_fdata() > 0.5)
        ixi_to_tpl_img_data = ixi_to_tpl_img.get_fdata().astype(np.float32)
        ixi_to_tpl_img_data = np.where(tpl_mask_resampled, ixi_to_tpl_img_data, 0.0).astype(np.float32)

        # Convert to torch
        source = normalize(to_torch(ixi_to_tpl_img_data))
        target = normalize(to_torch(self._tpl_data_masked))
        seg_source = to_torch(ixi_segs_data.astype(np.float32))
        seg_target = to_torch(tpl_segs_data.astype(np.float32))

        # Resize if requested
        if resize_factor != 1.0:
            source = tb.resize_image(source,     resize_factor)
            target = tb.resize_image(target,     resize_factor)
            seg_source = tb.resize_image(seg_source, resize_factor, mode="nearest")
            seg_target = tb.resize_image(seg_target, resize_factor, mode="nearest")

        # Optional quick figure (central slice)
        if self.do_plot:
            self._quick_plot(source, target, seg_source, seg_target, name=name)
            # self._quick_plot(ixi_img_data, self._tpl_data_masked, ixi_to_tpl_img_data, ixi_segs_data, tpl_segs_data, name=name)

        return source, target, seg_source, seg_target

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

    # ---------- “call” convenience ----------
    def get_aligned_sujects(
        self,
        number: int | None = None,
        resize_factor: float = 1.0,
        first_only: bool = True,
        name: str | None = None,
    ):
        """
        If first_only=True: return a 4-tuple.
        If first_only=False: return a **list** of 4-tuples (not a generator).
        """
        it = self.iter_subjects(number)
        if first_only:
            ixi_dir = next(it, None)
            if ixi_dir is None:
                raise FileNotFoundError(
                    f"No IXI subject found for number={number} under {self.ixi_root}"
                )
            return self.align_subject_with_affines(ixi_dir, resize_factor=resize_factor, name=name)

        # No `yield` here → __call__ is a normal function
        return [
            self.align_subject_with_affines(d, resize_factor=resize_factor, name=name or d.parent.name)
            for d in it
        ]


if __name__ == '__main__':
    import subprocess
    # cwd = subprocess.check_output("pwd", text=True).strip()
    # if "content" in cwd:
    #     template_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
    #     ixi_folder = "/content/drive/MyDrive/demeter_data/ixi-T1/"
    #     template_seg_path = ""
    # else:
    #     template_folder ="/home/turtlefox/Documents/11_metamorphoses/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c"
    #     ixi_folder = "/home/turtlefox/Documents/11_metamorphoses/data/IXI-T1_fastsurfer"
    #     template_seg_path = "fastsurfer_seg/mri/"

    pp = IXIToTemplatePreprocessor(do_plot=False)
    # 1) Get paths for IXI040
    img_p, mask_p, aseg_p = pp.subject_paths(number=40, first_only=True)
    print(img_p)
    print(mask_p)
    print(aseg_p)

    # 2) Iterate all available subjects, yielding dicts
    for d in pp.iter_subject_paths(number=None):
        print(d["mri_dir"].parent.name, "→", d["image"].name, d["mask"].name, d["aseg"].name)


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

    # 2) Rigid search
    # 2.a  Align barycenters
    # source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
    # id_grid = tb.make_regular_grid(source_b.shape[2:],dx_convention="2square")
    # seg_target_b = tb.imgDeform(seg_target, (id_grid + trans_t), mode="nearest")
    # seg_source_b = tb.imgDeform(seg_source, (id_grid + trans_s), mode="nearest")
    #
    # # 2.b Intial exploration:
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
    #     integration_steps = 10,
    #     cost_cst=.1,
    # )
    # top_params = rg.initial_exploration(mr,r_step=10, max_output = 15, verbose=True)
    # print(top_params)
    #
    # # 2.c Optimize on best finds
    # best_loss, best_momentum_R, best_momentum_T, best_momentum_S, best_rot = rg.optimize_on_rigid(mr, top_params, n_iter=10,verbose=True)

    # 3) [Optionnal] Check rigid search

    # 4) Apply LDDMM
