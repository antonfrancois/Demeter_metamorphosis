"""
.. _lddmm_on_MRIbrains:

 Applying LDDMM on MRI of brains
================================================

In this file we apply LDDMM on two MRI. One from the IXIbrain dataset to a sri template

"""
import torch

from demeter import ROOT_DIRECTORY
import nibabel as nib

import demeter.utils.torchbox as tb
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import matplotlib.pyplot as plt

from draft.Build_simplex_brain import data_cost

cuda = torch.cuda.is_available()
# cuda = False
device = 'cpu'
if cuda:
    device = 'cuda:0'
# Keep track of the backend so plots/logs show which device was actually used.
print('device used :',device)


import numpy as np

def _to_tensor(img):
    # Allow callers to pass either numpy arrays or tensors.
    return torch.from_numpy(img) if isinstance(img, np.ndarray) else img

def _ensure_hwc(t, num_channels):
    """
    Ensure slice is (H, W, C). If 2D and C>1, add channel axis.
    """
    if t.ndim == 2 and num_channels > 1:
        t = t.unsqueeze(-1)
    return t

def _pad_to_hw(hwc, H, W):
    """
    Pad a (H, W) or (H, W, C) tensor on height then width.
    Uses (C, H, W) to ensure we pad H and W (not the channel).
    """
    if hwc.ndim == 2:
        chw = hwc.unsqueeze(0)  # (1, H, W)
        C_axis = False
    elif hwc.ndim == 3:
        chw = hwc.permute(2, 0, 1)  # (C, H, W)
        C_axis = True
    else:
        raise ValueError("Expected (H, W) or (H, W, C) slice.")

    dH = max(0, H - chw.shape[1])
    dW = max(0, W - chw.shape[2])
    chw = torch.nn.functional.pad(chw, (0, dW, 0, dH), value=0)  # pad W then H

    if C_axis:
        return chw.permute(1, 2, 0)  # (H, W, C)
    else:
        return chw.squeeze(0)        # (H, W)

def get_orthogonal_views_concatenated(image, coords):
    """
    Returns axial, sagittal, and coronal views at (z, y, x) concatenated horizontally.
    Accepts [D, H, W] or [D, H, W, C] (C<=4). For color, keeps channels.
    """
    image = _to_tensor(image)

    # Normalize rank & channels
    if image.ndim == 3:
        # [D, H, W] -> treat as grayscale (C=1)
        image_to_slice = image.unsqueeze(-1)  # [D, H, W, 1]
        num_channels = 1
    elif image.ndim == 4:
        if image.shape[-1] <= 4:
            # [D, H, W, C]
            image_to_slice = image
            num_channels = image.shape[-1]
        else:
            raise ValueError("Expected [D,H,W] or [D,H,W,C] with C<=4.")
    else:
        raise ValueError("Input must be 3D or 4D.")

    D, H, W, C = image_to_slice.shape
    z, y, x = coords
    z = int(max(0, min(z, D - 1)))
    y = int(max(0, min(y, H - 1)))
    x = int(max(0, min(x, W - 1)))

    # Extract slices
    # Axial: [H, W, C]
    axial = image_to_slice[z, :, :, :].detach().cpu()
    axial = axial.squeeze(-1) if num_channels == 1 else axial
    axial = _ensure_hwc(axial, num_channels)

    # Sagittal: start [D, H, C] -> rotate 90° over (D,H) to get [H, D, C] ≡ (H, W_sag, C)
    sagittal = image_to_slice[:, :, x, :].detach().cpu()
    sagittal = sagittal.squeeze(-1) if num_channels == 1 else sagittal
    if sagittal.ndim == 2:  # (D, H) for grayscale
        sagittal = torch.rot90(sagittal, 1, [0, 1])              # -> (H, D)
    else:                    # (D, H, C)
        sagittal = torch.rot90(sagittal, 1, [0, 1])              # -> (H, D, C)
    sagittal = _ensure_hwc(sagittal, num_channels)

    # Coronal: [D, W, C] -> keep as (H_cor=W? nope) we keep height=D, width=W
    coronal = image_to_slice[:, y, :, :].detach().cpu()
    coronal = coronal.squeeze(-1) if num_channels == 1 else coronal
    coronal = _ensure_hwc(coronal, num_channels)  # currently (D, W) or (D, W, C); treat D as height

    # Target sizes
    ha, wa = axial.shape[:2]
    hs, ws = sagittal.shape[:2]
    hc, wc = coronal.shape[:2]
    Hmax = max(ha, hs, hc)
    Wmax = max(wa, ws, wc)

    # Pad to common (Hmax, Wmax)
    axial_p     = _pad_to_hw(axial,    Hmax, Wmax)
    sagittal_p  = _pad_to_hw(sagittal, Hmax, Wmax)
    coronal_p   = _pad_to_hw(coronal,  Hmax, Wmax)

    # Concat along width
    if num_channels == 1:
        # slices are (H, W) → stack into (H, 3*W)
        concatenated = torch.cat((axial_p, sagittal_p, coronal_p), dim=1)
    else:
        # slices are (H, W, C) → concat along W
        concatenated = torch.cat((axial_p, sagittal_p, coronal_p), dim=1)

    return concatenated


def slc_plt(img, coord):
  # if len(img.shape) == 5:
  #   return img[0,0,..., slice_index].detach().cpu()
  # elif len(img.shape) == 4:
  #   return img[..., slice_index,:].detach().cpu()
  if len(img.shape) == 5:
    # Batch with channel dimension (N,C,D,H,W) → take first item for display
    return get_orthogonal_views_concatenated(img[0,0], coord)

  elif len(img.shape) == 4:
    # Already (D,H,W,C) or (D,H,W) layout
    return get_orthogonal_views_concatenated(img, coord)

#######################################################
# 1. load files
# Fetch demo MRI volumes, coerce them to matching shapes, and do a quick sanity
# check/visualization (resize + basic normalization) before registration.


def open_nii(path):
    # Load a NIfTI file and return it as a torch tensor on the chosen device.
    nii = nib.load(path)
    nump = nib.as_closest_canonical(nii).get_fdata()
    return torch.from_numpy(nump).to(device)[None,None]

img_path = ROOT_DIRECTORY / "examples/im3Dbank/"
moving_p  = "IXI015-HH-1258-T1.nii.gz"
# fixed_p = "sri_spm_template_T1.nii"
fixed_p = "mni_icbm152_t1_tal_nlin_asym_09c_masked.nii.gz"

moving = open_nii(img_path / moving_p)
fixed = open_nii(img_path / fixed_p)

# Images may have differents shapes
ic(moving.shape, fixed.shape)

# For real applications,
to_shape = fixed.shape
# to_shape = (1,1,25,25,25)  # to test on cpu
if to_shape != moving.shape:
    fixed = tb.resize_image(fixed, to_shape = to_shape) # to test on cpu
    moving = tb.resize_image(moving, to_shape = fixed.shape)
ic(moving.shape, fixed.shape)
_,_,D,W,H = fixed.shape
coord = (D//2, H//2, W//3)

# Normalising images, here we do minimal normalisation
# you might want to do something more rafined
fixed = fixed /fixed.max()
moving = moving /moving.max()

cmp_img = tb.imCmp(moving, fixed, 'seg')[0]
fig, ax = plt.subplots(3,1)
# Quick visual inspection of the initial alignment
ax[0].imshow(slc_plt(moving, coord), cmap='gray')
ax[0].set_title('moving')
ax[1].imshow(slc_plt(fixed, coord), cmap='gray')
ax[1].set_title('fixed')
ax[2].imshow(slc_plt(cmp_img, coord))
ax[2].set_title('images compared')
plt.show()
##############################################
# 2. Define kernel
# Build a multi-scale Gaussian RKHS kernel that controls the deformation
# smoothness/scale used by the LDDMM optimizer.
#%%
ratios = [50, 30, 20]
sigmas = rk.get_sigma_from_img_ratio(moving.shape, ratios)

# sigma should be of the form [(s,s,s),(u,u,u),..]
kernelOp = rk.Multi_scale_GaussianRKHS(
    sigmas, normalized= False
)
print(kernelOp)


################################################
# 3. Apply LDDMM
# Run the registration: optimize the momentum field, integrate to get the
# deformation, visualize convergence, and save the results.
# rho = 0  Pure photometric registration
# rho = 1  Pure geometric registration

# data_cost = mt.Ssd(fixed)
data_cost = mt.Mutual_Information(fixed)


rho = .7
print("\nApply LDDMM")
mr = mt.metamorphosis(moving,fixed,
    momentum_ini = 0,
    rho = rho,
    kernelOperator=kernelOp,       #  Kernel
    data_cost=data_cost,      # You can also design your own data_cost
    cost_cst=0.001,         # Regularization parameter
    integration_steps=10,   # Number of integration steps
    n_iter=15,             # Number of optimization steps
    grad_coef=1,            # max Gradient coefficient
    data_term=None,         # Data term (default Ssd)
    safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
    integration_method='semiLagrangian',  # You should not use Eulerian for real usage
)
# Plot the evolution of the objective to check convergence.
mr.plot_cost()
# Extract the deformation field to inspect/serialize if needed.
deform = mr.mp.get_deformation().detach()
img_deformed = tb.imgDeform(fixed.to("cpu"), deform, dx_convention= mr.dx_convention)
cmp_imgtar = tb.imCmp(mr.mp.image.detach(), fixed.cpu(), 'seg')[0]
cmp_deftar = tb.imCmp(img_deformed, fixed.cpu(), 'seg')[0]

fig, ax = plt.subplots(2,3, figsize=(15,10), constrained_layout=True)
ax[0,0].imshow(slc_plt(moving, coord), cmap='gray')
ax[0,0].set_title('moving')
ax[0,1].imshow(slc_plt(mr.mp.image, coord), cmap='gray')
ax[0,1].set_title('integrated moving')
ax[0,2].imshow(slc_plt(fixed, coord), cmap='gray')
ax[0,2].set_title('fixed')

ax[1,0].imshow(slc_plt(img_deformed, coord), cmap='gray')
ax[1,0].set_title('only deformation')
ax[1,1].imshow(slc_plt(cmp_deftar, coord))
ax[1,1].set_title('img deformed vs target')
ax[1,2].imshow(slc_plt(cmp_imgtar, coord))
ax[1,2].set_title('img integrated vs target')
plt.show()
fig.savefig(ROOT_DIRECTORY / f"examples/results/lddmm_on_MRIbrains_rho_{rho}.png")


# Save the registration object (controls, deformation, etc.) for reuse.
mr.save("test_lddm_on_MRIbrains")
