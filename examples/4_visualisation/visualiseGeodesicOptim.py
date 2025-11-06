"""
.. visualisze_geodesic_optim:

Displays 3D images along with deformation field
======================================
"""
#############################################
# import the necessary packages
import sys
import os

import numpy as np

from demeter.utils.torchbox import SegmentationComparator
import re
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
import torch
import matplotlib.pyplot as plt

import demeter.metamorphosis as mt
import  demeter.utils.axes3dsliders_plt as a3s


def _to_tensor(img):
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
    chw = torch.nn.functional.pad(chw, (0, dW, 0, dH))  # pad W then H

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

def extract_ixi_id(filename: str) -> str | None:
    m = re.search(r"(IXI\d+-[A-Za-z]+-\d+)", filename)
    return m.group(1) if m else None

#############################################
from demeter.constants import *
import demeter.utils.torchbox as tb
file = "3D_20250918_IXI027-Guys-0710-T1_rigid_along_lddmm_francoisa_000.pk1"

file = "3D_20250930_IXI049-HH-1358-T1_rigid_along_lddmm_francoisa_000.pk1"
file = "3D_20251001_IXI027-Guys-0710-T1_rigid_along_lddmm_francoisa_000.pk1"
file = "3D_20250930_IXI063-Guys-0742-T1_rigid_along_lddmm_francoisa_001.pk1"
file = "3D_20251001_IXI026-Guys-0696-T1_rigid_along_lddmm_francoisa_001.pk1"
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),
    # path="/home/turtlefox/.local/share/Demeter_metamorphosis/saved_optim"
path = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/rigid_along_lddmm/"
)

#%%
dices = mr.get_DICE()
dice = dices[0] | dices[1]


dice_str = f"dice diffeo+affine: {dice["(all) dice average"]:.3f}, dice affine only: {dice['(rotation only) dice average']:.3f}"
print(dice_str)


name = file.split('.')[0]
# print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
print("IMG stock :",mr.mp.image_stock.shape)
T, _,D,H,W = mr.mp.image_stock.shape

mr.plot_cost()




# %matplotlib qt5
src_rot = tb.imgDeform(mr.source, mr.mp.get_rigidor())
img_rot = tb.imgDeform(mr.mp.image, mr.mp.get_rigidor())

cmp_src = tb.imCmp(src_rot, mr.target, "seg")[0]
cmp_img = tb.imCmp(img_rot, mr.target, "seg")[0]
cmp_st = tb.imCmp(mr.source, mr.target, "seg")[0]

# cs = tb.SegmentationComparator()
# cmp_seg_rot = cs(mr.source_seg_rotated, mr.target_segmentation)[0]
# cmp_seg = cs(mr.source_seg_deformed, mr.target_segmentation)[0]



T, _, D, H, W = mr.source.shape
print(f"residual min {mr.mp.residuals.min()} max {mr.mp.residuals.max()}")
# Choose a central slice for plotting
coord = (D//2, H//2, W//2+5)
def slc_plt(img):
  # if len(img.shape) == 5:
  #   return img[0,0,..., slice_index].detach().cpu()
  # elif len(img.shape) == 4:
  #   return img[..., slice_index,:].detach().cpu()
  if len(img.shape) == 5:
    return get_orthogonal_views_concatenated(img[0,0].detach().cpu(), coord)

  elif len(img.shape) == 4:
    return get_orthogonal_views_concatenated(img, coord)



# fig, ax = plt.subplots(4,2,figsize = (11 * .7,10* .7), constrained_layout=True)
# ax[0,0].imshow(slc_plt(mr.source), cmap="gray", origin="lower")
# ax[0,0].set_title(f'source')
# ax[0,1].imshow(slc_plt(mr.target), cmap='gray', origin="lower")
# ax[0,1].set_title(f'target')
#
# ax[1,0].imshow(slc_plt(cmp_st), cmap='gray', origin="lower")
# ax[1,0].set_title("source vs target")
# ax[1,1].imshow(slc_plt(img_rot), cmap='gray', origin="lower")
# # ax[1,0].imshow(slc_plt(cmp_seg), origin="lower")
# ax[1,1].set_title(f'diffeomorphism + affine')
#
# ax[2,0].imshow(slc_plt(mr.mp.image), cmap='gray', origin="lower")
# ax[2,0].set_title(f'diffeomorphism only')
# ax[2,1].imshow(slc_plt(src_rot), cmap='gray', origin="lower")
# ax[2,1].set_title(f'rotation rot')
#
# ax[3,0].imshow(slc_plt(cmp_src), origin="lower")
# ax[3,0].set_title(f'source + affine vs target')
# ax[3,1].imshow(slc_plt(cmp_img), origin="lower")
# ax[3,1].set_title('image def+rotated vs target')


fig, ax = plt.subplots(3,3,figsize = (10,5), constrained_layout=True)
ax[0,0].imshow(slc_plt(mr.source), cmap="gray", origin="lower")
ax[0,0].set_title(f'source')
ax[0,2].imshow(slc_plt(mr.target), cmap='gray', origin="lower")
ax[0,2].set_title(f'target')
ax[0,1].axis('off')

ax[1,0].imshow(slc_plt(src_rot), cmap='gray', origin="lower")
ax[1,0].set_title(f'affine only')
ax[1,1].imshow(slc_plt(mr.mp.image), cmap='gray', origin="lower")
ax[1,1].set_title(f'diffeomorphism only')
ax[1,2].imshow(slc_plt(img_rot), cmap='gray', origin="lower")
ax[1,2].set_title(f'diffeomorphism + affine')

ax[2,0].imshow(slc_plt(cmp_st), cmap='gray', origin="lower")
ax[2,0].set_title("source vs target")
ax[2,1].imshow(slc_plt(cmp_src), origin="lower")
ax[2,1].set_title(f'affine only vs target')
ax[2,2].imshow(slc_plt(cmp_img), origin="lower")
ax[2,2].set_title('diffeo + affine vs target')


# ax[1,0].imshow(slc_plt(mr.source), cmap="gray", origin="lower")
# ax[1,0].imshow(slc_plt(mr.source_segmentation), cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5, alpha =.7, origin="lower")
# ax[1,0].set_title(f'source')
# ax[1,1].imshow(slc_plt(src_rot), cmap='gray', origin="lower")
# ax[1,1].imshow(slc_plt(mr.source_seg_rotated), cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5, alpha =.7,  origin="lower")
# ax[1,1].set_title(f'source rot')
# ax[1,2].imshow(slc_plt(img_rot), cmap='gray', origin="lower")
# ax[1,2].imshow(slc_plt(mr.source_seg_deformed), cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5, alpha =.7,  origin="lower")
# ax[1,2].set_title(f'image diffeo + affine')
# ax[1,3].imshow(slc_plt(mr.target), cmap='gray', origin="lower")
# ax[1,3].imshow(slc_plt(mr.target_segmentation), cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5, alpha =.7,  origin="lower")
# ax[1,3].set_title(f'target')

# ax[2,0].imshow(slc_plt(mr.target), cmap='gray', origin="lower")
# ax[2,0].imshow(slc_plt(cmp_seg_rot), origin="lower")
# ax[2,0].set_title(f'deformed_source_seg vs target')
# ax[2,1].imshow(slc_plt(mr.target), cmap='gray', origin="lower")
# ax[2,1].imshow(slc_plt(cmp_seg), origin="lower")
# ax[2,1].set_title(f'deformed_source_seg vs target')


set_ticks_off(ax)
name_subject = extract_ixi_id(file )
fig.suptitle(name_subject + "\n" +dice_str)
path_save = ROOT_DIRECTORY + "/examples/results/rigid_meta/ixi/"
fig.savefig(path_save + file + '.svg')
plt.show()


#%%
# plt.show()
# a3s.Visualize_GeodesicOptim_plt(mr, name)
# import demeter.utils.torchbox as tb
# #
# deformation = mr.mp.get_deformation(save=True)
# temporal_seg = tb.imgDeform(mr.source_segmentation.expand(10,-1,-1,-1,-1),
#                             deformation, mode="nearest", dx_convention=mr.dx_convention)
# #
# seg_cmpd = tb.temporal_img_cmp(
#     mr.source_segmentation,
#     mr.target_segmentation,
#     seg = True#, method = "compose"
# )
# print(seg_cmpd.shape)
#
# # fig, ax = plt.subplots()
# # ax.imshow(seg_cmpd[0,D//2])
#
# print(temporal_seg.unique())
# fig, ax = plt.subplots(1,7)
# for i in range(7):
#     ax[i].imshow(seg_cmpd[i,D//2])
# plt.show()
#
# print(temporal_seg.shape)
# print("target_seg :",mr.target_segmentation.shape)
#
# ias = a3s.Image3dAxes_slider(mr.mp.image_stock)
# ias.add_image_overlay(temporal_seg, alpha = .5)
# plt.show()


# image_dict = [
#     {"name" : "img stock","image": mr.mp.image_stock,"cmap":"gray",
#      # "seg": temporal_seg
#      },
#     {"name" : "residual","image": mr.mp.residuals, "cmap":"cividis"},
#     {"name":  "target","image":mr.target,"cmap": "gray",
#      # "seg": mr.target_segmentation
#      },
#     {"name" : "source","image": mr.source,"cmap":"gray",
#      # "seg":mr.source_segmentation
#     },
# ]
#
# img_toggle = a3s.ToggleImage3D(image_dict,)
# plt.show()


# dou = a3s.Image3dAxes_slider(mr.mp.image_stock)
#
# deformation = mr.mp.get_deformation(save=True)
#
# grid = a3s.Grid3dAxes_slider(deformation,
#                              shared_context=dou.ctx,
#                              dx_convention=mr.dx_convention,
#                              color_grid='green',
#                             button_position = [0.8, 0.9, 0.1, 0.04]
#                              )
# plt.show()