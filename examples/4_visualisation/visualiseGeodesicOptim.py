"""
.. visualisze_geodesic_optim:

Displays 3D images along with deformation field
======================================
"""
#############################################
# import the necessary packages
import sys
import os

from demeter.utils.torchbox import SegmentationComparator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt

import demeter.metamorphosis as mt
import  demeter.utils.axes3dsliders_plt as a3s
#############################################

#%%

# file = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"
file = "3D_02_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"

file = "3D_20250725_BraTSReg_021_rigid_metamorphosis_rho0_colab_root_000.pk1"
file = "3D_20250727_BraTSReg_021_rigid_metamorphosis_rho0.05_colab_root_000.pk1"
file = "3D_20250730_IXI002_to_template_rigid_metamorphosis_colab_root_000.pk1"
file = "3D_20250730_IXI002_to_template_flirt_LDDMM_colab_root_001.pk1"
file = "3D_20250730_IXI002_to_template_flirt_LDDMM_colab_root_002.pk1"
file = "3D_20250730_IXI002_to_template_ixibrain_rigidscalingLDDMM_colab_root_000.pk1"
file = "3D_20250829_IXI002_to_template_ixibrain_rigidscalingLDDMM_colab_root_002.pk1"
file = "3D_20250902_IXI002_to_template_ixibrain_rigidscalingLDDMM_colab_turtlefox_000.pk1"
file = "3D_20250910_IXI040-Guys-0724-T1_flirt_lddmm_francoisa_000.pk1"
file = "3D_20250910_IXI022-Guys-0701-T1_flirt_lddmm_francoisa_000.pk1"
file = "3D_20250910_IXI026-Guys-0696-T1_flirt_lddmm_francoisa_000.pk1"

# file ="3D_20250910_IXI026-Guys-0696-T1_flirt_lddmm_francoisa_000.pk1"
file ="3D_20250911_IXI026-Guys-0696-T1_flirt_lddmm_francoisa_000.pk1"

# file = "3D_20250911_IXI026-Guys-0696-T1_rigid_along_lddmm_alpha0.1_francoisa_000.pk1"
file = "3D_20250911_IXI026-Guys-0696-T1_rigid_along_lddmm_costcst0.1_francoisa_000.pk1"
# file = "3D_20250911_IXI026-Guys-0696-T1_flirt_lddmm_francoisa_000.pk1"

file = "3D_20250926_IXI026-Guys-0696-T1_rigid_along_lddmm_root_001.pk1"

file = "3D_20250930_IXI026-Guys-0696-T1_rigid_along_lddmm_root_000.pk1"

file = "3D_20250929_IXI040-Guys-0724-T1_rigid_along_lddmm_francoisa_000.pk1" #{"gpu": "Tesla V100S-PCIE-32GB", "alpha": 0.3, "rho": 1, "cost_cst": 1000000.0, "cst_field": 0.005, "sigma": [[1, 1, 1], [3, 3, 3], [7, 7, 7]], "integration_steps": 10, "file": "/gpfs/workdir/francoisa/data/IXI_results/rigid_along_lddmm/
# file =  "3D_20250930_IXI035-IOP-0873-T1_rigid_along_lddmm_francoisa_001.pk1" # {"gpu": "Tesla V100-PCIE-32GB", "alpha": 0.3, "rho": 1, "cost_cst": 1000000.0, "cst_field": 0.005, "sigma": [[1, 1, 1], [3, 3, 3], [7, 7, 7]], "integration_steps": 10, "file": "/gpfs/workdir/francoisa/data/IXI_results/rigid_along_lddmm/"}

import demeter.utils.torchbox as tb
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),
    # path="/home/turtlefox/.local/share/Demeter_metamorphosis/saved_optim"
    path = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/rigid_along_lddmm/"
)


name = file.split('.')[0]
# print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
print("IMG stock :",mr.mp.image_stock.shape)
T, _,D,H,W = mr.mp.image_stock.shape

mr.plot_cost()


src_rot = tb.imgDeform(mr.source, mr.mp.get_rigidor())
img_rot = tb.imgDeform(mr.mp.image, mr.mp.get_rigidor())

cmp_img = tb.imCmp(img_rot, mr.target, "compose")[0]

cs = tb.SegmentationComparator()
cmp_seg = cs(mr.source_seg_deformed, mr.target_segmentation)[0]

T, _, D, H, W = mr.source.shape
print(f"residual min {mr.mp.residuals.min()} max {mr.mp.residuals.max()}")
# Choose a central slice for plotting
slice_index = W // 2 +3
fig, ax = plt.subplots(2,4,figsize = (10,5), constrained_layout=True)
ax[0,0].imshow(mr.source_seg_deformed[0,0,..., slice_index].detach().cpu(), cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5)
ax[0,0].set_title(f'deformed_source_seg')
ax[0,1].imshow(mr.mp.image[0,0,..., slice_index].cpu().detach(), cmap='gray')
ax[0,1].set_title(f'image sans rot')
ax[0,2].imshow(src_rot[0,0,..., slice_index].detach().cpu(), cmap='gray')
ax[0,2].set_title(f'source rot')
ax[0,3].imshow(mr.target[0,0,..., slice_index].detach().cpu(), cmap='gray')
ax[0,3].set_title(f'target')

ax[1,0].imshow(mr.target_segmentation[0,0,..., slice_index], cmap =tb.DLT_SEG_CMAP, vmin=0, vmax = 5)
ax[1,0].set_title('target')

ax[1,1].imshow(cmp_seg[..., slice_index, :])
ax[1,1].set_title(f'deformed_source_seg vs target')



ax[1,2].imshow(img_rot[0,0,..., slice_index].detach().cpu(), cmap='gray')
ax[1,2].set_title('image def+rotated')

ax[1,3].imshow(cmp_img[..., slice_index,:], cmap='gray')
ax[1,3].set_title('image def+rotated vs target')



# fig.suptitle(subject_name)
plt.show()

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