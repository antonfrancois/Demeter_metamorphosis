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
# file = "3D_20250911_IXI026-Guys-0696-T1_rigid_along_lddmm_costcst0.1_francoisa_000.pk1"
# file = "3D_20250911_IXI026-Guys-0696-T1_flirt_lddmm_francoisa_000.pk1"
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),
    # path="/home/turtlefox/.local/share/Demeter_metamorphosis/saved_optim"
    # path = "/home/turtlefox/Documents/11_metamorphoses/data/IXI_results/rigid_along_lddmm/"
)


name = file.split('.')[0]
# print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
print("IMG stock :",mr.mp.image_stock.shape)
T, _,D,H,W = mr.mp.image_stock.shape

# mr.plot_cost()
a3s.Visualize_GeodesicOptim_plt(mr, name)
# import demeter.utils.torchbox as tb
#
# deformation = mr.mp.get_deformation(save=True)
# temporal_seg = tb.imgDeform(mr.source_segmentation.expand(7,-1,-1,-1,-1),
#                             deformation, mode="nearest", dx_convention=mr.dx_convention)
#
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
plt.show()


# image_dict = [
#     {"name" : "img stock","image": mr.mp.image_stock,"cmap":"gray",
#      "seg": temporal_seg},
#     {"name" : "residual","image": mr.mp.residuals, "cmap":"cividis"},
#     {"name":  "target","image":mr.target,"cmap": "gray","seg": mr.target_segmentation},
#     {"name" : "source","image": mr.source,"cmap":"gray", "seg":mr.source_segmentation},
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