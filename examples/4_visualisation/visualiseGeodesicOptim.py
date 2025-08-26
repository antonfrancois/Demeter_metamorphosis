"""
.. visualisze_geodesic_optim:

Displays 3D images along with deformation field
======================================
"""
#############################################
# import the necessary packages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import demeter.utils.axes3dsliders_plt as a3s

import demeter.metamorphosis as mt
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
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),

)

name = file.split('.')[0]
print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
a3s.Visualize_GeodesicOptim_plt(mr, name)