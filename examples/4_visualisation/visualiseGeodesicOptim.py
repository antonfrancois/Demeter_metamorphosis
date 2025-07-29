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
from demeter.utils.decorators import deprecated
from demeter.utils.image_3d_plotter import SimplexToHSV
#############################################

import examples.brats_utils as bu
import demeter.utils.torchbox as tb
import matplotlib.pyplot as plt
import demeter.utils.rigid_exploration as rg

brats_list = [
    # "BraTSReg_021",
    "BraTSReg_086",
    ]


device = 'cuda:0'
valid = False
brats_folder= '2022_valid' if valid else '2022_train'
modality = 'flair'
pb = bu.parse_brats(brats_list=brats_list,brats_folder=brats_folder,modality=modality)
save_folder = None
scale_img = .7

i = 0
name = brats_list[i]
img_1,img_2,seg_1,seg_2,landmarks = pb(i,to_torch=True,scale=scale_img,modality=modality)
img_1, img_2 = bu.normalize_mri_with_gliomas(img_1, img_2, seg_1, seg_2, verbose=True)
# img_1 = torch.nn.functional.pad(img_1,(0,0,10,20,0,0), "constant",.5)
print(landmarks[0].shape)
print(img_1.shape)
print(img_2.shape)
# land_2square =  pixel_to_2square_landmark(landmarks[0], img_1.shape)
# ic(landmarks,land_2square)
land_1, land_2 = landmarks
# land_1 =  land_0 +5
print("landmarks 0 \n",land_1)
print("landmarks 1 \n", land_2)
id_grid = tb.make_regular_grid( img_1.shape[2:])

# ias = a3s.Image3dAxes_slider(img_1)
# plt.show()

#%%

def apply_grid_to_landmarks(landmarks, grid):
    """
    Transport landmarks by a deformation grid.
    """
    new_land = torch.zeros_like(landmarks)
    for i, l in enumerate(landmarks):
        new_land[i]  = grid[:, int(l[2]), int(l[1]), int(l[0])]            # Bad orient

    return new_land

def inverse_affine(affine):
    """
    Compute the inverse of the affine transformation.
    """
    rot = affine[:3, :3]
    trans = affine[:3, 3][None]
    inv_rot = rot.T
    inv_trans = - trans.T

    new_aff = torch.cat([inv_rot, inv_trans], dim=1)
    return  torch.cat([new_aff, affine[-1][None]], dim=0)

param = torch.tensor([
    0,0,torch.pi/4,
    0,-0.07,0.05,
    1,1,1
])
affine = tb.create_affine_mat_3d(param)
grid = tb.affine_to_grid_3d(affine, img_1.shape[2:])
inv_aff = inverse_affine(affine)
inv_grid = tb.affine_to_grid_3d(inv_aff, img_1.shape[2:])

img_1_aff = tb.imgDeform(img_1, grid  )

inv_grid = tb.square2_to_pixel_convention(inv_grid,True)
land_1_aff = apply_grid_to_landmarks(land_1, inv_grid)
source =  img_1_aff.contiguous()
target  = img_2.contiguous()

print("="*20)
print("Barycentre alignement\n")
shape_list = torch.tensor(source.size()[2:])/2
print(source.shape)
print(shape_list)

source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
land_1_b = land_1_aff - trans_s * shape_list
land_2_b = land_2 - trans_t * shape_list

print(f"Landmark b 1\n {land_1_b}")
print(f"Landmark b 2\n {land_2_b}")

dist_land_ref= tb.landmark_distance(land_1, land_2)
dist_land_affine = tb.landmark_distance(land_1_aff, land_2)
dist_land_barycentred = tb.landmark_distance(land_1_b, land_2_b)
print()
print("dist_land_ref : ", dist_land_ref)
print("dist_land_affine : ", dist_land_affine)
print("dist_land_barycentred : ", dist_land_barycentred)

a3s.compare_images_with_landmarks(source, source_b,
                                  land_1_aff, land_1_b,
                                  labels = ["source", "source_barycentred"])
a3s.compare_images_with_landmarks(target, target_b,land_2, land_2_b,
                                  labels = ["target", "target_barcentred"])
a3s.compare_images_with_landmarks(source_b, target_b,
                                  land_1_b, land_2_b,
                                  labels = ["source_barycentred", 'target_barcentred'])

raise Exception("boom")

#%%

# file = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"
file = "3D_02_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"

file = "3D_20250725_BraTSReg_021_rigid_metamorphosis_rho0_colab_root_000.pk1"
file = "3D_20250727_BraTSReg_021_rigid_metamorphosis_rho0.05_colab_root_000.pk1"
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),

)
a3s.compare_images_with_landmarks(mr.source,  mr.target, land_1_b, land_2_b, labels=["source", "target"])

print("land source :", mr.source_landmark)
name = file.split('.')[0]
print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
# a3s.Visualize_GeodesicOptim_plt(mr, name)