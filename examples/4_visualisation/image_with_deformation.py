"""
.. _image_with_deformation:

Displays 3D images along with deformation field
======================================
"""
#############################################
# import the necessary packages
import os
import torch
import demeter.utils.axes3dsliders_plt as a3s

import demeter.metamorphosis as mt

#############################################

# file = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"
file = "3D_02_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),

)
name = file.split('.')[0]
deform =  mr.mp.get_deformation(save=True)
img = mr.mp.image_stock
img = torch.clip(img, 0, 1)

print("image, ", img.shape)
print("def ", deform.shape)

ias = a3s.Image3dAxes_slider(img)


grid_slider = a3s.Grid3dAxes_slider(deform,
                                    dx_convention=mr.dx_convention,
                                    shared_context=ias.ctx,
                                    color_grid = "r",
                                    alpha=.7
)
grid_slider.show()

# vg =VisualizeGeodesicOptim(mr, name)
# vg.show()