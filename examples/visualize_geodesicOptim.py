import torch
import numpy as np

import __init__
import demeter.metamorphosis as mt
import demeter.utils.image_3d_plotter as i3p
import demeter.utils.torchbox as tb
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# file = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"

# file = "3D_30_01_2025_ballforhance_LDDMM_000.pk1"
file = "3D_01_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"
# # file = "3D_02_12_2024_PSL_005_M06_to_PSL_005_M07_000.pk1"
mr = mt.load_optimize_geodesicShooting(file)
name = file.split('.')[0]
#
# print(mr.mp.image_stock.shape)


# i3v.Visualize_geodesicOptim(mr)




vg = i3p.Visualize_GeodesicOptim_plt(mr, name)
# vg.save_all_times("ballforhance_LDDMM")
# vg.show_grid()
vg.show()


# deform = mr.mp.get_deformator(save=True)
# # TODO : vérifier que vf est déjà bien un cv intégré  (les fleches de vraient
# # se suivre)
# vf = torch.cat(
#     [deform[0][None] - mr.mp.id_grid, deform[:-1] - deform[1:]], dim=0
# )
# T, D, H, W, _ = deform.shape
#
# d, h, w = (20, 20, 20)
# print("T:", T, ", D:", D, ", H:", H, ", W:", W, ":: ", H * D * W)
# print(" d:", d, ", h:", h, ", w:", w, ":: ", h * d * w)
#
# lx: torch.Tensor = torch.linspace(0, D - 1, d)
# ly: torch.Tensor = torch.linspace(0, H - 1, h)
# lz: torch.Tensor = torch.linspace(0, W - 1, w)
#
# mx, my, mz = torch.meshgrid([lx, ly, lz])
# pts = torch.stack([mz.flatten(), my.flatten(), mx.flatten()], dim=1).numpy()
# reg_grid = torch.stack((mx, my, mz), dim=-1)[None]  # shape = [1,d,h,w]
# print("\nMAKE ARROW reg_grid", reg_grid.shape)
# print(vf.shape)
# # deform = deform[:,lx.to(int),ly.to(int),lz.to(int)]
# deform = deform[:, :, :, lz.to(int)][:, :, ly.to(int)][:, lx.to(int)]
# vf = vf[:, :, :, lz.to(int)][:, :, ly.to(int)][:, lx.to(int)]
#
# ic(deform.shape,vf.shape)
# ic(mr.mp.id_grid.min(), mr.mp.id_grid.max(), mr.dx_convention)



# fig, ax = plt.subplots(1,1)
# ax.imshow(mr.mp.image[0,0,:,100])
# ax.plot(vf[0,:,100,:,0])
# plt.show()

