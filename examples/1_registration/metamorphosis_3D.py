"""
.. _metamorphosis_3D:
Metamorphosis on 3D images
================================================

In this file we apply the metamorphosis algorithm to 3D images.

"""
################################################################################
# 1. Import necessary libraries
try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass

import torch

from demeter.constants import *
import demeter.metamorphosis as mt
import demeter.utils.image_3d_plotter as i3p
from demeter.utils import *
import demeter.utils.reproducing_kernels as rk
import demeter.utils.torchbox as tb

cuda = torch.cuda.is_available()
# cuda = True
device = 'cpu'
if cuda:
    device = 'cuda:0'
print('device used :',device)
if device == 'cpu':
    print('Warning : the computation will be slow, it is recommended to use a GPU')

################################################################################
# 2. Load the images and visualize them

path = ROOT_DIRECTORY+"/examples/im3Dbank/"
source_name = "ball_for_hanse"
target_name = "hanse_w_ball"
S = torch.load(path+source_name+".pt").to(device)
T = torch.load(path+target_name+".pt").to(device)

# if the image is too big for your GPU, you can downsample it quite barbarically :
step = 2 if device == 'cuda:0' else 3
if step > 0:
    S = S[:,:,::step,::step,::step]
    T = T[:,:,::step,::step,::step]
_,_,D,H,W = S.shape

st = tb.imCmp(S,T,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and Target (blue)')
plt.show()

## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
momentum_ini = 0

# reg_grid = tb.make_regular_grid(S.size(),device=device)

################################################################################
#  Inititialize a kernel Operator

kernelOp = rk.GaussianRKHS((4,4,4), normalized= True)


########################################################################
# LDDMM

# mu = 0
# mu,rho,lamb = 0, 0, .0001   # LDDMM

# print("\nApply LDDMM")
# mr_lddmm = mt.lddmm(S,T,momentum_ini,
#     kernelOperator=kernelOp,       #  Kernel
#     cost_cst=0.001,         # Regularization parameter
#     integration_steps=10,   # Number of integration steps
#     n_iter=4,             # Number of optimization steps
#     grad_coef=1,            # max Gradient coefficient
#     data_term=None,         # Data term (default Ssd)
#     safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
#     integration_method='semiLagrangian',  # You should not use Eulerian for real usage
# )
# mr_lddmm.plot_cost()
#
# mr_lddmm.save("ballforhance_LDDMM",light_save= True)
# mr_lddmm.to_device('cpu')
# deformation = mr_lddmm.mp.get_deformation()
# # # you can save the optimization:
# # # mr_lddmm.save(source_name,target_name)
#
# image_to_target = tb.imCmp(mr_lddmm.mp.image.cpu(),T,method = 'compose')
# sl = i3p.imshow_3d_slider(image_to_target, title = 'LDDMM result')
# plt.show()

#  visualization tools with issues,TO FIX !
# i3p.Visualize_geodesicOptim(mr_lddmm,alpha=1)

# plt_v = i3p.compare_3D_images_vedo(T.cpu(),mr_lddmm.mp.image_stock.cpu())
# plt_v.show_deformation_flow(deformation,1,step=3)
# plt_v.plotter.show(interactive=True).close()


########################################################################
# Metamorphosis
# rho = 0  Pure photometric registration
# rho = 1  Pure geometric registration
rho = 0.2
dx_convention = 'square'
# # print("\nApply Metamorphosis")
print("\nApply Metamorphosis")
mr_meta = mt.metamorphosis(S, T, momentum_ini,
                           rho=rho,  # ratio deformation / intensity addition
                           kernelOperator=kernelOp,  #  Kernel
                           cost_cst=0.001,  # Regularization parameter
                           integration_steps=10,  # Number of integration steps
                           n_iter=15,  # Number of optimization steps
                           grad_coef=1,  # max Gradient coefficient
                           data_term=None,  # Data term (default Ssd)
                           safe_mode = False,  # Safe mode toggle (does not crash when nan values are encountered)
                           integration_method='semiLagrangian',  # You should not use Eulerian for real usage
                           dx_convention=dx_convention
                           )
mr_meta.plot_cost()
mr_meta.save(f"{source_name}_{target_name}_Metamorphosis")
image_to_target = tb.imCmp(mr_meta.mp.image.cpu(),T,method = 'compose')



sl = i3p.imshow_3d_slider(image_to_target, title = 'Metamorphosis result')
image_deformed = tb.imgDeform(S.cpu(),mr_meta.mp.get_deformator(),dx_convention=dx_convention)
imdef_target = tb.imCmp(image_deformed,T,method = 'compose')
sl = i3p.imshow_3d_slider(imdef_target, title = 'Metamorphosis only deformation')
ic(mr_meta.mp.image_stock.shape)
# sl = i3p.imshow_3d_slider(mr_meta.mp.image_stock, title = 'Metamorphosis evolution')
plt.show()


#
# # you can get the deformation grid:
# deformation  = mr_meta.mp.get_deformation()

# We provide some visualisation tools :

# i3v.Visualize_geodesicOptim(mr_meta,alpha=1)
# plt_v = i3p.compare_3D_images_vedo(T,mr_meta.mp.image_stock.cpu())
# plt_v.show_deformation_flow(deformation,1,step=3)
# plt_v.plotter.show(interactive=True).close()