"""
.. _brain_radioaide_metamorphosis:

VIsualize Metamorphosis on 3D brain MRI
================================================

In this file we apply the metamorphosis algorithm to 3D images. (or load the
optimization results if they have been computed before)
And we visualize the results with a visualisation tool.

"""

################################################################################
# 1. Import necessary libraries
# try:
#     # sys.path.insert(0,os.path.join(base_path,'examples'))
#     import sys, os
#     # add the parent directory to the path
#     base_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     sys.path.insert(0,os.path.join(base_path,'src'))
# except ImportError:
#
#     base_path = ROOT_DIRECTORY
#     pass
try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass

from demeter.constants import *
import torch
from icecream import ic
import matplotlib.pyplot as plt

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
from nibabel import load
ic(ROOT_DIRECTORY)
base_path = ROOT_DIRECTORY
source_path = base_path+ "/examples/im3Dbank/bet_PSL_001_M01_flair_mr_dataset_FLAIR_3D.nii.gz"
# target_path = "im3Dbank/reg_bet_from_PSL_001_M03_to_PSL_001_M01_FLAIR3D.nii.gz"
target_path = base_path+ "/examples/im3Dbank/reg_bet_from_PSL_001_M26_to_PSL_001_M23_FLAIR3D.nii.gz"
S = load(source_path).get_fdata()[None,None]
T = load(target_path).get_fdata()[None,None]

S = torch.tensor(S).to(device) / S.max()
T = torch.tensor(T).to(device) / T.max()
ic(S.shape,T.shape)
# if the image is too big for your GPU, you can downsample it quite barbarically :
factor = .5
if factor < 1:
    S = tb.resize_image(S,factor)
    T = tb.resize_image(T,factor)
    ic(S.shape,T.shape)
    # S = S[:,:,::step,::step,::step]
    # T = T[:,:,::step,::step,::step]
_,_,D,H,W = S.shape


st = tb.imCmp(S,T,method = 'compose')
# sl = i3p.imshow_3d_slider(S, title = 'Source (orange) ')
# sl = i3p.imshow_3d_slider(T, title = 'Target (blue)')

sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and Target (blue)')
plt.show()




# reg_grid = tb.make_regular_grid(S.size(),device=device)
#%%
################################################################################
#  Inititialize a kernel Operator

kernelOp = rk.GaussianRKHS((8,8,8), normalized= True)

###########################################################################
# For this example in particular, the images are too big for a laptop computer
# even with a GPU. If you have access to a server with a GPU with vram, you can try to
# run the following code (adapting the path to the server's file system)
# If not, I provide the results of an optimization for you enjoy the visualisation.

recompute = False
saved_lddmm_optim = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"
saved_meta_optim = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_Metamorphosis_000.pk1"

########################################################################
# LDDMM
#

if recompute:
    momentum_ini = 0
    ## Setting momentum_ini to 0 is equivalent to writing the
    ## following line of code :
    # momentum_ini= torch.zeros((D,H,W),device = device)
    # momentum_ini.requires_grad = True

    print("\nApply LDDMM")
    mr_lddmm = mt.lddmm(S,T,momentum_ini,
        kernelOperator=kernelOp,       #  Kernel
        cost_cst=0.001,         # Regularization parameter
        integration_steps=10,   # Number of integration steps
        n_iter=20,             # Number of optimization steps
        grad_coef=1,            # max Gradient coefficient
        data_term=None,         # Data term (default Ssd)
        safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
        integration_method='semiLagrangian',  # You should not use Eulerian for real usage
    )
    mr_lddmm.plot_cost()

    # # you can save the optimization:
    mr_lddmm.save('PSL_001_M01_to_M26_FLAIR3D_LDDMM',light_save= True)
    mr_lddmm.to_device('cpu')
else:
    mr_lddmm = mt.load_optimize_geodesicShooting(saved_lddmm_optim)

deformation = mr_lddmm.mp.get_deformation()

# We provide some visualisation tools :
i3p.Visualize_GeodesicOptim_plt(mr_lddmm,name="PSL_001_M01_to_M26_FLAIR3D_LDDMM")
plt.show()



########################################################################
# Metamorphosis
# rho = 0  Pure photometric registration
# rho = 1  Pure geometric registration
rho = 0.2
dx_convention = 'square'
if recompute:
    # print("\nApply Metamorphosis")
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
    mr_meta.save('PSL_001_M01_to_M26_FLAIR3D_Metamorphosis',light_save= True)
else:
    mr_meta = mt.load_optimize_geodesicShooting(saved_meta_optim)

i3p.Visualize_GeodesicOptim_plt(mr_meta,name="PSL_001_M01_to_M26_FLAIR3D_Metamorphosis")
# # you can get the deformation grid with mr_meta.mp.get_deformation()
image_deformed = tb.imgDeform(S.cpu(),mr_meta.mp.get_deformator(),dx_convention=dx_convention)
imdef_target = tb.imCmp(image_deformed,T,method = 'compose')
sl = i3p.imshow_3d_slider(imdef_target, title = 'Metamorphosis only deformation')
plt.show()



