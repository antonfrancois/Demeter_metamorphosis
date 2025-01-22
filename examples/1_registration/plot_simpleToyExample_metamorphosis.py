"""
This is an informative docstring

"""

try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass


#####################################################################
# Import the necessary packages

from demeter.utils.constants import *
# import torch
# import kornia.filters as flt
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb



#####################################################################
# Open and visualise images before registration. The source and target are 'C' shapes.
# The source is a 'C' shape that is deformed. The target is a 'C' shape that was
# cut in half changing its topology and a point was added. The goal is to register
# the source to the target by deforming the 'C' shape and accounting the cut
# and the point as intensity additions.

source_name,target_name = 'm0t', 'm1c'
# source_name,target_name = '17','20'
size = (300,300)

S = tb.reg_open(source_name,size = size)
T = tb.reg_open(target_name,size = size)

fig, ax = plt.subplots(1,3,figsize=(10,5))
ax[0].imshow(S[0,0],**DLT_KW_IMAGE)
ax[0].set_title('source')
ax[1].imshow(T[0,0],**DLT_KW_IMAGE)
ax[1].set_title('target')
ax[2].imshow(tb.imCmp(S,T,'seg'),origin='lower')
ax[2].set_title('superposition of S and T')
plt.show()

#####################################################################
# Before choosing the optimisation method, we need to define a
# reproducing kernel and choose a good sigma. The simpler reproducing
# kernel is the Gaussian kernel. To choose the sigma, we can use the
# helper functions. `get_sigma_from_img_ratio` and `plot_kernel_on_image`.
# The first one will compute a good sigma to match the level of details desired.
# A big sigma will produce a smoother deformation field that will register better
# big structures. A smaller sigma will register better small details. The subdivisions
# is basically in how many parts we want to divide the image to get the size
# of wanted details. The second function will plot the kernel on the image to
# help us validate our choice of sigma.

image_subdivisions = 10
sigma = rk.get_sigma_from_img_ratio(T.shape,subdiv = image_subdivisions)

kernelOperator = rk.GaussianRKHS(sigma,kernel_reach=7)

rk.plot_kernel_on_image(kernelOperator,image= T,subdiv=image_subdivisions)
plt.show()


#####################################################################
# Perform a first Metamorphosis registration
device = 'cuda:0'
S = S.to(device)
T = T.to(device)
dx_convention = 'square'
# dx_convention = 'pixel'

rho = .1
#
# data_cost = mt.Ssd_normalized(T)
data_cost = mt.Ssd(T)

mr = mt.metamorphosis(S,T,0,
                      rho,
                      cost_cst=.001,
                      kernelOperator=kernelOperator,
                      integration_steps=10,
                      n_iter=15,
                      grad_coef=1,
                      dx_convention=dx_convention,
                    data_term=data_cost,
                    hamiltonian_integration=True
                      )

mr.save('simpleToyExample_test',light_save = True)
#%%
mr.plot()
mr.plot_deform()
mr.mp.plot()
plt.show()
#%%
#####################################################################
# We will test different values of rho to see how the registration behaves
# with different values of rho. To save you time I already computed the
# optimisation for the values of rho in the files listed in list_optim.
# Feel free to try yourselves If you want to recompute them by setting
# recompute to True. The number of rho to test is set by n_plot.

list_optim = [
    "2D_21_01_2025_simpleToyExample_rho_0.00_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.11_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.22_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.33_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.33_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.44_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.56_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.67_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.78_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_0.89_to__000.pk1",
    "2D_21_01_2025_simpleToyExample_rho_1.00_to__000.pk1",
]
recompute = False
n_plot = 10
rho_list = torch.linspace(0,1,n_plot)

fig,ax = plt.subplots(2,n_plot,figsize=(20,5))

for i,rho in enumerate(rho_list):
    print(f'\nrho = {rho}, {i+1}/{n_plot}')
    if recompute:
        mr = mt.metamorphosis(S,T,0,
                          rho,
                          cost_cst=.001,
                          kernelOperator=kernelOperator,
                          integration_steps=10,
                          n_iter=30,
                          grad_coef=.1,
                          dx_convention=dx_convention,
                          data_term=data_cost,
                          hamiltonian_integration=True
                          )
        mr.save(f'simpleToyExample_rho_{rho:.2f}','')
    else:
        mr = mt.load_optimize_geodesicShooting(list_optim[i])

    # mr.plot_cost()
    ax[0,i].set_title(f'rho = {rho}')
    ax[0,i].imshow(mr.mp.image[0,0].detach().cpu(),**DLT_KW_IMAGE)
    deform = mr.mp.get_deformator()
    img_deform = tb.imgDeform(S.cpu(),deform,dx_convention=dx_convention)
    ax[1,i].imshow(img_deform[0,0].detach().cpu(),**DLT_KW_IMAGE)


plt.show()

# sphinx_gallery_thumbnail_number = 6