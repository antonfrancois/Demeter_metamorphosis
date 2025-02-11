r"""
.. _toyExample_grayScott_weightedMetamorphosis:

Weighted metamorphosis - simulated cancer growth
================================================

This toy example was build to simulate a cancer growth in a brain.
This is a simple example of how to use the weighted metamorphosis to register two images..
This example is part of an exercise, it has been truncated to make you complete it.
"""
#####################################################################
# Import the necessary packages
import matplotlib.pyplot as plt

try:
    import sys, os

    # add the parent directory to the path
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, base_path)
    import __init__

except NameError:
    pass

from demeter.constants import *
import torch
import kornia.filters as flt
# %reload_ext autoreload
# %autoreload 2
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
print(f"Used device: {device}")

#####################################################################
# Load the images
# ----------------

size = (300, 300)
source_name, target_name = '23', '24'
S = tb.reg_open(source_name, size=size).to(device)  # Small oval with gray dots
T = tb.reg_open(target_name, size=size).to(device)  # Big circle with deformed gray dots
seg = tb.reg_open('21_seg', size=size).to(device)  # rounded triangle

## Construct the target image
ini_ball, _ = tb.make_ball_at_shape_center(seg, overlap_threshold=.1, verbose=True)
ini_ball = ini_ball.to(device)
T[seg > 0] = 0.5  # Add the rounded triangle to the target

source = S
target = T
# mask = mr.mp.image_stock

source_name = 'oval_w_round'
target_name = 'round_w_triangle_p_rd'

kw_img = dict(cmap='gray', vmin=0, vmax=1)
plt.rcParams["figure.figsize"] = (20, 20)
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(source[0, 0, :, :].cpu().numpy(), **kw_img)
ax[0, 0].set_title('source')
ax[0, 1].imshow(target[0, 0, :, :].cpu().numpy(), **kw_img)
ax[0, 1].set_title('target')
ax[1, 0].imshow(tb.imCmp(source, target), vmin=0, vmax=1)
ax[1, 1].imshow(seg[0, 0].cpu().numpy(), **kw_img)
ax[1, 1].set_title('segmentation')
# plt.show()


#####################################################################
#  Define the kernel operator

sigma = [(5, 5), (15, 15)]
kernelOp = rk.Multi_scale_GaussianRKHS(sigma, normalized=False)
kernelOp.plot()
rk.plot_kernel_on_image(kernelOp, image=target.cpu())
print("sigma", sigma)
# %%
#####################################################################
#  Classic metamorphosis
# -----------------------
#
# We first compute the classic metamorphosis without any mask to compare results.

rho = 0.7
momentum_ini = 0
mr = mt.metamorphosis(source, target, momentum_ini,
                      kernelOperator=kernelOp,
                      rho=rho,
                      integration_steps=10,
                      cost_cst=1e-2,
                      n_iter=20,
                      grad_coef=1,
                      dx_convention='pixel',
                      )
mr.plot()
mr.plot_deform()
mr.mp.plot()
plt.show()
mr.save_to_gif("image deformation", f"simpleCancer_Meta_rho{rho}_image",
               folder="simpleCancer_Meta")

# plt.show()
# %%
#####################################################################
#  Weighted metamorphosis with time constant mask.
# ---------------------------------------------------------
#
# inverse the mask to have M(x) = 0 where we want to add
# intensity.
#

print("\n\nComputing weighted metamorphosis - time constant mask")
print("=" * 20)

cst_mask = 1 - seg.repeat(10, 1, 1, 1) * .5
lamb = .0001
n_iter, grad_coef = (20, .1)
momentum_ini = 0
mr_wm = mt.weighted_metamorphosis(source, target, momentum_ini, cst_mask,
                                  kernelOperator=kernelOp,
                                  cost_cst=lamb,
                                  n_iter=n_iter,
                                  grad_coef=grad_coef,
                                  safe_mode=False,
                                  dx_convention='pixel',
                                  optimizer_method='LBFGS_torch'
                                  )

mr_wm.plot()
mr_wm.plot_deform()
plt.show()

#####################################################################
# Why the result is not as expected? What can you do to improve it?
#
#  Weighted metamorphosis with time evolving mask.
# =============================================================================
#
# Your mission is to model a smart evolving mask that will guide the
# registration process.
print("\n\n Weighted metamorphosis - evolving mask")
print("=" * 20)
print("\tComputing evolving mask")

mask = torch.rand_like(mr.mp.image_stock)

# display the mask at different time
L = [0, 2, 8, -1]
fig, ax = plt.subplots(1, len(L), figsize=(len(L) * 5, 10), constrained_layout=True)
ax[0].set_title('orienting mask')
ax[0].set_title('residuals mask')
for i, ll in enumerate(L):
    ax[i].imshow(mask[ll, 0].cpu(), cmap='gray', vmin=0, vmax=1, origin="lower")

plt.show()
# %%
#####################################################################

# %%

print("\n\tComputing weighted metamorphosis")
n_iter = 15
grad_coef = 1
cost_cst = .0001
residuals = 0
mask = mask.to(device)
# residuals = mr_wm.to_analyse[0].clone().to(device)
mr_wm = mt.weighted_metamorphosis(source, target, residuals,
                                  mask,
                                  kernelOp,
                                  cost_cst,
                                  n_iter,
                                  grad_coef,
                                  safe_mode=False,
                                  dx_convention='pixel',
                                  optimizer_method='LBFGS_torch'
                                  # optimizer_method='adadelta'
                                  )

mr_wm.plot()
plt.show()
# %%
mr_wm.plot_deform()
plt.show()
# %%

mr_wm.mp.plot()
plt.show()
