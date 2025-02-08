"""
.. _toyExample_grayScott_JoinedMetamorphosis:

This toy example was build to simulate a cancer growth in a brain.
The gray scott texture as been used to add intricate patterns to the
brain background.
"""
import matplotlib.pyplot as plt


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
size = (300,300)

#####################################################################
# Open and visualise images before registration

source_name,target_name = 'teGS_mbs_S','teGS_mbs_T'

S = tb.reg_open(source_name,size = size).to(device)
T = tb.reg_open(target_name,size = size).to(device)
forDice_source = tb.reg_open('te_s_v_seg',size=size)
forDice_target = tb.reg_open('teGS_mbs_segTdice',size=size)
seg_necrosis = tb.reg_open('teGS_mbs_segNec',size=size).to(device)
seg_oedeme = tb.reg_open('te_o_seg',size=size)

# S = torch.zeros_like(S)
# T = torch.zeros_like(T)

# fig,ax = plt.subplots(2,2,figsize=(10,10))
# ax[0,0].imshow(S[0,0].cpu(),**DLT_KW_IMAGE)
# ax[0,0].set_title('source')
# ax[0,1].imshow(T[0,0].cpu(),**DLT_KW_IMAGE)
# ax[0,1].set_title('target')
# ax[1,0].imshow(tb.imCmp(S,T,'seg'),origin='lower')
# ax[1,0].set_title('superposition of S and T')
# ax[1,1].imshow(tb.imCmp(forDice_source,seg_oedeme,'seg'),origin='lower')
# ax[1,1].set_title('segmentations')
# # ax[2,0].imshow(tb.imCmp(seg_oedeme,forDice_target))
# # ax[2,1].imshow(tb.imCmp(ini_ball,seg_necrosis))
# plt.show()

#####################################################################
# Put some landmarks on the images to assess registration quality

source_landmarks = torch.Tensor([
    [int(187),int(145)],
    [int(160),int(65)],
    [int(140),int(84)],
    [int(145),int(210)],
    [int(170),int(180)],
    [int(125),int(105)],
    [int(117),int(175)]
])
target_landmarks = torch.Tensor([
    [int(212),int(144)], # ok
    [int(167),int(65)],
    [int(131),int(80)], # ok
    [int(135),int(222)], # ok
    [int(197),int(202)], # ok
    [int(110),int(100)], # ok
    [int(101),int(189)] # ok
])

id_grid = tb.make_regular_grid(size)

col1 = 'C1'
col2 = 'C9'
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(S[0,0].cpu(),**DLT_KW_IMAGE)
ax[0].plot(source_landmarks[:,0],source_landmarks[:,1],'x',markersize=10,c=col2)
ax[1].imshow(T[0,0].cpu(),**DLT_KW_IMAGE)
ax[1].plot(source_landmarks[:,0],source_landmarks[:,1],'x',markersize=10,c=col2)
ax[1].plot(target_landmarks[:,0],target_landmarks[:,1],'x',markersize=10,c=col1)

ax[1].quiver(source_landmarks[:,0],source_landmarks[:,1],
             target_landmarks[:,0]-source_landmarks[:,0],
             target_landmarks[:,1]-source_landmarks[:,1],
             color= GRIDDEF_YELLOW)

ax[2].imshow(tb.imCmp(T,S,'seg'),origin='lower')
ax[2].plot(source_landmarks[:,0],source_landmarks[:,1],'x',markersize=10,c=col2)
ax[2].plot(target_landmarks[:,0],target_landmarks[:,1],'x',markersize=10,c=col1)
for i in range(source_landmarks.shape[0]):
    ax[2].plot([source_landmarks[i,0],target_landmarks[i,0]],
           [source_landmarks[i,1],target_landmarks[i,1]],'--',c=col1)





#####################################################################
# Build masks for constrained metamorphosis. Constrained Metamorphosis
# can take tree elements as priors:
# - Residual_mask : a temporal mask with at each pixel a value between 0 and 1
#                  controlling the amount of deformation vs photometric changes.
# - orienting_field :a prior precomputed vector field that our deformation
#                          will try to match.
# - orienting_mask : a mask that will be used to weight the orienting field.
#                   for example, if we want to deform only a part, the rest
#                   we can indicate it with this mask.
#
# First we need to set source and target masks.

print("`\n==== Building temporal masks ====")
val_o,val_n = .5,1
segs = torch.zeros(seg_necrosis.shape)
segs[seg_oedeme > 0] = val_o
segs[seg_necrosis > 0] = val_n
# plt.imshow(segs[0,0],vmin=0,vmax=1,cmap='gray')

# make source image
center_o = (160,155)
center_n = (167,163)
ini_ball_n,_ = tb.make_ball_at_shape_center(
    seg_necrosis,verbose=True,force_r=12,force_center=center_n
)
ini_ball_o,_ = tb.make_ball_at_shape_center(
    seg_necrosis,verbose=True,force_r=21,force_center=center_o
)
ini_ball_on = torch.zeros(ini_ball_o.shape)
ini_ball_on[ini_ball_o > 0] = val_o
ini_ball_on[ini_ball_n > 0] = val_n

# segs = torch.ones_like(segs) * .5
# ini_ball_on = torch.ones_like(ini_ball_on) * .5

fig, ax = plt.subplots(1,4,figsize=(15,5))
ax[0].imshow(segs[0,0],vmin=0,vmax=1,cmap='gray',origin='lower')
ax[0].set_title('segs')
ax[1].imshow(ini_ball_on[0,0],vmin=0,vmax=1,cmap='gray',origin='lower')
ax[1].set_title('ini_ball_on')
ax[2].imshow(tb.imCmp(ini_ball_on,segs,'seg'),origin='lower')
ax[2].set_title('superposition')
ax[3].imshow(tb.imCmp(ini_ball_on,S,'seg'),origin='lower')
plt.show()
#%%
#####################################################################
# Register the prior masks to the source and target images using LDDMM:
# First we build the kernel operator that will be used for the registration and
# fix other constants

## Build temporal masks

sigma = [(5,5),(10,10),(15,15)]
# sigma = [(10,10)]
kernelOp = rk.Multi_scale_GaussianRKHS(sigma,normalized=True)
rk.plot_kernel_on_image(kernelOp,subdiv=10,image=T.cpu())
plt.show()
print(kernelOp)
dx_convention = 'square'
n_steps= 20

#%%
#####################################################################
# Register the orienting mask to the source and target images using LDDMM:
print(">>>> Mask for orienting field <<<<")
recompute = True
if recompute:
    momentum_ini = 0
    mr_mask_orienting = mt.lddmm(ini_ball_n.to(device),seg_necrosis,momentum_ini,
                       kernelOperator=kernelOp,cost_cst=1e-5,integration_steps=n_steps,
                       n_iter=10,grad_coef=1,
                       optimizer_method='LBFGS_torch',
                       dx_convention=dx_convention,)
    mr_mask_orienting.save(f"mask_tE_gs_CM_{dx_convention}_n_step{n_steps}_orienting")
else:
    file = "2D_23_01_2025_mask_tE_gs_CM_square_n_step20_orienting_000.pk1"

    mr_mask_orienting = mt.load_optimize_geodesicShooting(file)

mr_mask_orienting.compute_landmark_dist(source_landmarks,target_landmarks)
mr_mask_orienting.plot_imgCmp()
plt.show()
# #%%
# mr_mask_orienting.plot_deform()
mr_mask_orienting.mp.plot()
plt.show()
#%%
#####################################################################
# Register the residual mask to the source and target images using LDDMM:
print(">>>> Mask for residuals field <<<<")

if recompute:
    momentum_ini = 0
    mr_mask_residuals = mt.lddmm(ini_ball_on.to(device),segs,momentum_ini,
                       kernelOperator=kernelOp,cost_cst=1e-5,integration_steps=n_steps,
                       n_iter=10,grad_coef=1,
                       optimizer_method='LBFGS_torch',
                       dx_convention=dx_convention,)
    mr_mask_residuals.save(f"mask_tE_gs_CM_{dx_convention}_n_step{n_steps}_residuals")
else:
    # file = "2D_23_01_2025_mask_tE_gs_CM_square_necrosisOnly_000.pk1"
    file ="2D_23_01_2025_mask_tE_gs_CM_square_n_step20_necrosisOnly_000.pk1"
    # 2D_23_01_2025_mask_tE_gs_CM_square_n_step20_necrosisOnly_001.pk1

    mr_mask_residuals = mt.load_optimize_geodesicShooting(file)

mr_mask_residuals.plot_imgCmp()
plt.show()
#%%
mr_mask_residuals.mp.plot()
plt.show()
# #%%
# # file  ="2D_23_01_2025_mask_tE_gs_CM_square_000.pk1"
# # file = "2D_23_01_2025_mask_tE_gs_CM_square_all_000.pk1"
# file = "2D_23_01_2025_mask_tE_gs_CM_square_n_step20_necrosisOnly_000.pk1"
# mr_mask_oedeme = mt.load_optimize_geodesicShooting(file)
# mr_mask_oedeme.plot_imgCmp()
# plt.show()
# #%%
# # file  ="2D_23_01_2025_mask_tE_gs_CM_square_necrosisOnly_000.pk1"
# file = "2D_23_01_2025_mask_tE_gs_CM_square_n_step20all_000.pk1"
# mr_mask_necrosis = mt.load_optimize_geodesicShooting(file)
# mr_mask_necrosis.compute_landmark_dist(source_landmarks,target_landmarks)
# mr_mask_necrosis.plot_imgCmp()
# plt.show()

#%%
#####################################################################
# Finally, we extract the masks and the fields from the LDDMM object
# and tweak them to our linking.


residuals_mask = mr_mask_residuals.mp.image_stock.clone() #* .5
# residuals_mask *= 1.3
residuals_mask = 1 - residuals_mask
residuals_mask = torch.ones(residuals_mask.shape)



orienting_field = mr_mask_orienting.mp.field_stock.clone()
# In this case we though best to set the orienting mask as
# at constant value
orienting_mask = torch.ones(residuals_mask.shape)
orienting_mask *= .5
# # but you can try with different types of masks, like the one below
# orienting_mask = mr_mask_residuals.mp.image_stock.clone()
# orienting_mask[orienting_mask>.5] =.7

# if w is zero give back power to v
norm_2_on_w  = torch.sqrt((orienting_field**2).sum(dim = -1))
orienting_mask = norm_2_on_w[:,None]/ norm_2_on_w.max()
# orienting_mask[]


# sig = 5 # blur the mask to avoid sharp transitions
# residuals_mask = flt.gaussian_blur2d(residuals_mask,(int(6*sig)+1,int(6*sig)+1),(sig,sig))
# sig = 5
# orienting_mask = flt.gaussian_blur2d(orienting_mask,(int(6*sig)+1,int(6*sig)+1),(sig,sig))

L = [0,2,8,-1]
fig,ax = plt.subplots(2,len(L),figsize=(len(L)*5,10))
ax[0,0].set_title('orienting mask')
ax[1,0].set_title('residuals mask')
for i,ll in enumerate(L):
    ax[0,i].imshow(orienting_mask[ll,0].cpu(),cmap='gray',vmin=0, vmax = 1,origin = "lower")
    tb.quiver_plot(orienting_mask[ll,0][...,None].cpu() * orienting_field[ll][None].cpu(),
                   ax[0,i],
                   step = 10,color='C3',dx_convention=dx_convention)

    ax[1,i].imshow(residuals_mask[ll,0].cpu(),cmap='gray',vmin=0, vmax = 1,origin = "lower")

plt.show()

#%%
fig1,ax1 = plt.subplots(1,1)
ax1.plot(orienting_mask[-1,0,:,150].cpu(),label="orienting_mask")
ax1.plot(residuals_mask[-1,0,:,150].cpu(),label="residuals_mask")
ax1.set_ylim(0,1)
ax1.legend()
plt.title('orienting and residuals masks profiles cut at x=150')
# l = orienting_mask.shape[0]
# L = range(l)
# fig,ax = plt.subplots(l,1,figsize=(10,l*10))
plt.show()



# sigma = [(5,5),(20,20),(30,30)]
# kernelOp = rk.Multi_scale_GaussianRKHS(sigma,normalized=True)

print(kernelOp)

#%%
#####################################################################
# We are now ready to perform the registration with the constrained metamorphosis
# orienting_field =None
# orienting_mask = None
# you can also load the optimisation object from a file
# file = "2D_25_01_2025_TEST_toyExample_grayScott_CM_square_n_step20_000.pk1"
# mr = mt.load_optimize_geodesicShooting(file)



print("\n==== Constrained Metamorphosis ====")
momentum_ini = 0
# momentum_ini = mr_cm.to_analyse[0]
ic.disable()
mr_cm = mt.constrained_metamorphosis(S,T,momentum_ini,
                                     orienting_mask,
                                     orienting_field,
                                     residuals_mask,
                                     kernelOperator=kernelOp,
                                     cost_cst=.00001,
                                     grad_coef=.000001,
                                    n_iter=30,
                                     dx_convention=dx_convention,
                                     hamiltonian_integrator = True
                                        # optimizer_method='adadelta',
                                     )

mr_cm.compute_landmark_dist(source_landmarks,target_landmarks)
mr_cm.plot_cost()
plt.show()


#%%
mr_cm.plot_imgCmp()
plt.show()

#%%

mr_cm.plot_deform()
plt.show()


#%%
mr_cm.mp.plot()
plt.show()
#%%
mr_cm.save(f"TEST_toyExample_grayScott_CM_{dx_convention}_n_step{n_steps}")

#%%

L = [0,2,8,-1]
fig,ax = plt.subplots(1,len(L),figsize=(len(L)*5,10), constrained_layout=True)
ax[0].set_title('orienting mask')
for i,ll in enumerate(L):
    ax[i].imshow(mr_cm.mp.image_stock[ll,0].cpu(),cmap='gray',vmin=0, vmax = 1,origin = "lower")
    ax[i].imshow(residuals_mask[ll,0].cpu(),cmap='Oranges',vmin=0, vmax = 1,origin = "lower",alpha = .5)

plt.show()

plt.show()

