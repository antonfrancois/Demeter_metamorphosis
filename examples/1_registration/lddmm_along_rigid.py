import __init__
import torch
from math import cos,sin
import matplotlib.pyplot as plt

import demeter.utils.torchbox as tb
import demeter.metamorphosis.rotate as mtrt
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.rigid_exploration as rg
from demeter.constants import set_ticks_off, GRIDDEF_YELLOW


#
def smooth(image, sigma):
    if isinstance(sigma, int):
        sigma = (sigma,sigma)
    kernel = rk.GaussianRKHS(sigma).kernel

    return rk.fft_filter(image,kernel,border_type='constant')

def plot(self):
    affine = self.mp.get_rigidor()
    deform = self.mp.get_deformation()
    deform = self.mp.get_rigid(deform)

    img_rot = tb.imgDeform(self.mp.image.to('cpu'),affine,dx_convention='2square')
    source_rt = tb.imgDeform(self.source.to('cpu'),affine,dx_convention='2square')
    srt = tb.imCmp(source_rt,self.target,method = 'compose')
    irt = tb.imCmp(img_rot,self.target,method = 'compose')
    kwargs = {"origin": "lower", 'cmap': "gray"}

    fig,ax = plt.subplots(3,3, constrained_layout=True)
    ax[0,0].imshow(self.source[0,0], **kwargs)
    ax[0,0].set_title("source")

    ax[0,1].imshow(self.target[0,0], **kwargs)
    ax[0,1].set_title("target")

    tb.gridDef_plot_2d(self.id_grid, step = 40, ax = ax[0,2], color = None, alpha = .4)

    tb.gridDef_plot_2d(deform, step = 40, ax = ax[0,2])

    ax[1,0].imshow(self.mp.image.to('cpu')[0,0], **kwargs)
    ax[1,0].set_title("image deformed")

    ax[1,1].imshow(img_rot[0,0], **kwargs)

    ax[1,2].imshow(source_rt[0,0], **kwargs)
    ax[1,2].set_title("source affne")

    ax[2,1].imshow(irt[0], **kwargs)
    ax[2,1].set_title("registered vs Target")
    ax[2,2].imshow(srt[0], **kwargs)
    ax[2,2].set_title("source affine vs Target")

path = "examples/results/rigid_meta/"
###########################################################
# open images
size = (300, 300)
source = tb.reg_open('rigid_s',size=size)
target = tb.reg_open('rigid_t',size=size)

# source = smooth(source, 20)
# target = smooth(target, 20)

fig, ax = plt.subplots(1,3)
ax[0].imshow(source[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(target[0,0],cmap='gray')
ax[1].set_title("target")
ax[2].imshow(tb.imCmp(source,target, 'compose')[0])
plt.show()

# Align barycenters

source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
fig, ax = plt.subplots(1,3, constrained_layout=True, figsize=(5.5,2))
ax[0].imshow(source_b[0,0],cmap='gray')
ax[0].set_title("Source")
ax[1].imshow(target_b[0,0],cmap='gray')
ax[1].set_title("Target")
ax[2].imshow(tb.imCmp(source_b,target_b, 'seg')[0])
ax[2].set_title("Source vs Target")
set_ticks_off(ax)
plt.show()
# fig.savefig(path + "toyexample_sourcetarget.pdf")

# %%
integration_steps = 10



kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'), gamma=1, plot=False)
# datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)

mr_rigid = mt.rigid_along_metamorphosis(
    source_b, target_b, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='LBFGS_torch',
    cost_cst=.1,
    n_iter=0,
    lbfgs_max_iter=10
)

top_params = rg.initial_exploration(mr_rigid, r_step = 100,
                                    max_output = 5, verbose=True)
print(top_params)

best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
    mr_rigid, top_params,
    n_iter=10, grad_coef = .1,
    verbose=True, plot = True, affine=True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_momenta : {best_momenta}")
id = 1


# #%%
# #####################################################
# # Choose a specific rigid optimisation changes the optimisation
#
# best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
#     mr_rigid, [top_params[-1]], n_iter=2,verbose=True, plot = True,
# )
# print(f"best_loss : {best_loss}")
# print(f"best_rot : {best_rot}")
# print(f"best_momenta : {best_momenta}")
# # {'rot_prior': torch.tensor(-1.0472), 'trans_prior': None, 'scale_prior': None}
# id = 2
#
# #%%
# best_momenta = {'affine_prior': torch.tensor([[-0.5799,  3.0117],
#         [-3.0049, -0.5558]]),
#                 'rot_prior': None,
#                 'trans_prior': torch.tensor([0.4199, 0.0598]),
#                 'scale_prior': None}
#%%
#####################################################
# Check the rigid optimisation


print(f"best_momenta : {best_momenta}")
param = best_momenta.copy()
momenta = mt.prepare_momenta(
    source_b.shape,
    diffeo = False,
    affine = True,
    device = "cpu",
    requires_grad = False,
    **param
)
print(f"best_momenta : {best_momenta}")

print(f"momenta : {momenta}")

mr_rigid.mp.forward(source_b, momenta.copy(), save =  True)

plot(mr_rigid)
plt.show()

#%%

#%% lddmm along rigid
#########################################################
# perfom lddmm along rigid
sigma= [  7, 15]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 1
cst_field = 1

kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(target_b.to("cuda:0"), gamma=alpha)


# best_loss = torch.inf
# for i,param in enumerate(top_param_rot):
#     print(f"\n\noptimistion {i} on  {len(top_param_rot)}")
momenta = mt.prepare_momenta(
    source_b.shape,
    affine = True,
    # rotation=True,scaling=True,translation=True,
    **best_momenta
)
# momenta["momentum_R"].requires_grad = False
# momenta["momentum_S"].requires_grad = False
# momenta["momentum_T"].requires_grad = False


mr = mt.rigid_along_metamorphosis(
  source_b, target_b, momenta_ini=momenta,
  kernelOperator= kernelOperator,
  rho = rho,
  data_term=datacost ,
  integration_steps = integration_steps,
  cost_cst=cost_cst,
  cost_field_cst=cst_field,
  cost_affine_cst = 1,
  n_iter=10,
    grad_coef=.1,
    # optimizer_method='Adam',
  save_gpu_memory=False,
  lbfgs_max_iter = 40,
  lbfgs_history_size = 20,
    safe_mode=True
)
best = False
mr.plot_cost()
plt.show()
plot(mr)
plt.show()
# if mr.data_loss < best_loss or mr.data_loss == 0:
#     print(param)
#     best_mr = mr

mt.free_GPU_memory(mr)


# file_save, path = mr.save(f"{paths["subject_dir"].name}_rigid_along_lddmm",
#         light_save=True,
#         save_path = os.path.join(result_folder, "rigid_along_lddmm")
#         )

# best_mr.plot_cost()
# plt.show()

#%%
plot(mr)
plt.show()

raise TypeError("miam")
#%%

n_figs = 5
plot_id = (
    torch.quantile(
        torch.arange(mr.mp.image_stock.shape[0], dtype=torch.float),
        torch.linspace(0, 1, n_figs),
    )
    .round()
    .int()
)

kw_image_args = dict(
    cmap="gray", extent=[-1, 1, -1, 1], vmin=0, vmax=1
)
# v_abs_max = (mr.mp.residuals_stock.abs().max()).max()
# v_abs_max = torch.quantile(mr.mp.momenta.abs(), 0.99)
momentum =  mr.mp.momenta['momentum_I']
v_abs_max = torch.quantile(momentum.abs(), 0.99)
kw_residuals_args = dict(
    cmap="RdYlBu_r",
    extent=[-1, 1, -1, 1],
    origin="lower",
    vmin=-v_abs_max,
    vmax=v_abs_max,
)
color = "green"
size_fig = 2
# C = self.momentum_stock.shape[1]
fig, ax = plt.subplots(
    n_figs,
    4,
    constrained_layout=True,
    figsize=(size_fig * 4, n_figs * size_fig),
)
for i, t in enumerate(plot_id):
    deform = mr.mp.get_deformation(to_t = t + 1)

    i_s = ax[i, 0].imshow(
        mr.mp.image_stock[t, 0, :, :].detach().numpy(),
        **kw_image_args,
    )
    ax[i, 0].set_ylabel("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    # fig.colorbar(i_s, ax=ax[i, 0], fraction=0.046, pad=0.04)

    tb.gridDef_plot_2d(
        deform,
        add_grid=False,
        ax=ax[i, 1],
        step=int(min(mr.mp.field_stock.shape[2:-1]) / 25),
        check_diffeo=False,
        dx_convention=mr.mp.dx_convention,
        # color = color
    )

    deform = mr.mp.get_rigid(deform)
    img = tb.imgDeform(
        mr.mp.image_stock[t, :, :, :][None],
        mr.mp.get_rigidor()
    ).detach().numpy()[0,0]

    ax[i, 2].imshow(img, **kw_image_args,)
    # ax[i, 2].set_title("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    ax[i, 2].axis("off")

    # ax[i, 3].imshow(torch.rand((10,10)), **kw_image_args,)
    # ax[i, 3].set_title("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    # ax[i, 3].axis("off")
    # # fig.colorbar(i_s, ax=ax[i, 0], fraction=0.046, pad=0.04)

    #
    tb.gridDef_plot_2d(
        deform,
        add_grid=False,
        add_markers=True,
        ax=ax[i, 3],
        step=int(min(mr.mp.field_stock.shape[2:-1]) / 25),
        check_diffeo=False,
        dx_convention=mr.mp.dx_convention,
    )


set_ticks_off(ax)
plt.show()
fig.savefig(path + f"toyexample_star_{id}_integration.pdf")

#%%
def small_plot(self):
    affine = self.mp.get_rigidor()
    deform = self.mp.get_deformation()
    deform = self.mp.get_rigid(deform)

    img_rot = tb.imgDeform(self.mp.image.to('cpu'),affine,dx_convention='2square')
    source_rt = tb.imgDeform(self.source.to('cpu'),affine,dx_convention='2square')
    srt = tb.imCmp(source_rt,target_b,method = 'seg')
    irt = tb.imCmp(img_rot,target_b,method = 'seg')
    kwargs = {'cmap': "gray"}

    fig,ax = plt.subplots(2,2, constrained_layout=True, figsize = (5,5))

    ax[0,0].imshow(img_rot[0,0], **kwargs)
    ax[0,0].set_title("(a) Registered image")
    ax[0,1].imshow(source_rt[0,0], **kwargs)
    ax[0,1].set_title("(b) affine on source")

    ax[1,0].imshow(irt[0], **kwargs)
    ax[1,0].set_title("(c) registered vs Target")
    ax[1,1].imshow(srt[0], **kwargs)
    ax[1,1].set_title("(d) source affine vs Target")
    set_ticks_off(ax)
    return fig
fig  = small_plot(mr)
fig.savefig(path + f"toyexample_star_{id}_summary.pdf")

plt.show()
#%%
###########################################################
# Compare with pure LDDMM on a rigid output

# put the target on source
integration_steps = 10

kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(source_b.to('cuda:0'), gamma=1)
# datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)

mr_rigid_first = mt.rigid_along_metamorphosis(
    target_b,source_b, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='LBFGS_torch',
    cost_cst=.1,
    n_iter=0
)

top_params = rg.initial_exploration(mr_rigid_first,r_step=10, max_output = 10, verbose=True)
best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
    mr_rigid_first, top_params, n_iter=10,verbose=True, plot = True,
)
id = 1
momenta = mt.prepare_momenta(
    source_b.shape,
    diffeo = False,device = "cpu",requires_grad = False,
    **best_momenta
)

print(f"best_momenta : {best_momenta}")
mr_rigid_first.mp.forward(target_b, momenta.copy(), save =  True)
plot(mr_rigid_first)
plt.show()
#%%
source_lddmm = source_b.clone()
target_lddmm = tb.imgDeform(target_b, mr_rigid_first.mp.get_rigidor())
ref = "source"
fig, ax = plt.subplots(1,3, constrained_layout=True)
ax[0].imshow(source_lddmm[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
ax[2].imshow(target_lddmm[0,0],cmap='gray')
ax[2].set_title("target")
plt.show()
#%%%
source_lddmm = tb.imgDeform(source_b, mr_rigid.mp.get_rigidor())
target_lddmm = target_b.clone()
ref = 'target'
fig, ax = plt.subplots(1,3, constrained_layout=True)
ax[0].imshow(source_lddmm[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
ax[1].set_title("target")
ax[2].imshow(target_b[0,0],cmap='gray')
plt.show()

#%%
sigma= [  7, 15]
# sigma = [15, 20]
sigma = [(s,)*2 for s in sigma]
kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)

mr_l = mt.lddmm(
    source_lddmm.to("cuda:0"), target_lddmm.to("cuda:0"), 0, kernelOperator,
    cost_cst=1,
    grad_coef=.1,
    integration_steps=10,
    n_iter  = 75,
)
#%%
mr_l.plot_cost()
plt.show()
#%%
fig, ax = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
image_kw = dict(cmap="gray", origin="lower", vmin=0, vmax=1)
set_ticks_off(ax)
ax[0, 0].imshow(mr_l.source[0, 0, :, :].detach().cpu().numpy(), **image_kw)
ax[0, 0].set_title("source", fontsize=25)
ax[0, 1].imshow(mr_l.target[0, 0, :, :].detach().cpu().numpy(), **image_kw)
ax[0, 1].set_title("target", fontsize=25)
ax[0,2].imshow(
        tb.imCmp(mr_l.target, mr_l.source, method="seg")[0],
    **image_kw,
)
ax[0,2].set_title("source vs target", fontsize=25)

ax[1, 1].imshow(
    tb.imCmp(mr_l.target, mr_l.mp.image.detach().cpu(), method="seg")[0],
    **image_kw,
)
ax[1, 1].set_title("comparaison registred with target", fontsize=25)
ax[1, 0].imshow(mr_l.mp.image[0, 0].detach().cpu().numpy(), **image_kw)
ax[1, 0].set_title("Integrated source image", fontsize=25)
# tb.quiver_plot(
#     mr_l.mp.get_deformation().detach().cpu() - mr_l.mp.id_grid,
#     ax=ax[1, 1],
#     step=15,
#     color=GRIDDEF_YELLOW,
#     dx_convention=mr_l.dx_convention,
# )
tb.gridDef_plot_2d(
        mr_l.mp.get_deformation().detach().cpu(),
        add_grid=False,
        ax=ax[1,1],
        step=int(min(mr_l.mp.field_stock.shape[2:-1]) / 20),
        check_diffeo=False,
        origin=image_kw["origin"],
        dx_convention=mr_l.mp.dx_convention,
        color = GRIDDEF_YELLOW,
    alpha = .5
    )
tb.gridDef_plot_2d(
        mr_l.mp.get_deformation().detach().cpu(),
        add_grid=False,
        ax=ax[1,2],
        step=int(min(mr_l.mp.field_stock.shape[2:-1]) / 20),
        check_diffeo=False,
        origin=image_kw["origin"],
        dx_convention=mr_l.mp.dx_convention,
        color = 'black',
    alpha = .5
    )
plt.show()
fig.savefig(path+f"classic_lddmm_ref{ref}.pdf")
#%%
mr_l.mp.plot()
plt.show()
#%%
tb.gridDef_plot_2d(
    mr_l.mp.get_deformation(),
    step = 10
)
plt.show()

#%%
sigma= [  7, 15]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 10
cst_field = 100000

kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(target_lddmm.to("cuda:0"), gamma=alpha)


# best_loss = torch.inf
# for i,param in enumerate(top_param_rot):
#     print(f"\n\noptimistion {i} on  {len(top_param_rot)}")
momenta = mt.prepare_momenta(
    source_b.shape,
    rotation=True,scaling=True,translation=True,
    # **best_momenta
)
# momenta["momentum_R"].requires_grad = False
# momenta["momentum_S"].requires_grad = False
# momenta["momentum_T"].requires_grad = False


mr = mt.rigid_along_metamorphosis(
  source_lddmm, target_lddmm, momenta_ini=momenta,
  kernelOperator= kernelOperator,
  rho = rho,
  data_term=datacost ,
  integration_steps = integration_steps,
  cost_cst=cost_cst,
  cst_field=cst_field,
  n_iter=10,
    grad_coef=.1,
    # optimizer_method='Adam',
  save_gpu_memory=False,
  lbfgs_max_iter = 40,
  lbfgs_history_size = 20,
    safe_mode=True
)
best = False
mr.plot_cost()
plt.show()
plot(mr)
plt.show()
# if mr.data_loss < best_loss or mr.data_loss == 0:
#     print(param)
#     best_mr = mr

mt.free_GPU_memory(mr)
