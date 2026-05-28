import __init__
import torch
from math import cos,sin
import sys, pathlib


sys.path.insert(0, str(pathlib.Path("examples/1_registration").resolve()))
import lddmm_along_utils as lu
import matplotlib.pyplot as plt


import demeter.utils.torchbox as tb
import demeter.metamorphosis.affine as mtrt
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.rigid_exploration as rg
# from build.lib.demeter import ROOT_DIRECTORY
from demeter.constants import set_ticks_off, GRIDDEF_YELLOW, ROOT_DIRECTORY
from demeter.utils.cost_functions import SumSquaredDifference


def _norm(x):
    return torch.linalg.norm(x.reshape(-1)).item()

path = "examples/results/rigid_meta/"
device = "cuda:0"
###########################################################
# open images
size = (300, 300)
# source = tb.reg_open('rigid_s',size=size)
# target = tb.reg_open('rigid_t',size=size)
source = tb.reg_open('minifish',size=size)
target = tb.reg_open('fish',size=size)
# source = tb.reg_open('20',size=size)
# target = tb.reg_open('17',size=size)
# source = tb.reg_open('L',size=size)




fig,ax = plt.subplots(2,2)
ax[0,0].imshow(source[0,0], cmap='gray')
# tb.gridDef_plot_2d(deform, ax = ax[1,0], step = 10, check_diffeo=False)
ax[0,1].imshow(target[0,0], cmap='gray')
ax[1,1].imshow(tb.imCmp(source,target)[0], cmap='gray')
plt.show()


theta = torch.tensor([5*torch.pi/8])              # radians
translation = torch.tensor([[0.12, -0.13]]) # in 2square coords
scale = torch.tensor([.8])

# A_full = torch.tensor([
#     [1.240, -0.032],
#     [0.449,  0.614]
# ], dtype=target.dtype)
A_full = torch.tensor([
    [0.417, -1.3],
    [0.873,  -0.393]
], dtype=target.dtype)

res = lu.apply_registration_models(
    target,
    rotation=theta,
    translation=translation,
    scale=scale,
    full_affine=A_full,
)
keys = ["full_affine","rotation_translation_scaling","rotation_translation","rotation_scaling"]
lu.show_deforms(target, res, keys)

target_name = keys[1]
target = res[target_name]["image"]

rotation=True
scaling=False
translation=True
def _strf_(valbool):
    return "T" if valbool else "F"
modifier_str = (
        "r"+_strf_(rotation)+
        "_s"+_strf_(scaling)+
        "_t"+_strf_(translation)
                )
#%%


source.to(device)
# source = smooth(source, 20)
# target = smooth(target, 20)
target = res[target_name]["image"]

fig, ax = plt.subplots(1,3)
ax[0].imshow(source[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(target[0,0],cmap='gray')
ax[1].set_title("target")
ax[2].imshow(tb.imCmp(source,target, 'compose')[0])
plt.show()
#%%
# Align barycenters

# source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
#
#
# ssd  = SumSquaredDifference(target_b)
# print("ssd target_b - source_b :",ssd(source_b))
#
# fig, ax = plt.subplots(1,3, constrained_layout=True, figsize=(5.5,2))
# ax[0].imshow(source_b[0,0],cmap='gray')
# ax[0].set_title("Source")
# ax[1].imshow(target_b[0,0],cmap='gray')
# ax[1].set_title("Target")
# ax[2].imshow(tb.imCmp(source_b,target_b, 'seg')[0])
# ax[2].set_title("Source vs Target")
# set_ticks_off(ax)
# plt.show()
# # fig.savefig(path + "toyexample_sourcetarget.pdf")

# %%
print("")
print("="*20)
print("Initial exploration")
def init_explo():
    pass
integration_steps = 10



kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target.to('cuda:0'),
                                gamma=1, normalize_ssd=False,
                                plot=False)
# datacost = mt.Rotation_MutualInformation_Cost(target.to('cuda:0'), alpha=1)

mr_rigid = mt.affine_decoupled_along_metamorphosis(
    source, target, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='LBFGS_torch',
    cost_cst=.1,
    n_iter=0,
    lbfgs_max_iter=20
)

top_params = rg.initial_exploration(mr_rigid, r_step = 100,
                                    max_output =10, verbose=True)
# top_params = None
print("top_params : ",top_params)

print("")
print("="*20)
print("Optimize on best exploration ")
best_loss, best_priors, best_rot = rg.optimize_on_rigid(
    mr_rigid, top_params,
    n_iter=50, grad_coef = .1,
    # affine=True,
    rotation=rotation, scaling=scaling, translation=translation,
    verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_priors : {best_priors}")
id = 1
plt.show()

lu.plot(mr_rigid)
plt.show()
#%%
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


# #%%
# best_momenta = {'affine_prior': torch.tensor([[-0.5799,  3.0117],
#         [-3.0049, -0.5558]]),
#                 'rot_prior': None,
#                 'trans_prior': torch.tensor([0.4199, 0.0598]),
#                 'scale_prior': None}
#%%
#####################################################
# Check the rigid optimisation
# print("")
# print("="*20)
# print("Check the rigid optimisation")
#
# print(f"best_momenta : {best_priors}")
# # param = best_priors.copy()
# momenta = mt.prepare_momenta(
#     source.shape,
#     diffeo = False,
#     # affine = True,
#     rotation=True, scaling=False, translation=True,
#     device = "cpu",
#     requires_grad = False,
#     **best_priors
# )
# print(f"best_priors : {best_priors}")
#
# print(f"momenta : {momenta}")
# mr_rigid.mp.debug = False
# mr_rigid.mp.forward(source, momenta.copy(), save =  True)
#
# lu.plot(mr_rigid)
# plt.show()



#%%

# sigmoid_a = 20
# sigmoid_b = 70
# sigmoid_c = -5
#
# iter = torch.linspace(0,100, 100)
# alpha = 2 * sigmoid_c /( sigmoid_b - sigmoid_a)
# beta = - (sigmoid_a + sigmoid_b) / 2
# g = alpha *( iter + beta)
# gamma = 1/(1 + torch.exp(-g))
#
# plt.plot(iter, gamma)
# plt.show()

#%% lddmm along rigid
#########################################################
# perfom lddmm along rigid
integration_steps = 10
sigma= [7, 15]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 1
cost_field_cst = 1
cost_affine_cst = 1
adam_dt_step_field=1e-6,
adam_dt_step_affine=1e-1,

verbose_datacost = False
plot_datacost = True
enable_grad_debug = False



saving_plots= pathlib.Path(
        ROOT_DIRECTORY +
       "/examples/results/rigid_meta_integrations/rigid_lddmm/" +
        f"decoupled_rigid_{modifier_str}_lddmm"
)

gamma_kwargs = {'c': 10, 'nu':.05}
n_iter = 150

kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(
        target.to("cuda:0"),
        gamma_mode = 'variationnal',
        gamma_kwargs = gamma_kwargs,
        normalize_ssd=False,
        verbose=verbose_datacost,
        plot=plot_datacost,
        save_plot=saving_plots,
        save_values=True
    )
# best_loss = torch.inf
# for i,param in enumerate(top_param_rot):
#     print(f"\n\noptimistion {i} on  {len(top_param_rot)}")

print("\n" + "=" * 20)
momenta = mt.prepare_momenta(
    source.shape,
    diffeo=True,
    rotation=rotation,
    scaling=scaling,
    translation=translation,
    device="cuda:0",
    **best_priors
)


mr = mt.affine_decoupled_along_metamorphosis(
  source, target, momenta_ini=momenta,
  kernelOperator= kernelOperator,
  rho = rho,
  data_term=datacost,
  integration_steps = integration_steps,
  cost_cst=cost_cst,
  cost_field_cst = cost_field_cst,
  cost_affine_cst = cost_affine_cst,
  n_iter=n_iter,
    grad_coef=.1,
    # optimizer_method='adadelta',
  # lbfgs_max_iter = 20,
  # lbfgs_history_size = 20,
  optimizer_method='Adam',
  adam_dt_step_field=adam_dt_step_field,
  adam_dt_step_affine=adam_dt_step_affine,
  save_gpu_memory=False,
    safe_mode=True,
    debug=False,
)


best = False
fig_cost, _ = mr.plot_cost()
fig_cost.savefig(str(saving_plots) + "_cost.png")
plt.show()
lu.plot(mr)
plt.show()
# if mr.data_loss < best_loss or mr.data_loss == 0:
#     print(param)
#     best_mr = mr

mt.free_GPU_memory(mr)

#%%
lu.frames_to_video_ffmpeg(
  frames_dir=saving_plots.parent,
  # frames_dir="examples/results/rigid_meta_integrations/rigid_lddmm",
  stem=saving_plots.name,
  fps=12,
)
#%%
file_save, path = mr.save(f"fishes_method_{modifier_str}-along_target_{target_name}",
        light_save=True,
        save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
        )

# best_mr.plot_cost()
# plt.show()

#%%
plot(mr)
# plt.show()


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

    deform = mr.mp.get_affine_deformation(deform)
    img = tb.imgDeform(
        mr.mp.image_stock[t, :, :, :][None],
        mr.mp.get_affine_deformator()
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
raise TypeError("GARUGA")
# fig.savefig(path + f"toyexample_star_{id}_integration.pdf")

#%%
def small_plot(self):
    affine = self.mp.get_affine_deformator()
    deform = self.mp.get_deformation()
    deform = self.mp.get_affine_deformation(deform)

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
# fig.savefig(path + f"toyexample_star_{id}_summary.pdf")

plt.show()
#%%
###########################################################
# Compare with pure LDDMM on a rigid output
# put the target on source
integration_steps = 10

kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target.to('cuda:0'),
                                gamma=1, normalize_ssd=False,
                                plot=False)
# datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)

mr_rigid_first = mt.affine_decoupled_along_metamorphosis(
    source, target, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='LBFGS_torch',
    cost_cst=.1,
    n_iter=0,
    lbfgs_max_iter=20
)

top_params = rg.initial_exploration(mr_rigid_first, r_step = 30,
                                    max_output = 4, verbose=True)
# top_params = None
print("top_params : ",top_params)

print("")
print("="*20)
print("Optimize on best exploration ")
best_loss, best_priors, best_rot = rg.optimize_on_rigid(
    mr_rigid_first, top_params,
    n_iter=50, grad_coef = .1,
    # affine=True,
    rotation=rotation, scaling=scaling, translation=translation,
    verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_priors : {best_priors}")

lu.plot(mr_rigid_first)
id = 1
momenta = mt.prepare_momenta(
    source.shape,
    diffeo = False,
    # affine = True,
    rotation=rotation, scaling=scaling, translation=translation,
    device = "cpu",
    requires_grad = False,
    **best_priors
)


print(f"momenta : {momenta}")
mr_rigid_first.mp.debug = False
mr_rigid_first.mp.forward(source, momenta.copy(), save =  True)

lu.plot(mr_rigid_first)
plt.show()
#%%
# source_lddmm = source.clone()
# target_lddmm = tb.imgDeform(target, mr_rigid_first.mp.get_affine_deformator())
# ref = "source"
# fig, ax = plt.subplots(1,3, constrained_layout=True)
# ax[0].imshow(source_lddmm[0,0],cmap='gray')
# ax[0].set_title("source")
# ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
# ax[2].imshow(target_lddmm[0,0],cmap='gray')
# ax[2].set_title("target")
# plt.show()
#%%%
source_lddmm = tb.imgDeform(source, mr_rigid_first.mp.get_affine_deformator())
target_lddmm = target.clone()
ref = 'target'
fig, ax = plt.subplots(1,3, constrained_layout=True)
ax[0].imshow(source_lddmm[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
ax[1].set_title("target")
ax[2].imshow(target[0,0],cmap='gray')
plt.show()
#%%
file_save, path = mr_rigid_first.save(f"fishes_method_{modifier_str}-successive-part1_target_{target_name}",
    light_save=True,
    save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
)
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
mr_l.plot_cost()
plt.show()

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
mr_l.mp.plot()
plt.show()
#%%
file_save, path = mr_l.save(f"fishes_method_{modifier_str}-successive-part2_target_{target_name}",
        light_save=True,
        save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
        )
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


mr = mt.affine_along_metamorphosis(
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
