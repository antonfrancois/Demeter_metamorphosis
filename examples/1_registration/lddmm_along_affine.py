

import __init__
import sys, pathlib


sys.path.insert(0, str(pathlib.Path("examples/1_registration").resolve()))

import torch
import matplotlib.pyplot as plt


# from kornia.geometry.transform import resize

import demeter.utils.torchbox as tb
import demeter.metamorphosis.affine as mtrt
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.rigid_exploration as rg
from demeter.constants import set_ticks_off, GRIDDEF_YELLOW,  ROOT_DIRECTORY
from demeter.utils.cost_functions import SumSquaredDifference
import demeter.utils.bspline as bs
import lddmm_along_utils as lu

#
def smooth(image, sigma):
    if isinstance(sigma, int):
        sigma = (sigma,sigma)
    kernel = rk.GaussianRKHS(sigma).kernel

    return rk.fft_filter(image,kernel,border_type='constant')



#%%
# FISHIES
# # /home/turtlefox/Documents/11_metamorphoses/Demeter_metamorphosis/examples/im2Dbank/fish_bass_1.png
# source_rgb, source_g, source = lu.open_fish("bass_1", factor=.3)
# target_rgb, target_g, target = lu.open_fish("crappie_1", factor=.3)
#
# # from kornia.filters import bilateral_blur
# # source = bilateral_blur(source, 5, sigma_color=5, sigma_space=(5, 5))
# # target =  bilateral_blur(target, 5, sigma_color=5, sigma_space=(5, 5))
# source = source.to(torch.double)
# #
# rot_mat = tb.create_rot_mat_2d(torch.tensor(3./torch.pi))
# # rot_mat = torch.eye(2)
# rot_mat[1,1] /= 1.5
# trans = torch.tensor([0.2, -0.2])
# id_grid = tb.make_regular_grid((1,)+source.shape[2:]+ (2,), dx_convention='2square')
# grid = tb.grid_from_rotation_translation(id_grid, rot_mat, trans)
# target = tb.imgDeform(target, grid).to(torch.double)
# target_rgb = target_rgb[0,...,:-1].transpose(2, 3).transpose(1, 2)
# target_rgb_r = tb.imgDeform(target_rgb, grid).transpose(1, 2).transpose(2, 3)
#
#
# fig, ax = plt.subplots(1,3, constrained_layout=True)
# ax[0].imshow(source[0,0], cmap='gray')
# ax[1].imshow(target_rgb_r[0], cmap='gray')
# ax[2].imshow(tb.imCmp(source,target, 'seg')[0])
# plt.show()
# raise ValueError("kvnlk")

#%%
path = "examples/results/rigid_meta/"
device = "cuda:0"
# ###########################################################
# # open images
size = (300, 300)
# # source = tb.reg_open('rigid_s',size=size)
# # target = tb.reg_open('rigid_t',size=size)
source = tb.reg_open('minifish',size=size)
target_d = tb.reg_open('fish',size=size)
source_r = lu.open_rainbow_minifish()
# source = tb.reg_open('20',size=size)
# target = tb.reg_open('17',size=size)
# source = tb.reg_open('34',size=size)
# target = tb.reg_open('35',size=size)


source.to(device)
# source = smooth(source, 20)
# target = smooth(target, 20)

fig, ax = plt.subplots(1,3)
ax[0].imshow(source[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(target_d[0,0],cmap='gray')
ax[1].set_title("target")
ax[2].imshow(tb.imCmp(source,target_d, 'compose')[0])
plt.show()

#%% toy example L all deforms
#
# size = (300, 300)
# source = tb.reg_open('30',size=size)
# # source[source>.5] = 1
#
# # cms = bs.getCMS_turn()
# # cms = bs.getCMS_allcombinaision()
# torch.manual_seed(3)
# # print(torch.seed())
# cms = torch.rand((2,9,9)) - 0.5
# # cms = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., -0.05, -.2, 0., 0., 0., 0.],
# #          [0., -0.1, -0.3, -0.4, -0.5, -0., 0., 0., 0.],
# #          [0., 0., 0., 0., -0.2, 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
# #         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., .3, 0., 0.],
# #          [0., 0., 0., 0., 0., 0., .2, 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
#
# field = bs.field2D_bspline(cms, n_pts=size, degree=(2,2), dim_stack=-1)
# id_grid = tb.make_regular_grid((1,)+source.shape[2:]+ (2,), dx_convention='2square')
# deform = id_grid + field /5
# target_d = tb.imgDeform(source, deform)
#
#
# origin = 'lower'
# fig,ax = plt.subplots(2,2)
# ax[0,0].imshow(source[0,0], cmap='gray',origin=origin)
# ax[0,0].set_title('Source')
# tb.gridDef_plot_2d(deform, ax = ax[1,0], step = 10, check_diffeo=False)
# ax[1,0].set_title('Random spline deformation')
# ax[0,1].imshow(target_d[0,0], cmap='gray', origin=origin)
# ax[0,1].set_title('source deformed')
# ax[1,1].imshow(tb.imCmp(source,target_d, "segw")[0], cmap='gray',origin=origin)
# plt.show()
#%%
# A_full = torch.tensor([
#     [.563, -0.791],
#     [0.710,  0.117]
# ], dtype=target_d.dtype, device=target_d.device)
#
# fig, axes, sliders = lu.affine_matrix_slider_figure(
#     image=target_d,   # must be [1,C,H,W]
#     A_init=A_full,
#     slider_delta=1.5,
#     mode="bilinear",
#     clamp=False,
# )
#
# plt.show()
# raise TypeError("c'est toi le type")
#%%
theta = torch.tensor([5*torch.pi/8])              # radians
translation = torch.tensor([[0.12, -0.13]]) # in 2square coords
scale = torch.tensor([.8])

# A_full = torch.tensor([
#     [1.240, -0.032],
#     [0.449,  0.614]
# ], dtype=target_d.dtype)
A_full = torch.tensor([
    [0.417, -1.3],
    [0.873,  -0.393]
], dtype=target_d.dtype)

res = lu.apply_registration_models(
    target_d,
    rotation=theta,
    translation=translation,
    scale=scale,
    full_affine=A_full,
)
keys = ["full_affine","rotation_translation_scaling","rotation_translation","rotation_scaling"]
lu.show_deforms(target_d, res, keys)

target_name = keys[0]
target = res[target_name]["image"]
print(f"target used : {target_name}")



# %%
# print("")
# print("="*20)
# print("Initial exploration")
# integration_steps = 10
#
#
#
# kernelOperator = rk.DummyKernel()
#
# datacost = mt.Rotation_Ssd_Cost(target.to('cuda:0'),
#                                 gamma=1, normalize_ssd=False,
#                                 plot=False)
# # datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)
#
# mr_rigid = mt.affine_along_metamorphosis(
#     source, target, momenta_ini=0,
#     kernelOperator= kernelOperator,
#     rho = 1,
#     data_term=datacost ,
#     integration_steps = integration_steps,
#     # optimizer_method='LBFGS_torch',
#     optimizer_method="Adam",
#     adam_dt_step_affine= 1e-2,
#     cost_cst=.1,
#     n_iter=0,
#     lbfgs_max_iter=20
# )
#
# top_params = rg.initial_exploration(mr_rigid, r_step = 50,
#                                     max_output =6, verbose=True)
# print("top_params : ",top_params)
# #
# # print("")
# print("="*20)
# print("Optimize on best exploration ")
# best_loss, best_priors, best_rot = rg.optimize_on_rigid(
#     mr_rigid, top_params,
#     n_iter=150, grad_coef = .001,
#     affine=True,
#     # rotation=True, scaling=False, translation=True,
#     verbose=True, plot = True,
# )
# print(f"best_loss : {best_loss}")
# print(f"best_rot : {best_rot}")
# print(f"best_priors : {best_priors}")
# id = 1
#
# # best_priors = {'affine_prior': torch.tensor([[-0.3171, -1.0310],
# #         [ 0.8974, -0.6376]]),
# #                'rot_prior': None,
# #                'trans_prior': torch.tensor([-0.0269,  0.0821]), 'scale_prior': None}
#


#%% lddmm along rigid
#########################################################
# perfom lddmm along rigid
integration_steps = 10

print("")
print("="*20)
print("Start real optimization")
# input("Press Enter to continue")
sigma= [ 5, 7, 15,]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 1
cost_field_cst = 1
cost_affine_cst = 1
adam_dt_step_field=1e-6,
adam_dt_step_affine=5e-2,

verbose_datacost = False
plot_datacost = False
enable_grad_debug = False

saving_plots= pathlib.Path(
        ROOT_DIRECTORY +
       "/examples/results/rigid_meta_integrations/affine_lddmm/" +
        f"general_affine_lddmm"
)

# gamma_kwargs = {"sigmoid_a":sigmoid_a,"sigmoid_b":sigmoid_b,"sigmoid_c":sigmoid_c}
gamma_kwargs = {'c': 10, 'nu':.05}
n_iter = 500

kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(
        target.to("cuda:0"),
        gamma_mode = 'variationnal',
        gamma_kwargs = gamma_kwargs,
        normalize_ssd=False,
        verbose=verbose_datacost,
        plot=plot_datacost,
        save_plot=saving_plots,
        save_values=True,
        edges_computes=1e-2
    )
# datacost = mt.Ssd(target_b.to("cuda:0"))

best_priors = {'rot_prior': torch.tensor(1.6029)}
print("\n" + "=" * 20)
momenta = mt.prepare_momenta(
    source.shape,
    diffeo=True,
    affine=True,
    device="cuda:0",
    **best_priors
)

# momenta["momentum_A"].requires_grad = False
# momenta["momentum_T"].requires_grad = False

for k,v in momenta:
    print(k, v.requires_grad)


mr = mt.affine_along_metamorphosis(
  source, target, momenta_ini=momenta,
  kernelOperator= kernelOperator,
  rho = rho,
  data_term=datacost,
  integration_steps = integration_steps,
  cost_cst=cost_cst,
  cost_field_cst = cost_field_cst,
  cost_affine_cst = cost_affine_cst,
  n_iter=n_iter,
    convergence_tol=2e-3,
    convergence_patience=3,
    grad_coef=.1,
    # optimizer_method='LBFGS_torch',
  # lbfgs_max_iter = 20,
  # lbfgs_history_size = 20,
  optimizer_method='Adam',
  adam_dt_step_field=adam_dt_step_field,
  adam_dt_step_affine=adam_dt_step_affine,
    adam_scheduler="reduce_on_plateau",
  save_gpu_memory=False,
    safe_mode=True,
    debug=False,
)


# print("\n" + "=" * 20)
# print("[Comparison summary]")
# for key in ["final_data_loss", "ssd", "ssd_rot", "theta_deg", "tx_pix", "ty_pix"]:
#     a = case_results["with_I"]["summary"][key]
#     b = case_results["without_I"]["summary"][key]
#     print(f"  - {key}: with_I={a:.6f} | without_I={b:.6f}")

#%%
best = False
fig_cost, _ = mr.plot_cost()
# fig_cost.savefig(str(saving_plots) + "_cost.svg")
# fig_cost.savefig(str(saving_plots) + "_cost.pdf")
plt.show()
fig,  ax= mr.data_term.plot_cost_data_term()
ax.set_xlabel("iterations")
# fig.savefig(str(saving_plots) + "_var_cost.svg")
# fig.savefig(str(saving_plots) + "_var_cost.pdf")
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



#%%
file_save, path = mr.save(f"fishes_method_affine-along_target_{target_name}",
        light_save=True,
        save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
        )

# best_mr.plot_cost()
# plt.show()

#%%
# deformator = mr.mp.get_deformator()
# deformation = mr.mp.get_deformation()
#
#
# source_rgb_t = source_rgb[0,...,:-1].transpose(2, 3).transpose(1, 2)
# source_rgb_deform = tb.imgDeform(source_rgb_t, deformator).transpose(1, 2).transpose(2, 3)
#
# affine = mr.mp.get_affine_deformator()
# # full = mr.mp.get_affine_deformator(affine)
# source_r = tb.imgDeform(source_b, affine)#.transpose(1, 2).transpose(2, 3)
#
# fig, ax = plt.subplots(2,2, constrained_layout=True)
# ax[0, 0].imshow(source_rgb[0,0])
# ax[0, 0].set_title("Source RGB")
# ax[0, 1].imshow(target_rgb[0])
# ax[0, 1].set_title("Target RGB")
# ax[1,0].imshow(source_rgb_deform[0])
# ax[1,0].set_title("Source RGB Deform")
# # ax[1, 1].imshow(source_r[0,0], cmap='gray')
# # ax[1,1].imshow(tb.imCmp(source_r,target_b, 'seg')[0])
# # ax[1,1].set_title("source vs target")
# tb.gridDef_plot_2d(deformation, ax = ax[1,1], step = 30)
# plt.show()

#%%

n_figs = 6
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
momentum =  mr.mp.momenta.momentum_I
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
    4,
    n_figs,
    constrained_layout=True,
    figsize=(
        n_figs * size_fig,
        size_fig * 4,
    ),
)
for i, t in enumerate(plot_id):
    deform = mr.mp.get_deformation(to_t = t + 1)
    deformator = mr.mp.get_deformator(to_t = t + 1)

    # raw_img = mr.mp.image_stock[t, 0, :, :].detach().numpy()
    # --- Pour couleur ---
    raw_img = source_rgb[0,...,:-1].transpose(2, 3).transpose(1, 2)
    raw_img = tb.imgDeform(raw_img, deformator).transpose(1, 2).transpose(2, 3)[0]
    # --- End pour couleur ---
    i_s = ax[ 0,i].imshow(
        raw_img,
        **kw_image_args,
    )
    # ax[ 0,i].set_ylabel("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    ax[0,i].set_title("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    # fig.colorbar(i_s, ax=ax[ 0,i], fraction=0.046, pad=0.04)

    ax[ 1,i].imshow(
        raw_img,
        alpha = .5,**kw_image_args,
    )
    tb.gridDef_plot_2d(
        deform,
        add_grid=False,
        ax=ax[ 1,i],
        step=int(min(mr.mp.field_stock.shape[2:-1]) / 25),
        check_diffeo=False,
        # dx_convention=mr.mp.dx_convention,
        color = "green",
        alpha=.5,
    )

    deform = mr.mp.get_affine_deformation(deform,to_t=t)
    #  --- Pour couleur -----
    raw_img = source_rgb[0,...,:-1].transpose(2, 3).transpose(1, 2)
    # --- end pour couleur
    img = tb.imgDeform(
        raw_img,
        mr.mp.get_affine_deformator(to_t=t)
    ).detach().numpy()
    print(img.shape)
    img = img.transpose(1, 2).transpose(2, 3)[0]
    print(img.shape)

    ax[ 2,i].imshow(img, **kw_image_args)
    # ax[ 2,i].set_title("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    ax[ 2,i].axis("off")

    # ax[ 3,i].imshow(torch.rand((10,10)), **kw_image_args,)
    # ax[ 3,i].set_title("t = " + str((t / (mr.mp.n_step - 1)).item())[:3])
    # ax[ 3,i].axis("off")
    # # fig.colorbar(i_s, ax=ax[ 0,i], fraction=0.046, pad=0.04)

    #
    tb.gridDef_plot_2d(
        deform,
        add_grid=False,
        add_markers=True,
        ax=ax[ 3,i],
        step=int(min(mr.mp.field_stock.shape[2:-1]) / 15),
        check_diffeo=False,
        dx_convention=mr.mp.dx_convention,
        origin="upper",
    )
    ax[ 3,i].set_aspect('auto')


set_ticks_off(ax)
plt.show()
# raise TypeError("GARUGA")
fig.savefig(saving_plots.parent / f"toyexample_fish_integration.pdf")

#%%
# def small_plot(self):
#     affine = self.mp.get_affine_deformator()
#     deform = self.mp.get_deformation()
#     deform = self.mp.get_affine_deformation(deform)
#
#     img_rot = tb.imgDeform(self.mp.image.to('cpu'),affine,dx_convention='2square')
#     source_rt = tb.imgDeform(self.source.to('cpu'),affine,dx_convention='2square')
#     srt = tb.imCmp(source_rt,target_b,method = 'seg')
#     irt = tb.imCmp(img_rot,target_b,method = 'seg')
#     kwargs = {'cmap': "gray"}
#
#     fig,ax = plt.subplots(2,2, constrained_layout=True, figsize = (5,5))
#
#     ax[0,0].imshow(img_rot[0,0], **kwargs)
#     ax[0,0].set_title("(a) Registered image")
#     ax[0,1].imshow(source_rt[0,0], **kwargs)
#     ax[0,1].set_title("(b) affine on source")
#
#     ax[1,0].imshow(irt[0], **kwargs)
#     ax[1,0].set_title("(c) registered vs Target")
#     ax[1,1].imshow(srt[0], **kwargs)
#     ax[1,1].set_title("(d) source affine vs Target")
#     set_ticks_off(ax)
#     return fig
# fig  = small_plot(mr)
# # fig.savefig(path + f"toyexample_star_{id}_summary.pdf")
#
# plt.show()
#%%


###########################################################
# Compare with pure LDDMM on a rigid output

# put the target on source
integration_steps = 10

kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target.to('cuda:0'), gamma=1)
# datacost = mt.Rotation_MutualInformation_Cost(target.to('cuda:0'), alpha=1)


mr_rigid_first = mt.affine_along_metamorphosis(
    source, target, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    # optimizer_method='LBFGS_torch',
    optimizer_method="Adam",
    adam_dt_step_affine= 1e-3,
    cost_cst=.1,
    n_iter=0,
    lbfgs_max_iter=20
)

# top_params = rg.initial_exploration(mr_rigid_first,r_step=50, max_output = 10, verbose=True)
# top_params = None
# best_loss, best_prior, best_rot = rg.optimize_on_rigid(
#     mr_rigid_first, top_params, n_iter=150,verbose=True, plot = True,affine=True,
# )

best_priors = {'rot_prior': torch.tensor(1.6029)}

id = 1
momenta = mt.prepare_momenta(
    source.shape,
    diffeo = False,device = "cuda:0",requires_grad = True,
    affine = True,
    **best_priors
)
# def _detach(p):
#     return {k: v.detach() for k, v in p.items()}.copy()
print(f"best_priors : {best_priors}")
mr_rigid_first.forward(momenta, n_iter=1000)

mr_rigid_first.plot_cost()
plt.show()
lu.plot(mr_rigid_first)
plt.show()
#%%
source_lddmm = tb.imgDeform(source.cpu(), mr_rigid_first.mp.get_affine_deformator().cpu(), dx_convention='2square')
target_lddmm = target.clone()
ref = "target"
fig, ax = plt.subplots(1,3, constrained_layout=True)
ax[0].imshow(source_lddmm[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
ax[2].imshow(target_lddmm[0,0],cmap='gray')
ax[2].set_title("target")
plt.show()

#%%
file_save, path = mr_rigid_first.save(f"fishes_method_affine-successive-part1_target_{target_name}",
    light_save=True,
    save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
)


#%%%
# source_lddmm = tb.imgDeform(source, mr_rigid_first.mp.get_affine_deformator().cpu())
# target_lddmm = target.clone()
# ref = 'target'
# fig, ax = plt.subplots(1,3, constrained_layout=True)
# ax[0].imshow(source_lddmm[0,0],cmap='gray')
# ax[0].set_title("source")
# ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
# ax[1].set_title("target")
# ax[2].imshow(target_b[0,0],cmap='gray')
# plt.show()

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
file_save, path = mr_l.save(f"fishes_method_affine-successive-part2_target_{target_name}",
        light_save=True,
        save_path = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm"
        )

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
