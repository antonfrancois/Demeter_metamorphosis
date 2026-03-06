import __init__
import torch
from math import cos,sin
import matplotlib.pyplot as plt

import demeter.utils.torchbox as tb
import demeter.metamorphosis.affine as mtrt
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.rigid_exploration as rg
from demeter.constants import set_ticks_off, GRIDDEF_YELLOW,  ROOT_DIRECTORY
from demeter.utils.cost_functions import SumSquaredDifference


#
def smooth(image, sigma):
    if isinstance(sigma, int):
        sigma = (sigma,sigma)
    kernel = rk.GaussianRKHS(sigma).kernel

    return rk.fft_filter(image,kernel,border_type='constant')

def plot(self):
    affine = self.mp.get_affine_deformator()
    deform = self.mp.get_deformation()
    deform = self.mp.get_affine_deformation(deform)

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

def _norm(x):
    return torch.linalg.norm(x.reshape(-1)).item()

def install_momenta_grad_debug(momenta, every=10, max_print=60):
    counters = {k: 0 for k in momenta.keys()}
    handles = []

    for key, tensor in momenta.items():
        if not tensor.requires_grad:
            continue

        def _hook(grad, _k=key):
            counters[_k] += 1
            c = counters[_k]
            if c <= max_print and (c == 1 or c % every == 0):
                gnorm = _norm(grad.detach())
                gmax = grad.detach().abs().max().item()
                print(f"[grad:{_k}] call={c:03d} |g|={gnorm:.4e} |g|max={gmax:.4e}")

        handles.append(tensor.register_hook(_hook))

    return handles

def print_momenta_delta(before, after):
    print("\n[Momenta delta summary]")
    for key in sorted(before.keys()):
        if key not in after:
            print(f"  - {key}: absent in final momenta")
            continue
        b = before[key].detach()
        a = after[key].detach()
        d = a - b
        bn = _norm(b)
        dn = _norm(d)
        rel = dn / (bn + 1e-12)
        print(f"  - {key}: |before|={bn:.4e} |delta|={dn:.4e} rel_delta={rel:.4e}")

def summarize_registration_case(name, mr, target):
    ssd_fn = cf.SumSquaredDifference(target)
    grid_rt = mr.mp.get_affine_deformator()
    rotated_image = tb.imgDeform(mr.mp.image, grid_rt, dx_convention='2square')
    rotated_source = tb.imgDeform(mr.source, grid_rt, dx_convention='2square')

    final_data_loss = float(mr.to_analyse[1]["data_loss"][-1])
    ssd = float(ssd_fn(rotated_image))
    ssd_rot = float(ssd_fn(rotated_source))

    rot_mat = mr.mp.rot_mat.detach()
    theta_deg = torch.atan2(rot_mat[1, 0], rot_mat[0, 0]).item() * 180.0 / torch.pi

    t = mr.mp.translation.detach()
    h, w = mr.source.shape[-2:]
    tx_pix = t[0].item() * (w - 1) / 2.0
    ty_pix = t[1].item() * (h - 1) / 2.0

    print(f"\n[{name}]")
    print(f"  - final_data_loss: {final_data_loss:.6f}")
    print(f"  - ssd(image->target): {ssd:.6f}")
    print(f"  - ssd(source_rigid->target): {ssd_rot:.6f}")
    print(f"  - angle_deg: {theta_deg:.4f}")
    print(f"  - translation_2square: [{t[0].item():.6f}, {t[1].item():.6f}]")
    print(f"  - translation_pixels:  [{tx_pix:.3f}, {ty_pix:.3f}]")

    if isinstance(mr.to_analyse[0], dict):
        p = mr.to_analyse[0]
        if "momentum_R" in p:
            print(f"  - |momentum_R|: {_norm(p['momentum_R']):.6e}")
        if "momentum_T" in p:
            print(f"  - |momentum_T|: {_norm(p['momentum_T']):.6e}")
        if "momentum_I" in p:
            print(f"  - |momentum_I|: {_norm(p['momentum_I']):.6e}")

    return {
        "final_data_loss": final_data_loss,
        "ssd": ssd,
        "ssd_rot": ssd_rot,
        "theta_deg": theta_deg,
        "tx_pix": tx_pix,
        "ty_pix": ty_pix,
    }

path = "examples/results/rigid_meta/"
device = "cuda:0"
###########################################################
# open images
size = (300, 300)
# source = tb.reg_open('rigid_s',size=size)
# target = tb.reg_open('rigid_t',size=size)
source = tb.reg_open('33',size=size)
target = tb.reg_open('fish',size=size)
# source = tb.reg_open('20',size=size)
# target = tb.reg_open('17',size=size)



source.to(device)
# source = smooth(source, 20)
# target = smooth(target, 20)

fig, ax = plt.subplots(1,3)
ax[0].imshow(source[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(target[0,0],cmap='gray')
ax[1].set_title("target")
ax[2].imshow(tb.imCmp(source,target, 'compose')[0])
# plt.show()

# Align barycenters

source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)


ssd  = SumSquaredDifference(target_b)
print("ssd target_b - source_b :",ssd(source_b))

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
print("")
print("="*20)
print("Initial exploration")
integration_steps = 10



kernelOperator = rk.DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'),
                                gamma=1, normalize_ssd=False,
                                plot=False)
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
    lbfgs_max_iter=20
)

top_params = rg.initial_exploration(mr_rigid, r_step = 20,
                                    max_output = 1, verbose=True)
print("top_params : ",top_params)
#
# print("")
print("="*20)
print("Optimize on best exploration ")
best_loss, best_priors, best_rot = rg.optimize_on_rigid(
    mr_rigid, top_params,
    n_iter=10, grad_coef = .1,
    affine=True,
    # rotation=True, scaling=False, translation=True,
    verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_priors : {best_priors}")
id = 1

best_priors = {'affine_prior': torch.tensor([[-0.3171, -1.0310],
        [ 0.8974, -0.6376]]),
               'rot_prior': None,
               'trans_prior': torch.tensor([-0.0269,  0.0821]), 'scale_prior': None}



#%%
#####################################################
# Check the rigid optimisation
# print("")
# print("="*20)
# print("Check the rigid optimisation")
# input("Press Enter to continue")
#
# print(f"best_momenta : {best_priors}")
# param = best_priors.copy()
# momenta = mt.prepare_momenta(
#     source_b.shape,
#     diffeo = False,
#     affine = True,
#     device = "cpu",
#     requires_grad = False,
#     **param
# )
# print(f"best_priors : {best_priors}")
#
# print(f"momenta : {momenta}")
# mr_rigid.mp.debug = False
# mr_rigid.mp.forward(source_b, momenta.copy(), save =  True)
#
# plot(mr_rigid)
# plt.show()



#%%
sigmoid_a = 20
sigmoid_b = 70
sigmoid_c = -5

iter = torch.linspace(0,100, 100)
alpha = 2 * sigmoid_c /( sigmoid_b - sigmoid_a)
beta = - (sigmoid_a + sigmoid_b) / 2
g = alpha *( iter + beta)
gamma = 1/(1 + torch.exp(-g))

plt.plot(iter, gamma)
plt.show()
#%% lddmm along rigid
#########################################################
# perfom lddmm along rigid
integration_steps = 10

print("")
print("="*20)
print("Start real optimization")
# input("Press Enter to continue")
sigma= [  7, 10]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 1
cost_field_cst = 1
cost_affine_cst = 1
adam_dt_step_field=1e-6,
adam_dt_step_affine=1e-2,

verbose_datacost = False
plot_datacost = True
enable_grad_debug = False

saving_plots= (
        ROOT_DIRECTORY +
       "/examples/results/rigid_meta_integrations/affine_lddmm/" +
        f"general_affine_lddmm"
)


kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(
        target_b.to("cuda:0"),
        # gamma=alpha,
        sigmoid_a=sigmoid_a,sigmoid_b=sigmoid_b,sigmoid_c=sigmoid_c,
        normalize_ssd=False,
        verbose=verbose_datacost,
        plot=plot_datacost,
        save_plot=saving_plots
    )
# datacost = mt.Ssd(target_b.to("cuda:0"))

# best_loss = torch.inf
# for i,param in enumerate(top_param_rot):
#     print(f"\n\noptimistion {i} on  {len(top_param_rot)}")
print("\n" + "=" * 20)
momenta = mt.prepare_momenta(
    source_b.shape,
    diffeo=True,
    affine=True,
    device="cuda:0",
    **best_priors
)
# momenta["momentum_A"].requires_grad = False
# momenta["momentum_T"].requires_grad = False

for k,v in momenta.items():
    print(k, v.requires_grad)


mr = mt.rigid_along_metamorphosis(
  source_b, target_b, momenta_ini=momenta,
  kernelOperator= kernelOperator,
  rho = rho,
  data_term=datacost,
  integration_steps = integration_steps,
  cost_cst=cost_cst,
  cost_field_cst = cost_field_cst,
  cost_affine_cst = cost_affine_cst,
  n_iter=10,
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


# print("\n" + "=" * 20)
# print("[Comparison summary]")
# for key in ["final_data_loss", "ssd", "ssd_rot", "theta_deg", "tx_pix", "ty_pix"]:
#     a = case_results["with_I"]["summary"][key]
#     b = case_results["without_I"]["summary"][key]
#     print(f"  - {key}: with_I={a:.6f} | without_I={b:.6f}")



best = False
fig_cost, _ = mr.plot_cost()
fig_cost.savefig(saving_plots + "_cost.png")
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
target_lddmm = tb.imgDeform(target_b, mr_rigid_first.mp.get_affine_deformator())
ref = "source"
fig, ax = plt.subplots(1,3, constrained_layout=True)
ax[0].imshow(source_lddmm[0,0],cmap='gray')
ax[0].set_title("source")
ax[1].imshow(tb.imCmp(source_lddmm,target_lddmm,'compose')[0])
ax[2].imshow(target_lddmm[0,0],cmap='gray')
ax[2].set_title("target")
plt.show()
#%%%
source_lddmm = tb.imgDeform(source_b, mr_rigid.mp.get_affine_deformator())
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
