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
    srt = tb.imCmp(source_rt,target_b,method = 'compose')
    irt = tb.imCmp(img_rot,target_b,method = 'compose')
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

#%% Align barycenters

source_b, target_b, trans_s, trans_t = rg.align_barycentres(source, target, verbose=True)
fig, ax = plt.subplots(1,3)
ax[0].imshow(source_b[0,0],cmap='gray')
ax[0].set_title("source barycentred")
ax[1].imshow(target_b[0,0],cmap='gray')
ax[1].set_title("target barycentred")
ax[2].imshow(tb.imCmp(source_b,target_b, 'compose')[0])
plt.show()

# # %%
integration_steps = 10

class DummyKernel:
    def __call__(self, x):
        return x  # Identité pour test simple

    def init_kernel(self, image):
        pass

kernelOperator = DummyKernel()

datacost = mt.Rotation_Ssd_Cost(target_b.to('cuda:0'), alpha=1)
# datacost = mt.Rotation_MutualInformation_Cost(target_b.to('cuda:0'), alpha=1)

mr_rigid = mt.rigid_along_metamorphosis(
    source_b, target_b, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='LBFGS_torch',
    cost_cst=.1,
    n_iter=0
)

top_params = rg.initial_exploration(mr_rigid,r_step=10, max_output = 10, verbose=True)
print(top_params)
#%%
best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
    mr_rigid, top_params, n_iter=10,verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_momenta : {best_momenta}")
#%%

best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
    mr_rigid, [top_params[-1]], n_iter=2,verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_momenta : {best_momenta}")
# {'rot_prior': torch.tensor(-1.0472), 'trans_prior': None, 'scale_prior': None}
#%%
mr_rigid = mt.rigid_along_metamorphosis(
    source_b, target_b, momenta_ini=0,
    kernelOperator= kernelOperator,
    rho = 1,
    data_term=datacost ,
    integration_steps = integration_steps,
    optimizer_method='Adam',
    cost_cst=.1,
    n_iter=0
)

param = [(best_loss, best_momenta)]
best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
    mr_rigid, param, n_iter=10,verbose=True, plot = True,
)
print(f"best_loss : {best_loss}")
print(f"best_rot : {best_rot}")
print(f"best_momenta : {best_momenta}")

#%%
# r_step = 10
# top_param_rot = [( best_loss, dict(
#                      rot_prior=r.detach(),
#                 trans_prior=None,
#                 scale_prior=None,)) for r in torch.linspace(-torch.pi, torch.pi  , r_step)]
#
# best_loss, best_momenta, best_rot = rg.optimize_on_rigid(
#     mr_rigid, top_param_rot, n_iter=5,verbose=False, plot=True
# )
# print(f"best_loss : {best_loss}")
# print(f"best_momenta : {best_momenta}")

#%%

#%% Check rigid search
mr_rigid.plot()
#%%
print(f"best_momenta : {best_momenta}")
param = best_momenta.copy()
momenta = mt.prepare_momenta(
    source_b.shape,
    diffeo = False,
    device = "cpu",
    requires_grad = False,
    **param
)
print(f"best_momenta : {best_momenta}")

print(f"momenta : {momenta}")

mr_rigid.mp.forward(source_b, momenta.copy(), save =  True)

plot(mr_rigid)
plt.show()


#%% lddmm

sigma= [  7, 15]
sigma = [(s,)*2 for s in sigma]
alpha = .5
rho = 1
cost_cst = 10
cst_field = 1

kernelOperator = rk.Multi_scale_GaussianRKHS(sigma, normalized=False, kernel_reach =6)
datacost = mt.Rotation_Ssd_Cost(target_b.to("cuda:0"), alpha=alpha)


# best_loss = torch.inf
# for i,param in enumerate(top_param_rot):
#     print(f"\n\noptimistion {i} on  {len(top_param_rot)}")
momenta = mt.prepare_momenta(
    source_b.shape,
    rotation=True,scaling=True,translation=True,
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
best_momentum_I = mr.to_analyse[0]["momentum_I"]
best_momentum_R = mr.to_analyse[0]["momentum_R"]
best_momentum_T = mr.to_analyse[0]["momentum_T"]
best_momentum_S = mr.to_analyse[0]["momentum_S"]
#%%
plot(best_mr)
plt.show()
#%%
mr.mp.plot()
plt.show()