import matplotlib.pyplot as plt

import demeter.utils.torchbox as tb
import demeter.metamorphosis.rotate as mtrt
import demeter.metamorphosis as mt
import demeter.utils.reproducing_kernels as rk

size = (200,200)
source = tb.reg_open('02', size=size)
target = tb.reg_open('02bis', size=size)

kernelOperator = rk.GaussianRKHS((1,1), normalized=False)

datacost = mt.Rotation_Ssd_Cost(target.to("cuda:0"), alpha=.5)

momenta = mtrt.prepare_momenta(
    # mr.source.shape,
    source.shape,
    image=False,
    rotation=True,
    translation=True,
    scaling=True,
    # scale_prior=[2,5]
)

rho = 1
mr = mt.rigid_along_metamorphosis(
    source, target, momenta_ini=momenta,
    kernelOperator= kernelOperator,
    rho = rho,
    data_term=datacost ,
    integration_steps = 10,
    cost_cst=1e-6,
    n_iter=5,
    save_gpu_memory=False,
    lbfgs_max_iter = 20,
    lbfgs_history_size = 20,
)
# mr.mp.forward(source, momenta)


# mr.plot_cost()
# plt.show()
#%%
grid = mr.mp.get_rigidor()
rot_source = tb.imgDeform(source, grid.detach().cpu())

fig, ax = plt.subplots(1,5)
ax[0].imshow(source[0,0], origin='lower', cmap='gray')
ax[0].set_title('Source')
ax[1].imshow(rot_source[0,0], origin='lower', cmap='gray')
ax[1].set_title("image")
ax[2].imshow(target[0,0], origin='lower', cmap='gray')
ax[2].set_title("target")
ax[3].imshow(tb.imCmp(target, rot_source, "compose")[0], origin='lower')
ax[3].set_title("image vs target")
ax[4].imshow(tb.imCmp(source, rot_source, "compose")[0], origin='lower')
ax[4].set_title("image vs source")

plt.show()