import __init__
import torch
from math import cos,sin
import matplotlib.pyplot as plt

import demeter.utils.torchbox as tb
import demeter.metamorphosis.rotate as mtrt
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf


# # %%
def create_rot_mat(theta):
    return torch.tensor([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]],
                        dtype=torch.float)

def apply_rot_mat(grid,rot_mat):
    rotated_grid = torch.einsum('ij,bhwj->bhwi',rot_mat, grid)
    return rotated_grid

def make_exp(xxx, yyy, centre, sigma):
    ce_x, ce_y = centre
    sigma_x,sigma_y = sigma
    exp = torch.exp(
        - 0.5*((xxx - ce_x) / sigma_x) ** 2
        - 0.5*((yyy - ce_y) / sigma_y) ** 2)
    return exp

img = tb.reg_open('23',size=(200,200))
id_grid = tb.make_regular_grid(img.shape[2:],dx_convention='2square')

xx,yy = id_grid[...,0].clone(),id_grid[...,1].clone()
xx -= .2* make_exp(xx,yy,(0.25,0.25),(0.1,0.1))
yy -= (.2* make_exp(xx,yy,(0,-0.25),(0.15,0.15)))

deform = torch.stack([xx,yy],dim = -1)
deform = id_grid.clone()

new_img = tb.imgDeform(img,deform,dx_convention='2square')
# fig,ax = plt.subplots(1,1)
# ax.imshow(new_img[0,0])
# tb.gridDef_plot(deform,ax = ax,step=30,dx_convention='2square',color='orange')
# plt.show()

## %%
theta = -torch.pi/3
rot = create_rot_mat(theta)
rot_grid = apply_rot_mat(id_grid,rot)
newimg_r = tb.imgDeform(new_img,rot_grid,dx_convention='2square')

#apply rot to id_grid

# tb.gridDef_plot(rot_grid,step=30)
# newimg_r += torch.randn_like(newimg_r)*0.1
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(img[0,0])
# ax[1].imshow(new_img[0,0])
# ax[2].imshow(newimg_r[0,0])
# plt.show()

#%%
print("images made, starting metamorphosis")

# dx = tuple([2./(s-1) for s in img.shape[2:]])
# s=0.15
# sigma = (s,s)
# # sigma = (3,9)
# kernelOperator = rk.VolNormalizedGaussianRKHS(
#         sigma=sigma,
#         sigma_convention='continuous', # by passing 'continuous' we specify that sigma is continuous and compute sigma_pixel using dx
#         dx=dx,
#         border_type='constant'
#     )

kernelOperator = rk.GaussianRKHS(sigma=(10,10),normalized=False)
kernelOperator = rk.VolNormalizedGaussianRKHS(
    sigma=(10,10),
    sigma_convention='pixel',
    dx=(1,1),
)

print(kernelOperator)

class Rotation_Ssd_Cost(mt.DataCost):

    def __init__(self, target, alpha, **kwargs):

        super(Rotation_Ssd_Cost, self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)
        self.alpha = alpha

    def set_optimizer(self, optimizer):
        """
        DataCost object are meant to be used along a
        method inherited from `Optimize_geodesicShooting`.
        This method is used to set the optimizer object and is usually
        used at the optimizer initialisation.
        """
        self.optimizer = optimizer
        # try:
        #     self.optimizer.mp.rot_mat
        # except AttributeError:
        #     raise AttributeError(f"The optimizer must have Rotation implemented, optimizer is {optimizer.__class__.__name__}")
        if self.target.shape != self.optimizer.source.shape and not self.target is None:
            raise ValueError(
                "Target and source shape are different."
                f"Got source.shape = {self.optimizer.source.shape}"
                f"and target.shape = {self.target.shape}."
                f"Have you checked your DataCost initialisation ?"
            )



    def __call__(self,at_step=None):
        # if at_step == -1:
        ssd = self.ssd(self.optimizer.mp.image)

        rot_def =   apply_rot_mat(self.optimizer.mp.id_grid,  self.optimizer.mp.rot_mat)
        rotated_image =  tb.imgDeform(self.optimizer.mp.image,rot_def,dx_convention='2square')
        ssd_rot = self.ssd(rotated_image)

        return self.alpha * ssd_rot + (1-self.alpha) * ssd


# datacost = mt.Ssd_normalized(newimg_r)
# datacost =  None
datacost = Rotation_Ssd_Cost(newimg_r.to('cuda:0'), alpha=0.8)

torch.autograd.set_detect_anomaly(True)
# Metamorphosis params
rho = 1
dx_convention = '2square'

r = 5
# r = torch.tensor([r])
momentum_I = torch.zeros(img.shape,
                         dtype=torch.float32,
                         device='cuda:0')

# r.requires_grad = True
momentum_I.requires_grad = True

momentum_R = torch.tensor(
    [[0,r],
     [-r,0]],
    dtype=torch.float32, device='cuda:0')
# momentum_R = torch.zeros((2,2),
#                         dtype=torch.float32,
#                         device='cuda:0'
#                         )
momentum_R.requires_grad = True
momenta = {'momentum_I':momentum_I,
           # 'r':r,
           'momentum_R':momentum_R}

# momenta = {k: v.to('cuda:0') for k, v in momenta.items()}

n_steps =  5
mp = mtrt.RotatingMetamorphosis_integrator(
    rho=rho,
    n_step=n_steps,
    kernelOperator=kernelOperator,
    dx_convention=dx_convention
)
mp.forward(img,momenta, save=False, plot=0)
# p = mp.image.sum().backward()
# ic(p.grad)

fig, ax = plt.subplots(1,2)
ax[0].imshow(mp.image[0,0].detach().cpu())
plt.show()

img =  img.to('cuda:0')
newimg_r = newimg_r.to('cuda:0')

# mr = mtrt.RotatingMetamorphosis_Optimizer(
#     source= img,
#     target= newimg_r,
#     geodesic = mp,
#     cost_cst=.001,
#     data_term=datacost,
#     # optimizer_method="adadelta",
# )
# mr.forward(momenta, n_iter=5, grad_coef=1)
# mr.plot()
# mr.mp.plot(n_figs=min(5,n_steps))

# mp.forward(img,momenta, save=True, plot=0)


plt.show()

