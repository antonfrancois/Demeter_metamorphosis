import torch

import demeter.utils.torchbox as tb
from demeter.constants import *
import demeter.utils.image_3d_plotter as i3p
import demeter.metamorphosis.rotate as mtrt
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.reproducing_kernels as rk


cuda = torch.cuda.is_available()
# cuda = True
device = 'cpu'
if cuda:
    device = 'cuda:0'
print('device used :',device)

def create_affine_mat_3d(params):
    r"""
    build a 2D affine matrix for 3D affine transformation such as
    $$A = \begin{bmatrix}
    \cos(\beta) \sin(\gamma)/s1 & -\sin(\alpha) \sin(\beta) \cos(\gamma) - \cos(\alpha) \sin(\gamma) & \cos(\alpha) \sin(\beta) \cos(\gamma) - \sin(\alpha) \sin(\gamma) & a \\
    \cos(\beta) \cos(\gamma) & (-\sin(\alpha) \sin(\beta) \sin(\gamma) + \cos(\alpha) \cos(\gamma))/s2 & \cos(\alpha) \sin(\beta) \sin(\gamma) + \sin(\alpha) \cos(\gamma) & b \\
    -\sin(\beta) & \sin(\alpha) \cos(\beta) & \cos(\alpha) \cos(\beta)/s3 & c \\
    0 & 0 & 0 & 1
    \end{bmatrix}$$

    params : tensor of len 9 containing: gamma,beta,alpha, a,b,c,s1,s2,s3
    """
    gamma,beta,alpha, a,b,c,s1,s2,s3 = params

    A = torch.stack(
        [
        torch.stack(
            [torch.cos(beta) * torch.cos(gamma)/s1,
             -torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma),
             torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.sin(alpha) * torch.sin(gamma),
             a]
        ),
        torch.stack(
            [
            torch.cos(beta) * torch.sin(gamma),
            (-torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma))/s2,
            torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.sin(alpha) * torch.cos(gamma),
                b
            ]),
        torch.stack(
            [
            - torch.sin(beta),
            torch.sin(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.cos(beta)/s3,
                c
            ]),
        torch.tensor([ 0, 0, 0, 1],device=params.device)
        ]
    )
    return A

def affine_to_grid_3d(affine_mat,img_shape):
    id_grid = tb.make_regular_grid(img_shape,dx_convention='2square')

    # apply affine to grid
    id_grid_aug = torch.cat(
        [id_grid,torch.ones_like(id_grid[...,0])[...,None]],
        dim = -1
    )
    aff_grid = torch.einsum('ij,hklmj->hklmi', affine_mat, id_grid_aug)
    return aff_grid[...,:-1]


path = ROOT_DIRECTORY+"/examples/im3Dbank/"
source_name = "hanse_w_ball"
target_name = "hanse_w_ball"
S = torch.load(path+source_name+".pt")
T = torch.load(path+target_name+".pt")

# if the image is too big for your GPU, you can downsample it quite barbarically :
step = 3 if device == 'cuda:0' else 3
if step > 0:
    S = S[:,:,::step,::step,::step]
    T = T[:,:,::step,::step,::step]
_,_,D,H,W = S.shape

args_aff = torch.tensor(
        [.6,-.3, 0, # angle
        0,0,0,   # translation
        1,1,1] # scaling
)

aff_mat = create_affine_mat_3d(args_aff)
aff_grid = affine_to_grid_3d(aff_mat,(D, H,W))

S = tb.imgDeform(S, aff_grid, dx_convention='2square', clamp=True)

# add deformations
def make_exp(xxx, yyy, zzz, centre, sigma):
    ce_x, ce_y, ce_z = centre
    sigma_x,sigma_y, sigma_z = sigma
    exp = torch.exp(
        - 0.5*((xxx - ce_x) / sigma_x) ** 2
        - 0.5*((yyy - ce_y) / sigma_y) ** 2
        - 0.5*((zzz - ce_z) / sigma_z) ** 2
    )
    return exp

xx,yy, zz = aff_grid[...,0].clone(),aff_grid[...,1].clone(), aff_grid[...,2].clone()
xx -= .2* make_exp(xx,yy,zz,(0.25,0.25, .25),(0.1,0.1, .5))
yy -= (.2* make_exp(xx,yy,zz,(0,-0.25,0),(0.15,0.15,.5)))

deform = torch.stack([xx,yy,zz],dim = -1)
# deform = id_grid.clone()
S = tb.imgDeform(S, deform, dx_convention='2square', clamp=True)

st = tb.imCmp(S,T,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and Target (blue)')
plt.show()

## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
momentum_ini = 0


#%%
######################################################################
kernelOperator = rk.GaussianRKHS(sigma=(10,10,10),normalized=False)
# kernelOperator = rk.VolNormalizedGaussianRKHS(
#     sigma=(10,10,10),
#     sigma_convention='pixel',
#     dx=(1, 1, 1),
# )

# class Rotation_Ssd_Cost(mt.DataCost):
#
#     def __init__(self, target, alpha, **kwargs):
#
#         super(Rotation_Ssd_Cost, self).__init__(target)
#         self.ssd = cf.SumSquaredDifference(target)
#         self.alpha = alpha
#
#     def set_optimizer(self, optimizer):
#         """
#         DataCost object are meant to be used along a
#         method inherited from `Optimize_geodesicShooting`.
#         This method is used to set the optimizer object and is usually
#         used at the optimizer initialisation.
#         """
#         self.optimizer = optimizer
#         # try:
#         #     self.optimizer.mp.rot_mat
#         # except AttributeError:
#         #     raise AttributeError(f"The optimizer must have Rotation implemented, optimizer is {optimizer.__class__.__name__}")
#         if self.target.shape != self.optimizer.source.shape and not self.target is None:
#             raise ValueError(
#                 "Target and source shape are different."
#                 f"Got source.shape = {self.optimizer.source.shape}"
#                 f"and target.shape = {self.target.shape}."
#                 f"Have you checked your DataCost initialisation ?"
#             )
#
#
#
#     def __call__(self,at_step=None):
#         # if at_step == -1:
#         ssd = self.ssd(self.optimizer.mp.image)
#
#         rot_def =   mtrt.apply_rot_mat(self.optimizer.mp.id_grid,  self.optimizer.mp.rot_mat)
#         rotated_image =  tb.imgDeform(self.optimizer.source,rot_def,dx_convention='2square')
#         ssd_rot = self.ssd(rotated_image)
#
#         return self.alpha * ssd_rot + (1-self.alpha) * ssd


# datacost = mt.Ssd_normalized(T.to('cuda:0'))
# datacost =  None
datacost = mt.Rotation_Ssd_Cost(T.to('cuda:0'), alpha=.5)


# datacost =  None

# torch.autograd.set_detect_anomaly(True)
# Metamorphosis params
rho = 1
dx_convention = '2square'
from math import pi
r1, r2, r3 = 0, 0, 0
# r = torch.tensor([r])
momentum_I = torch.zeros(S.shape,
                         dtype=torch.float32,
                         device='cuda:0')

# r.requires_grad = True
momentum_I.requires_grad = True

momentum_R = torch.tensor(
    [[0,-r1, r2 ],
     [r1, 0, -r3],
     [-r2, r3, 0]],
    dtype=torch.float32, device='cuda:0')

momentum_R = torch.tensor([[ 0.0000, -1.1283, -0.5543],
        [ 1.1283,  0.0000, -0.1769],
        [ 0.5543,  0.1769,  0.0000]],
    dtype=torch.float32, device='cuda:0')
# momentum_R = torch.zeros((2,2),
#                         dtype=torch.float32,
#                         device='cuda:0'
#                         )
momentum_R.requires_grad = True
momenta = {'momentum_I':momentum_I,
           # 'r':r,
           'momentum_R':momentum_R}

n_steps =  7
mp = mtrt.RotatingMetamorphosis_integrator(
    rho=rho,
    n_step=n_steps,
    kernelOperator=kernelOperator,
    dx_convention=dx_convention
)

S =  S.to('cuda:0')
T = T.to('cuda:0')

mr = mtrt.RotatingMetamorphosis_Optimizer(
    source= S,
    target= T,
    geodesic = mp,
    cost_cst=1e-1,
    data_term=datacost,
    hamiltonian_integration=False
    # optimizer_method="adadelta",
)
mr.forward(momenta, n_iter=5, grad_coef=1)
#%%
# img = mr.mp.image
# st = tb.imCmp(img,T,method = 'compose')
# sl = i3p.imshow_3d_slider(st, title = 'deformed (orange) and Target (blue)')
# plt.show()
#
# st = tb.imCmp(img,S,method = 'compose')
# sl = i3p.imshow_3d_slider(st, title = 'deformed (orange) and Source (blue)')
# plt.show()
mr.plot_cost()

# rot_def =   tb.apply_rot_mat(mr.mp.id_grid,  mr.mp.rot_mat.T)
# source_rot = tb.imgDeform(S.to('cpu'),rot_def,dx_convention='2square')
# st = tb.imCmp(source_rot,T,method = 'compose')
# print(source_rot.shape)
#
# i3p.imshow_3d_slider(st, title = "Source rotate cmp avec target")
# plt.show()

#%%
rot_def =   tb.apply_rot_mat(mr.mp.id_grid,  mr.mp.rot_mat.T)
img_rot = tb.imgDeform(mr.mp.image, rot_def.to('cpu'), dx_convention='2square')
st = tb.imCmp(img_rot,T,method = 'compose')

i3p.imshow_3d_slider(st, title = "Image rotate cmp avec target")
plt.show()

#%%
# i3p.Visualize_GeodesicOptim_plt(mr,"ball_for_hanse_rot")
# plt.show()

