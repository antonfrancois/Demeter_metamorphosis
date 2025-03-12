import src.demeter.utils.torchbox as tb
import torch
from math import cos,sin
import matplotlib.pyplot as plt

def create_rot_mat(theta):
    return torch.tensor([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]],
                        dtype=torch.float)

def create_affine_mat_2d(params):
    """
    params : theta, a,b,s1,s2
    """
    theta, a,b,s1,s2 = params

    A = torch.stack(
        [
        torch.stack([torch.cos(theta)/s1, -torch.sin(theta), a]),
        torch.stack([torch.sin(theta), torch.cos(theta)/s2, b]),
        torch.tensor([0, 0, 1])
        ]
    )
    return A
    # return torch.tensor(
    #     [
    #         [cos(theta)/s1, -sin(theta), a],
    #         [sin(theta), cos(theta)/s2, b],
    #         [0, 0, 1],
    #     ],
    #     dtype = torch.float
    # )

def affine_to_grid(affine_mat,img_shape):
    id_grid = tb.make_regular_grid(img_shape,dx_convention='2square')

    # apply affine to grid
    id_grid_aug = torch.cat(
        [id_grid,torch.ones_like(id_grid[...,0])[...,None]],
        dim = -1
    )
    aff_grid = torch.einsum('ij,klmj->klmi', affine_mat, id_grid_aug)
    return aff_grid[...,:-1]

img = tb.reg_open('b0')
# pad img
img = torch.nn.functional.pad(img,(40,40,40,40))
# theta = 2*torch.pi/3
theta = 2*torch.pi/3
# rot = create_rot_mat(theta)
A = create_affine_mat_2d(torch.tensor([theta, 0.3, 0, 1.2, .8]))
print(A)


aff_grid = affine_to_grid(A,img.shape[2:])
img_r = tb.imgDeform(img,aff_grid,dx_convention='2square')

# img_r += torch.randn_like(img_r)*0.1
plt.figure()
plt.imshow(tb.imCmp(img,img_r,'segw'))
plt.show()


# fig,ax = plt.subplots(1,1)
# tb.gridDef_plot(id_grid,ax = ax,step = 30)
# tb.gridDef_plot(aff_grid[...,:-1],ax = ax,step = 30, color='r')
# plt.show()

#%%
from src.demeter.utils import update_progress


class OptimAffine(torch.nn.Module):

    def __init__(self, source, target, n_iter, param_init = None, dt_step=1, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if param_init is None:
            self.parameter = torch.tensor(
                [3,0.,0.,1.,1.],device = source.device
            )
        else:
            self.parameter = torch.tensor([param_init],device=source.device)
        self.parameter.requires_grad = True
        self._initialize_LBFGS_(dt_step = dt_step,max_iter=n_iter)
        self.n_iter = n_iter
        self.source = source
        self.target = target.to(source.device)
        self.id_grid = tb.make_regular_grid(source.shape[2:],
                                            dx_convention='2square'
                                            ).to(source.device)

    def _initialize_LBFGS_(self,dt_step,max_iter = 20):
        self.optimizer = torch.optim.LBFGS([self.parameter],
                                           max_eval=15,
                                           max_iter=max_iter,
                                           lr=dt_step,
                                           line_search_fn='strong_wolfe')

        def closure():
            self.optimizer.zero_grad()
            L = self.cost()
            #save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward()
            return L
        self.closure = closure

    def forward(self):
        self.cost()

        for i in range(self.n_iter):
            self._step_LBFGS_()
            update_progress((i+1)/self.n_iter,message=('ssd : ',self.ssd))
        return (self.aff_source.detach(),
                self.parameter.detach(),
                create_affine_mat_2d(self.parameter))

    def _step_LBFGS_(self):
        self.optimizer.step(self.closure)

    def cost(self):
        # print('parameter : ',self.parameter.requires_grad)
        A_mat = create_affine_mat_2d(self.parameter)
        # print("A_mat grad : ",A_mat.requires_grad)
        affine_grid = affine_to_grid(A_mat,self.source.shape[2:])
        self.aff_source = tb.imgDeform(self.source, affine_grid)
        # SSD
        self.ssd = ((self.aff_source - self.target) ** 2).sum() #/ prod(self.source.shape)

        return self.ssd


affOpti = OptimAffine(img,img_r,n_iter = 40,dt_step=50)()
img_r2,theta_new, new_rot_mat = affOpti

fig, ax = plt.subplots(1,3)
ax[0].imshow(img[0,0])
ax[1].imshow(img_r2[0,0])
ax[2].imshow(img_r[0,0])
plt.show()
#%%
print(new_rot_mat)
print(A)

#%%  ROTATION =========================================
# from demeter.utils.constants import *
# from demeter.utils.toolbox import update_progress
#
# class OptimRot(torch.nn.Module):
#
#     def __init__(self, source, target, n_iter, param_init = None,dt_step=1, *args, **kwargs):
#
#         super().__init__(*args, **kwargs)
#         if param_init is None:
#             self.parameter = torch.rand((1,),
#                                      device = source.device,
#                                      )*2*torch.pi
#         else:
#             self.parameter = torch.tensor([param_init],device=source.device)
#         print('parameter : ',self.parameter.item())
#         self.parameter.requires_grad = True
#         self._initialize_LBFGS_(dt_step,max_iter=n_iter)
#         self.n_iter = n_iter
#         self.source = source
#         self.target = target.to(source.device)
#         self.id_grid = tb.make_regular_grid(source.shape[2:],
#                                             dx_convention='2square'
#                                             ).to(source.device)
#
#
#
#     def create_rot_mat(self,theta):
#         cos_theta = torch.cos(theta)
#         sin_theta = torch.sin(theta)
#         rot_mat = torch.stack([torch.stack([cos_theta, -sin_theta],dim=1),
#                                torch.stack([sin_theta, cos_theta],dim=1)],
#                               dim=1)
#         return rot_mat[None]
#
#     def forward(self):
#         self.cost()
#
#         for i in range(self.n_iter):
#             self._step_LBFGS_()
#             update_progress((i+1)/self.n_iter,message=('ssd : ',self.ssd))
#         return self.rot_source.detach(), self.parameter.detach(), create_rot_mat(self.parameter)
#
#
#     def cost(self):
#         rot_mat = self.create_rot_mat(self.parameter)
#
#         rot_grid = self.id_grid @ rot_mat
#         self.rot_source = tb.imgDeform(self.source,rot_grid)
#         # SSD
#         self.ssd =  ((self.rot_source - self.target)**2).sum() /prod(self.source.shape)
#
#         return self.ssd
#
#
#     def _initialize_LBFGS_(self,dt_step,max_iter = 20):
#         self.optimizer = torch.optim.LBFGS([self.parameter],
#                                            max_eval=15,
#                                            max_iter=max_iter,
#                                            lr=dt_step,
#                                            line_search_fn='strong_wolfe')
#
#         def closure():
#             self.optimizer.zero_grad()
#             L = self.cost()
#             #save best cms
#             # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
#             #     cms_tosave.data = self.cms_ini.detach().data
#             L.backward()
#             return L
#         self.closure = closure
#
#     def _step_LBFGS_(self):
#         self.optimizer.step(self.closure)


def optimize_rotation(source,target,n_iter = 20):
    orot_1 = OptimRot(source,target,n_iter =n_iter, param_init = torch.pi/3)
    orot_out_1 = orot_1.forward()

    orot_2 = OptimRot(source,target,n_iter =n_iter, param_init = 4*torch.pi/3)
    orot_out_2 = orot_2.forward()

    # orot_3 = OptimRot(source,target,n_iter =n_iter, param_init = 4*torch.pi/3)
    # orot_out_3 = orot.forward()
    #
    # orot_4 = OptimRot(source,target,n_iter =n_iter, param_init = 5*torch.pi/3)
    # orot_out_4 = orot.forward()

    if orot_1.ssd < orot_2.ssd:
        return orot_out_1
    else:
        return orot_out_2

    # return orot_out

img = tb.reg_open('23')

theta = 2*torch.pi/3
rot = create_rot_mat(theta)
id_grid = tb.make_regular_grid(img.shape[2:],dx_convention='2square')

#apply rot to id_grid
rot_grid = id_grid @ rot[None,None]

# tb.gridDef_plot(rot_grid,step=30)
img_r = tb.imgDeform(img,rot_grid,dx_convention='2square')
img_r += torch.randn_like(img_r)*0.1
plt.figure()
plt.imshow(img_r[0,0])
plt.show()

device = 'cpu'
orot_out = optimize_rotation(img,img_r,n_iter = 10)

img_r2,theta_new, new_rot_mat = orot_out
print('theta_new : ',theta_new)
# print("ssd : ",orot.ssd)
fig, ax = plt.subplots(1,3)
ax[0].imshow(img[0,0])
ax[1].imshow(img_r2[0,0])
ax[2].imshow(img_r[0,0])
plt.show()


#%%
# =============================================================================
#              3D
# =============================================================================
from nibabel import load as nib_load
import napari

def create_affine_mat_2d(params):
    """
    params : theta, a,b,s1,s2
    """
    theta, a,b,s1,s2 = params

    A = torch.stack(
        [
        torch.stack([torch.cos(theta)/s1, -torch.sin(theta), a]),
        torch.stack([torch.sin(theta), torch.cos(theta)/s2, b]),
        torch.tensor([0, 0, 1])
        ]
    )
    return A


path = "/home/turtlefox/Documents/11_metamorphoses/data/shanoir_data/raw/PSL_005/PSL_005_M02"
file = "PSL_005_M02_flair_mr_dataset_Sag_cube_FLAIR_TR9800.nii.gz"
img = nib_load(path+'/'+file)
img_data = img.get_fdata()
img_affine = img.affine

v = napari.Viewer()
v.add_image(img_data)
v.run()
