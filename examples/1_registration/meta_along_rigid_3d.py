import torch


import demeter.utils.torchbox as tb
from demeter.utils.decorators import time_it
from demeter.constants import *
# import demeter.utils.image_3d_plotter as i3p
import demeter.metamorphosis.rotate as mtrt
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.reproducing_kernels as rk
from demeter.metamorphosis import rigid_along_metamorphosis

import demeter.utils.rigid_exploration as re
import demeter.utils.axes3dsliders_plt as a3s

cuda = torch.cuda.is_available()
# cuda = True
device = 'cpu'
if cuda:
    device = 'cuda:0'
print('device used :',device)




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
        [-3.14/2, 3*3.14/4, 0, # angle
        0,0.2,0,
        # 0.5,-.1,.15,   # translation
        1,1,1] # scaling
)

aff_mat = tb.create_affine_mat_3d(args_aff)
print(aff_mat)
print(aff_mat.T @ aff_mat)
aff_grid = tb.affine_to_grid_3d(aff_mat.T,(D, H,W))


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
T = tb.imgDeform(T, deform, dx_convention='2square', clamp=True)



## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
momentum_ini = 0

# raise Error("shut")

######################################################################
#%% 1 Alignement des barycentres
# calcul du barycentre:
# def compute_img_barycentre(img, id_grid = None, dx_convention= '2square'):
#     if id_grid is None:
#         id_grid = tb.make_regular_grid(S.shape[2:],dx_convention=dx_convention,)
#
#     img_bin = (img[0,0,...,None] * id_grid[0])
#     bary = img_bin.sum(dim=[0,1,2]) / img.sum()
#     return bary
#
# id_grid = tb.make_regular_grid(S.shape[2:],dx_convention="2square",)
# b_S = compute_img_barycentre(S, id_grid)
# b_T = compute_img_barycentre(T, id_grid)
# print("S compute barycentre :",b_S)
# print("T compute barycentre :",b_T)
# print("diff : ", b_T - b_S)
#
# #
# S_b = tb.imgDeform(S, (id_grid + b_S))
# T_b = tb.imgDeform(T, (id_grid + b_T))
S_b, T_b, trans_s, trans_t = re.align_barycentres(S, T, verbose=True)

# image_list = [    # list of functions that returns the image to show
#      {'name': "source", "image": S, "cmap":"gray"},
#      {'name': "target", "image": T, "cmap":"gray"},
#     {'name': "source_b", "image": S_b, "cmap":"gray"},
#      {'name': "target_b", "image": T_b, "cmap":"gray"},
#                       ]
#
# # img_cmp = self.temporal_image_cmp_with_target()
# img = image_list[0]["image"]
# print(img.shape)
# img_ctx = a3s.Image3dAxes_slider(img, cmap='gray')
# img_toggle = a3s.ToggleImage3D(img_ctx, image_list)
# plt.show()

#%%
# def prepare_momenta(image_shape,
#                     image : bool = True,
#                     rotation : bool = True,
#                     translation : bool = True,
#                     rot_prior = None,
#                     trans_prior= None,
#                     device = "cuda:0",
#                     requires_grad = True):
#     dim = 2 if len(image_shape) == 4 else 3
#     if rot_prior is None:
#         rot_prior = torch.zeros((dim,))
#     if trans_prior is None:
#         trans_prior = [0] * dim
#     momenta = {}
#     kwargs = {
#         "dtype":torch.float32,
#         "device":device
#     }
#     if image:
#         momenta["momentum_I"]= torch.zeros(S.shape,**kwargs)
#     if rotation:
#         if len(rot_prior.shape)==2:
#             momenta["momentum_R"] = torch.tensor(rot_prior,**kwargs)
#         elif len(rot_prior.shape)==1:
#             r1, r2, r3 = rot_prior
#             momenta["momentum_R"] = torch.tensor(
#             [[0,-r1, -r2 ],
#                      [r1, 0, -r3],
#                      [r2, r3, 0]],
#                     dtype=torch.float32, device='cuda:0')
#         else:
#             raise ValueError("Rotation prior must be 2 or 1 dimensional")
#     if translation:
#         momenta["momentum_T"]= torch.tensor(trans_prior,
#                                             **kwargs)
#
#     for keys in momenta.keys():
#         momenta[keys].requires_grad=requires_grad
#
#     return momenta



#%%

# Metamorphosis params
kernelOperator = rk.GaussianRKHS(sigma=(10,10,10),normalized=False)
# kernelOperator = rk.VolNormalizedGaussianRKHS(
#     sigma=(10,10,10),
#     sigma_convention='pixel',
#     dx=(1, 1, 1),
# )



# datacost = mt.Ssd_normalized(T.to('cuda:0'))
# datacost =  None
datacost = mt.Rotation_Ssd_Cost(T_b.to('cuda:0'), alpha=.5)

integration_steps = 10
rho = 0.5
cost_cst= 1e-5
momenta = mtrt.prepare_momenta(
    S.shape,
    image=False,rotation=True,translation=True,
    # trans_prior=(0,0,0)
)
mr = rigid_along_metamorphosis(
    S_b,T_b,momenta, kernelOperator,
    rho= rho,
    data_term=datacost,
    integration_steps=integration_steps,
    n_iter=0,
    cost_cst=cost_cst,
    debug=False,
)

# torch.autograd.set_detect_anomaly(True)



top_params = re.initial_exploration(mr, r_step=5, max_output=10, verbose=True)
loss, best_momentum_R, best_momentum_T, best_rot = re.optimize_on_rigid(mr, top_params, verbose=True)
# best_momentum_R = torch. tensor([[ 0.0000, -5.0279, -0.5246],
#         [ 5.0279,  0.0000, -0.7909],
#         [ 0.5246,  0.7909,  0.0000]])
# best_momentum_T = torch. tensor([-0.0374, -0.0029, -0.0043])

print("real affine : ", aff_mat)

print("\nBegin Metamorphosis >>>")

momenta = mtrt.prepare_momenta(
    S.shape,
    image=True,rotation=True,translation=True,
    rot_prior=best_momentum_R,
    trans_prior=best_momentum_T,
    # trans_prior=(0,0,0)
)


datacost = mt.Rotation_Ssd_Cost(T_b.to('cuda:0'), alpha=0)
mr = rigid_along_metamorphosis(
    S_b,T_b,momenta, kernelOperator,
    rho= rho,
    data_term=datacost,
    integration_steps=integration_steps,
    n_iter=15,
    cost_cst=cost_cst,
    debug=False,
)

print("Estimated")
print("tau",mr.mp.translation)
# print("boom:", b_S + mr.mp.translation - b_T)
# print("baam:", b_S + mr.mp.translation - b_T)
print(mr.mp.rot_mat)
print("Real:")
print(aff_mat)
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
a3s.Visualize_GeodesicOptim_plt(mr, f"toy example rho = {rho}")

#%%
# i3p.Visualize_GeodesicOptim_plt(mr,"ball_for_hanse_rot")
# plt.show()

