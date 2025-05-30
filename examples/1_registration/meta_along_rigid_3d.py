from mailbox import Error

import torch

import demeter.utils.torchbox as tb
from demeter.constants import *
import demeter.utils.image_3d_plotter as i3p
import demeter.metamorphosis.rotate as mtrt
import demeter.metamorphosis as mt
import demeter.utils.cost_functions as cf
import demeter.utils.reproducing_kernels as rk
from demeter.metamorphosis import rigid_along_metamorphosis


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
        [3.14/2, 3.14/4, 0, # angle
        0,-0.1,0.15,
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
# xx -= .2* make_exp(xx,yy,zz,(0.25,0.25, .25),(0.1,0.1, .5))
# yy -= (.2* make_exp(xx,yy,zz,(0,-0.25,0),(0.15,0.15,.5)))

deform = torch.stack([xx,yy,zz],dim = -1)
# deform = id_grid.clone()
T = tb.imgDeform(T, deform, dx_convention='2square', clamp=True)

st = tb.imCmp(S,T,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and Target (blue)')
plt.show()

## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
momentum_ini = 0

# raise Error("shut")
#%%
######################################################################
def prepare_momenta(image_shape,
                    image : bool = True,
                    rotation : bool = True,
                    translation : bool = True,
                    rot_prior = None,
                    trans_prior= None,
                    device = "cuda:0"):
    dim = 2 if len(image_shape) == 4 else 3
    if rot_prior is None:
        rot_prior = torch.eye(dim)
    if trans_prior is None:
        trans_prior = [0] * dim
    momenta = {}
    kwargs = {
        "dtype":torch.float32,
        "device":device
    }
    if image:
        momenta["momentum_I"]= torch.zeros(S.shape,**kwargs)
    if rotation:
            momenta["momentum_R"] = torch.tensor(rot_prior,**kwargs)
    if translation:
        momenta["momentum_T"]= torch.tensor(trans_prior,
                                            **kwargs)

    for keys in momenta.keys():
        momenta[keys].requires_grad=True

    return momenta

#%%
from math import pi, sqrt
import demeter.utils.torchbox as tb
angles_list = torch.tensor([
    # [0, 0, 0],
    [pi/2, 0, 0],
    [0, pi/2, 0],

    [pi, 0, 0],
    [-pi/2, 0, 0],
    [0, pi/2, 0],
    [0, -pi/2, 0],
    # [0, 0, 0],
    [0, pi, 0],
    [0, 0, pi/2],
    [0, 0, -pi/2]
])
for  angles in angles_list:
    print(fr" TEST {angles}" +"- "*10)
    rot_mat_prior = tb.create_rot_mat_3d(angles)
    vect = torch.tensor([[sqrt(2)/2, sqrt(2)/2,0]], dtype=torch.float32)
    print(vect @ rot_mat_prior.T)
    print(rot_mat_prior)
    print(r" END TEST" +"- "*10)
    print()

#%%

kernelOperator = rk.GaussianRKHS(sigma=(10,10,10),normalized=False)
# kernelOperator = rk.VolNormalizedGaussianRKHS(
#     sigma=(10,10,10),
#     sigma_convention='pixel',
#     dx=(1, 1, 1),
# )



# datacost = mt.Ssd_normalized(T.to('cuda:0'))
# datacost =  None
datacost = mt.Rotation_Ssd_Cost(T.to('cuda:0'), alpha=.5)




# torch.autograd.set_detect_anomaly(True)
# Metamorphosis params
rho = 1



angles_list = torch.tensor([
    # [0, 0, 0],
    # [pi/2, 0, 0],
    # [pi, 0, 0],
    # [-pi/2, 0, 0],
    # [0, pi/2, 0],
    # [0, -pi/2, 0]
    [0, 0, 0],
    [0, pi/2, 0],
    [0, pi, 0],
    [0, -pi/2, 0],
    [0, 0, pi/2],
    [0, 0, -pi/2]
])

best_angle = 0
best_loss = torch.inf
for i,angles in  enumerate(angles_list):
    rot_mat_prior = tb.create_rot_mat_3d(angles)

    print(r" TEST" +"- "*10)
    vect = torch.tensor([[1,0,0]], dtype=torch.float32)
    print(vect @ rot_mat_prior.T)
    print(r" END TEST" +"- "*10)

    momenta = prepare_momenta(
        S.shape,
        image=False,rotation=True,translation=True,
        rot_prior=rot_mat_prior,trans_prior=(0,0,0),
    )

    n_steps =  7

    mr = rigid_along_metamorphosis(
        S,T,momenta, kernelOperator,
        rho= rho,
        data_term=datacost,
        n_steps=n_steps,
        n_iter=10,
        cost_cst=1e-1,
        safe_mode=True
    )
    best = False
    if mr.data_loss < best_loss:
        best_loss = mr.data_loss
        best_angle = angles
        best_translation = mr.mp.translation
        best = True


    print("="*10)
    print(f"i : {i}, best = {best}")
    print(angles)
    print(aff_mat)
    print(mr.mp.translation)
    print(mr.mp.rot_mat)
    print(mr.data_loss)
    print("="*10)

print(best_angle)
print(best_loss)


rot_mat_prior = tb.create_rot_mat_3d(best_angle)
momenta = prepare_momenta(
    S.shape,
    image=True,rotation=True,translation=True,
    rot_prior=rot_mat_prior,
    # trans_prior=best_translation,
    trans_prior=(0,0,0)
)

n_steps =  7

mr = rigid_along_metamorphosis(
    S,T,momenta, kernelOperator,
    rho= rho,
    data_term=datacost,
    n_steps=n_steps,
    n_iter=17,
    cost_cst=1e-1
)

print("Estimated")
print(mr.mp.translation)
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
rot_def =   tb.apply_rot_mat(mr.mp.id_grid,  mr.mp.rot_mat.T)
if mr.mp.flag_translation:
    rot_def += mr.mp.translation
img_rot = tb.imgDeform(mr.mp.image, rot_def.to('cpu'), dx_convention='2square')
img_rot = torch.clip(img_rot, 0, 1)
st = tb.imCmp(img_rot,T,method = 'compose')

i3p.imshow_3d_slider(st, title = "Image rotate cmp avec target")
plt.show()


#%%
# i3p.Visualize_GeodesicOptim_plt(mr,"ball_for_hanse_rot")
# plt.show()

