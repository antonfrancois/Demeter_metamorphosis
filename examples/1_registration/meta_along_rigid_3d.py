import torch


import demeter.utils.torchbox as tb
from demeter.utils.decorators import time_it
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

st = tb.imCmp(S,T,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and Target (blue)')
plt.show()

## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
momentum_ini = 0

# raise Error("shut")

######################################################################
#%% 1 Alignement des barycentres
# calcul du barycentre:
def compute_img_barycentre(img, id_grid = None, dx_convention= '2square'):
    if id_grid is None:
        id_grid = tb.make_regular_grid(S.shape[2:],dx_convention=dx_convention,)

    img_bin = (img[0,0,...,None] * id_grid[0])
    bary = img_bin.sum(dim=[0,1,2]) / img.sum()
    return bary

id_grid = tb.make_regular_grid(S.shape[2:],dx_convention="2square",)
b_S = compute_img_barycentre(S, id_grid)
b_T = compute_img_barycentre(T, id_grid)
print("S compute barycentre :",b_S)
print("T compute barycentre :",b_T)
print("diff : ", b_T - b_S)

#
S_b = tb.imgDeform(S, (id_grid + b_S))
T_b = tb.imgDeform(T, (id_grid + b_T))


st = tb.imCmp(S,S_b,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source (orange) and source _b (blue)')
plt.show()
st = tb.imCmp(S_b,T_b,method = 'compose')
sl = i3p.imshow_3d_slider(st, title = 'Source - b (orange) and Target _b (blue)')
plt.show()

#%%
def prepare_momenta(image_shape,
                    image : bool = True,
                    rotation : bool = True,
                    translation : bool = True,
                    rot_prior = None,
                    trans_prior= None,
                    device = "cuda:0",
                    requires_grad = True):
    dim = 2 if len(image_shape) == 4 else 3
    if rot_prior is None:
        rot_prior = torch.zeros((dim,))
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
        if len(rot_prior.shape)==2:
            momenta["momentum_R"] = torch.tensor(rot_prior,**kwargs)
        elif len(rot_prior.shape)==1:
            r1, r2, r3 = rot_prior
            momenta["momentum_R"] = torch.tensor(
            [[0,-r1, -r2 ],
                     [r1, 0, -r3],
                     [r2, r3, 0]],
                    dtype=torch.float32, device='cuda:0')
        else:
            raise ValueError("Rotation prior must be 2 or 1 dimensional")
    if translation:
        momenta["momentum_T"]= torch.tensor(trans_prior,
                                            **kwargs)

    for keys in momenta.keys():
        momenta[keys].requires_grad=requires_grad

    return momenta



#%%

kernelOperator = rk.GaussianRKHS(sigma=(10,10,10),normalized=False)
# kernelOperator = rk.VolNormalizedGaussianRKHS(
#     sigma=(10,10,10),
#     sigma_convention='pixel',
#     dx=(1, 1, 1),
# )



# datacost = mt.Ssd_normalized(T.to('cuda:0'))
# datacost =  None
datacost = mt.Rotation_Ssd_Cost(T_b.to('cuda:0'), alpha=1)




# torch.autograd.set_detect_anomaly(True)
# Metamorphosis params
rho = 1


#%% Explore angles
def _insert_(entry, top_losses, top_k = 10):
    # Insert in sorted position
    inserted = False
    for i, (existing_loss, _) in enumerate(top_losses):
        if entry[0] < existing_loss:
            top_losses.insert(i, entry)
            inserted = True
            break

    # If not inserted and list not full, append at the end
    if not inserted and len(top_losses) < top_k:
        top_losses.append(entry)

    # Prune list if too long
    if len(top_losses) > top_k:
        top_losses = top_losses[:top_k]

    return top_losses

@time_it
def initial_exploration(integration_steps, r_step = 4, max_output = 10, verbose:bool = True):
    r_list = torch.linspace(-torch.pi, torch.pi, r_step)
    r_combi = torch.cartesian_prod(r_list,r_list,r_list)
    top_losses = []
    for i,params_r in  enumerate(r_combi):
        if verbose:
            print(f"Init search : {i+1} / {len(r_combi)}")
        momenta = prepare_momenta(
            S.shape,
            image=False,rotation=True,translation=False,
            rot_prior=params_r,trans_prior=(0,0,0),
            requires_grad=False
        )
        mp = mtrt.RigidMetamorphosis_integrator(
            rho=1,
            n_step=integration_steps,
            kernelOperator=None,
            dx_convention="2square"
        )
        mp.forward(S, momenta)

        rot_def =   tb.grid_from_rotation(mp.id_grid, mp.rot_mat.T)
        img_rot = tb.imgDeform(S_b, rot_def.to('cpu'), dx_convention='2square')
        # img_rot = torch.clip(img_rot, 0, 1)
        loss_val = cf.SumSquaredDifference(T_b)(img_rot)

        # keep a list of the N best loss_val related with the corresponding param_r
        entry = (loss_val.detach(), params_r.detach())
        if verbose:
            print(f"\t {entry}")
        top_losses = _insert_(entry, top_losses, max_output)

    if verbose:
        print('Best params found')
        for loss, params in top_losses:
            print(f"Loss: {loss.item():.4f} | Params: {params.tolist()}")

    return top_losses





#%% rigid optimisation
def optimize_on_rigid(source, target, integration_steps, top_params, verbose = False):
    best_loss = top_params[0][0]
    for i,(val,params_r) in  enumerate(top_params):
        print(">"*10)
        ic(params_r)
        momenta = prepare_momenta(
            S.shape,
            image=False,rotation=True,translation=True,
            rot_prior=params_r,trans_prior=(0,0,0),
        )


        mr = rigid_along_metamorphosis(
            source,target,momenta, kernelOperator,
            rho= rho,
            data_term=datacost,
            n_steps=integration_steps,
            n_iter=10,
            cost_cst=1e-1,
            safe_mode=True,
            debug =  False
        )


        best = False
        if mr.data_loss < best_loss or mr.data_loss == 0:
            best_loss = mr.data_loss
            best_momentum = mr.to_analyse[0]["momentum_R"]
            best_translation = mr.mp.translation
            best = True
        # mr.plot_cost(
        # )
        # plt.show()
        if verbose:
            print(f"i : {i} / {len(top_params)}, best = {best}")
            print(aff_mat)
            print(mr.mp.translation)
            print(mr.mp.rot_mat)
            print("best mom",best_momentum)
            print("anti best mom", (best_momentum - best_momentum.T)/2)
            print("best loss",mr.data_loss)
            print("<"*10)
        if best_loss < 1:
            print("rigid_optim stop.")
            break
    if verbose:
        print("Best find : ")
        print(best_loss)
        print(best_momentum)
        print(best_translation)
    return best_loss, best_momentum, best_translation

integration_steps = 7
top_params = initial_exploration(integration_steps,r_step=5)
loss, mom_R_ini, mom_T_ini = optimize_on_rigid(S_b, T_b,integration_steps, top_params)

print("\nBegin Metamorphosis >>>")

momenta = prepare_momenta(
    S.shape,
    image=True,rotation=True,translation=True,
    rot_prior=mom_R_ini,
    trans_prior=mom_T_ini,
    # trans_prior=(0,0,0)
)


datacost = mt.Rotation_Ssd_Cost(T_b.to('cuda:0'), alpha=.5)
mr = rigid_along_metamorphosis(
    S_b,T_b,momenta, kernelOperator,
    rho= rho,
    data_term=datacost,
    n_steps=integration_steps,
    n_iter=15,
    cost_cst=1e-1,
    debug=False,
)

print("Estimated")
print("tau",mr.mp.translation)
print("boom:", b_S + mr.mp.translation - b_T)
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
rot_def =   tb.grid_from_rotation(mr.mp.id_grid, mr.mp.rot_mat.T)
if mr.mp.flag_translation:
    rot_def += mr.mp.translation
img_rot = tb.imgDeform(mr.mp.image, rot_def.to('cpu'),
                       dx_convention='2square', clamp=True)
st = tb.imCmp(img_rot,T_b,method = 'compose')

i3p.imshow_3d_slider(st, title = "Image rotate cmp avec target")
plt.show()


#%%
# i3p.Visualize_GeodesicOptim_plt(mr,"ball_for_hanse_rot")
# plt.show()

