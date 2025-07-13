import torch
from demeter.utils.decorators import time_it
import demeter.utils.torchbox as tb
import demeter.utils.cost_functions as cf
import  demeter.metamorphosis.rotate as mtrt

def compute_img_barycentre(img, id_grid = None, dx_convention= '2square'):
    if id_grid is None:
        id_grid = tb.make_regular_grid(img.shape[2:],dx_convention=dx_convention,)

    img_bin = (img[0,0,...,None] * id_grid[0])
    bary = img_bin.sum(dim=[0,1,2]) / img.sum()
    return bary

def align_barycentres(img_1, img_2, verbose = False):
    """

    """
    assert img_1.shape == img_2.shape
    id_grid = tb.make_regular_grid(img_1.shape[2:],dx_convention="2square",)
    b_1 = compute_img_barycentre(img_1, id_grid)
    b_2 = compute_img_barycentre(img_2, id_grid)
    if verbose:
        print("S compute barycentre :",b_1)
        print("T compute barycentre :",b_2)
        print("diff : ", b_2 - b_1)

    img_1_b = tb.imgDeform(img_1, (id_grid + b_1))
    img_2_b = tb.imgDeform(img_2, (id_grid + b_2))
    return img_1_b, img_2_b, b_1, b_2

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
def initial_exploration(rigid_meta_optim, r_step = 4, max_output = 10, verbose:bool = True):
    """
    Shoot in given directions, gives you the best outputs

    Parameters
    -----------
    rigid_meta_optim : metamorphosis.Optimize_geodesicShooting

    r_step : int
        Step of the angles grid
    max_output : int
        number of output to return

    """
    r_list = torch.linspace(-torch.pi, torch.pi * (1 - 2/r_step) , r_step)
    r_combi = torch.cartesian_prod(r_list,r_list,r_list)
    top_losses = []
    for i,params_r in  enumerate(r_combi):
        if verbose:
            print(f"Init search : {i+1} / {len(r_combi)}")
        momenta =mtrt.prepare_momenta(
            rigid_meta_optim.source.shape,
            image=False,rotation=True,translation=False,
            rot_prior=params_r,trans_prior=(0,0,0),
            requires_grad=False
        )
        print(momenta.keys())
        rigid_meta_optim.mp.forward(rigid_meta_optim.source, momenta)

        rot_def =   tb.apply_rot_mat(rigid_meta_optim.mp.id_grid, rigid_meta_optim. mp.rot_mat.T)
        img_rot = tb.imgDeform(rigid_meta_optim.source, rot_def.to('cpu'), dx_convention='2square')
        # img_rot = torch.clip(img_rot, 0, 1)
        loss_val = cf.SumSquaredDifference(rigid_meta_optim.target)(img_rot)

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




import matplotlib.pyplot as plt
#%% rigid optimisation
def optimize_on_rigid(mr, top_params, n_iter= 10, grad_coef = 1,verbose = False):
    best_loss = top_params[0][0]
    for i,(val,params_r) in  enumerate(top_params):
        if verbose:
            print(">"*10)
            print(f"{i}/{len(top_params)} Optimize wit params {params_r.tolist()}")

        momenta = mtrt.prepare_momenta(
            mr.source.shape,
            image=False,rotation=True,translation=True,
            rot_prior=params_r,trans_prior=(0,0,0),
        )

        mr.forward(momenta, n_iter = n_iter, grad_coef= grad_coef)
        # mr = rigid_along_metamorphosis(
        #     source,target,momenta, kernelOperator,
        #     rho= rho,
        #     data_term=datacost,
        #     n_steps=integration_steps,
        #     n_iter=10,
        #     cost_cst=1e-1,
        #     safe_mode=True,
        #     debug =  False
        # )

        best = False
        if mr.data_loss < best_loss or mr.data_loss == 0:
            best_loss = mr.data_loss
            best_momentum = mr.to_analyse[0]["momentum_R"]
            best_translation = mr.mp.translation
            best = True
            best_rot = mr.mp.rot_mat

        # mr.plot_cost(
        # )
        # plt.show()
        if verbose:
            print(f"best = {best}")
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
    return best_loss, best_momentum, best_translation, best_rot
