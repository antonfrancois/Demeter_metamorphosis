import torch

from demeter.metamorphosis import prepare_momenta
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
    kw_gridsample = dict(padding_mode="zeros",
                         align_corners=True
                         )
    img_1_b = tb.imgDeform(img_1, (id_grid + b_1), kw_gridsample)
    img_2_b = tb.imgDeform(img_2, (id_grid + b_2), kw_gridsample)
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
def initial_exploration(rigid_meta_optim,
                        r_step = 4,
                        max_output = 10,
                        verbose:bool = True
                        ):
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
    r_list = torch.linspace(-torch.pi, torch.pi  , r_step)
    # r_combi = torch.cartesian_prod(r_list,r_list,r_list)
    if len(rigid_meta_optim.source.shape) == 4:
        r_combi = r_list
    elif len(rigid_meta_optim.source.shape) == 5:
        r_combi = torch.cartesian_prod(
            torch.linspace(-torch.pi, torch.pi  , r_step),
            torch.linspace(0, torch.pi  , r_step//2),
            torch.tensor([0.])
        )
    else:
        raise ValueError("bad source shape")


    top_losses = []
    for i,params_r in  enumerate(r_combi):
        if verbose:
            print(f"Init search : {i+1} / {len(r_combi)}")
        momenta =mtrt.prepare_momenta(
            rigid_meta_optim.source.shape,
            diffeo=False,rotation=True,translation=False, scaling=False,
            rot_prior=params_r,
            requires_grad=False
        )
        print(momenta.keys())
        rigid_meta_optim.mp.forward(rigid_meta_optim.source, momenta, save=False)

        rot_def =   tb.grid_from_rotation(rigid_meta_optim.mp.id_grid, rigid_meta_optim. mp.rot_mat.T)
        img_rot = tb.imgDeform(rigid_meta_optim.source, rot_def.to('cpu'), dx_convention='2square')
        # img_rot = torch.clip(img_rot, 0, 1)
        loss_val = cf.SumSquaredDifference(rigid_meta_optim.target)(img_rot)

        # keep a list of the N best loss_val related with the corresponding param_r
        entry = (loss_val.detach(),
                 dict(
                     rot_prior=params_r.detach(),
                trans_prior=None,
                scale_prior=None,)
            )
        if verbose:
            print(f"\t {entry}")
        top_losses = _insert_(entry, top_losses, max_output)

    if verbose:
        print('Best params found')
        for loss, params in top_losses:
            print(f"Loss: {loss.item():.4f} | Params: {params}")

    return top_losses




import matplotlib.pyplot as plt
#%% rigid optimisation
def optimize_on_rigid(mr,
                      top_params,
                      rotation=True,
                      translation=True,
                      scaling= True,
                      affine= False,
                      n_iter= 10,
                      grad_coef = 1,
                      verbose = False,
                      plot = False):
    # best_loss = top_params[0][0]
    best_loss = torch.inf
    for i,(val,params_r) in  enumerate(top_params):
        if verbose:
            print(">"*10)
            print(f"{i}/{len(top_params)} Optimize wit params {params_r}")

        momenta = mtrt.prepare_momenta(
            mr.source.shape,
            diffeo=False,rotation=rotation,translation=translation,scaling= scaling,affine=affine,
            **params_r
        )
        try:
            mr.forward(momenta, n_iter = n_iter, grad_coef= grad_coef)
        except OverflowError:
            print("xxxxxx Optimization diverged xxxxxx")
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
        def _get_momentums(name):
            try:
                return mr.to_analyse[0][name].detach().cpu()
            except KeyError:
                return None

        if mr.data_loss < best_loss or mr.data_loss == 0:
            best_loss = mr.data_loss
            best_momenta = dict(
                affine_prior = _get_momentums("momentum_A"),
                 rot_prior= _get_momentums("momentum_R"),
                trans_prior=_get_momentums("momentum_T"),
                scale_prior=_get_momentums("momentum_S")
            )
            best = True
            best_rot = mr.mp.rot_mat

        # mr.plot_cost(
        # )
        # plt.show()
        if plot:
            rot_def = mr.mp.get_rigidor()
            rotated_source = tb.imgDeform(mr.source,rot_def,dx_convention='2square')
            if len(mr.source.shape) == 5:
                img = rotated_source[0,0,..., mr.source.shape[-1]//2].detach().cpu()
                img_target = tb.imCmp(rotated_source[..., mr.source.shape[-1]//2].detach().cpu(), mr.target[..., mr.source.shape[-1]//2].detach().cpu(), "compose")[0]
                img_source = tb.imCmp(rotated_source[..., mr.source.shape[-1]//2].detach().cpu(), mr.source[..., mr.source.shape[-1]//2].detach().cpu(), "compose")[0]
            else:
                img = rotated_source[0,0].detach().cpu()
                img_target = tb.imCmp(rotated_source.detach().cpu(), mr.target.detach().cpu(), "compose")[0]
                img_source = tb.imCmp(rotated_source.detach().cpu(), mr.source.detach().cpu(), "compose")[0]

            fig,ax = plt.subplots(1,3)
            fig.suptitle(f"i = {i}, best = {best}, loss = {mr.data_loss:.4f}"
                         f"\n {params_r}"
                         f"\n {dict(
                rot_prior= _get_momentums("momentum_R"),
                trans_prior=_get_momentums("momentum_T"),
                scale_prior=_get_momentums("momentum_S")
            )}")
            ax[0].imshow(img, cmap="gray")
            ax[0].set_title("Final image")
            ax[1].imshow(img_target, cmap="gray")
            ax[1].set_title("img vs target")
            ax[2].imshow(img_source, cmap="gray")
            ax[2].set_title("img vs source")
            plt.show()
        if verbose:
            print(f"best = {best}")
            print("translation = ",mr.mp.translation)
            print("rotation matrix = ",mr.mp.rot_mat)
            print("scaling = ", mr.mp.scale)
            # print("best mom",best_momentum)
            # print("anti best mom", (best_momentum - best_momentum.T)/2)
            # print("best loss",mr.data_loss)
            print("<"*10)
        if best_loss < 1:
            print("rigid_optim stop.")
            break
    if verbose:
        print("Best find : ")
        print("loss :",best_loss)
        print(f"best_momenta : {best_momenta}")

        print("best_rotation =", mr.mp.rot_mat)
    return best_loss, best_momenta, best_rot
