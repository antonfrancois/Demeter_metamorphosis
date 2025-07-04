import torch
from demeter.utils.decorators import time_it
import demeter.utils.torchbox as tb
import demeter.utils.cost_functions as cf
from demeter.metamorphosis.wraps import rigid_along_metamorphosis
import  demeter.metamorphosis.rotate as mtrt

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
def initial_exploration(source, target, integration_steps, r_step = 4, max_output = 10, verbose:bool = True):
    r_list = torch.linspace(-torch.pi, torch.pi, r_step)
    r_combi = torch.cartesian_prod(r_list,r_list,r_list)
    top_losses = []
    for i,params_r in  enumerate(r_combi):
        if verbose:
            print(f"Init search : {i+1} / {len(r_combi)}")
        momenta =mtrt.prepare_momenta(
            source.shape,
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
        mp.forward(source, momenta)

        rot_def =   tb.apply_rot_mat(mp.id_grid,  mp.rot_mat.T)
        img_rot = tb.imgDeform(source, rot_def.to('cpu'), dx_convention='2square')
        # img_rot = torch.clip(img_rot, 0, 1)
        loss_val = cf.SumSquaredDifference(target)(img_rot)

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
def optimize_on_rigid(mr, top_params, n_iter= 10, grad_coef = 1,verbose = False):
    best_loss = top_params[0][0]
    for i,(val,params_r) in  enumerate(top_params):
        print(">"*10)
        ic(params_r)
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
