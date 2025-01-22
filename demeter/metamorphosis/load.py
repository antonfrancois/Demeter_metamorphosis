import pickle
from icecream import ic

from ..utils.constants import *
from .classic import Metamorphosis_integrator, Metamorphosis_Shooting
from .constrained import (
    ConstrainedMetamorphosis_integrator,
    ConstrainedMetamorphosis_Shooting,
    Reduce_field_Optim,
)
from .joined import (
    Weighted_joinedMask_Metamorphosis_integrator,
    Weighted_joinedMask_Metamorphosis_Shooting,
)
from .simplex import Simplex_sqrt_Metamorphosis_integrator, Simplex_sqrt_Shooting
from ..utils.reproducing_kernels import (
    GaussianRKHS,
    VolNormalizedGaussianRKHS,
    Multi_scale_GaussianRKHS,
)


def find_meta_optimiser_from_repr(repr_str):
    if "Metamorphosis_Shooting" in repr_str:
        return Metamorphosis_integrator, Metamorphosis_Shooting
    if "ConstrainedMetamorphosis_Shooting" in repr_str:
        return ConstrainedMetamorphosis_integrator, ConstrainedMetamorphosis_Shooting
    if "Reduce_field_Optim" in repr_str:
        return ConstrainedMetamorphosis_integrator, Reduce_field_Optim
    if "Optimize_weighted_joinedMask" in repr_str:
        return (
            Weighted_joinedMask_Metamorphosis_integrator,
            Weighted_joinedMask_Metamorphosis_Shooting,
        )
    if "Simplex_sqrt_Shooting" in repr_str:
        return Simplex_sqrt_Metamorphosis_integrator, Simplex_sqrt_Shooting
    else:
        raise ValueError("No class found for the given repr_str")


def find_kernelOp_from_repr(repr_str):
    if "GaussianRKHS" in repr_str:
        return GaussianRKHS
    if "VolNormalizedGaussianRKHS" in repr_str:
        return VolNormalizedGaussianRKHS
    if "Multi_scale_GaussianRKHS" in repr_str:
        return Multi_scale_GaussianRKHS
    else:
        raise ValueError("no existing kernelOperator was found for the given repr_str")


def load_optimize_geodesicShooting(file_name, path=None, verbose=True):
    """load previously saved optimisation in order to plot it later."""

    # import pickle
    import io

    class CPU_Unpickler(pickle.Unpickler):
        """usage :
        #contents = pickle.load(f) becomes...
        contents = CPU_Unpickler(f).load()
        """

        def find_class(self, module, name):
            # print(f"Unpickler DEBUG : module:{module}, name:{name}")
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location="cpu", weights_only=False)
            else:
                if module == "metamorphosis":
                    module = "my_metamorphosis.metamorphosis"
                if name == "metamorphosis_path":
                    name = "Metamorphosis_path"
                if name == "multi_scale_GaussianRKHS":
                    name = "Multi_scale_GaussianRKHS"
                # print('module :',module,' name : ', name)
                return super().find_class(module, name)

    if path is None:
        path = OPTIM_SAVE_DIR
    if not file_name in os.listdir(path):
        raise FileNotFoundError("File " + file_name + " does not exist in " + path)
    with open(path + file_name, "rb") as f:
        opti_dict = CPU_Unpickler(f).load()

    if opti_dict["light_save"]:
        print(
            "Optimisation was saved in light mode. "
            "We proceed to re-shoot from saved initial momentum."
            "Be aware that any modification to the code will affect the saved result."
        )
        new_optim = _load_light_optim(opti_dict, verbose)
    else:
        new_optim = _load_heavy_optim(opti_dict, verbose)

    if "landmarks" in opti_dict.keys():
        new_optim.compute_landmark_dist(
            opti_dict["landmarks"][0], opti_dict["landmarks"][1]
        )

    new_optim.loaded_from_file = file_name
    if verbose:
        print(f"New optimiser loaded ({file_name}) :\n", new_optim.__repr__())
    return new_optim


def _load_light_optim(opti_dict, verbose):

    ## Find with which class we are dealing with
    integrator, optimizer = find_meta_optimiser_from_repr(opti_dict["__repr__"])
    # ic(optimizer,integrator)
    # Reinitialize the kernelOperator
    kernelOp = find_kernelOp_from_repr(
        opti_dict["args"]["kernelOperator"]["name"]
    )
    ic(kernelOp)
    ic(opti_dict["args"]["kernelOperator"])
    kernelOp = kernelOp(**opti_dict["args"]["kernelOperator"])
    ic(kernelOp)
    # and inject it in the args
    opti_dict["args"]["kernelOperator"] = kernelOp
    ic(opti_dict["args"])
    ## Re-shoot the integration
    mp = integrator(**opti_dict["args"])

    plt.imshow(opti_dict["parameter"][0, 0].detach().cpu())
    mp.forward(
        opti_dict["source"],
        opti_dict["parameter"],
        save=True,
        plot=0,
        hamiltonian_integration = opti_dict["args"]["hamiltonian_integration"]
    )
    print(mp)

    # inject the shooting in the optimizer
    opti_dict["geodesic"] = mp
    # ic(opti_dict)

    # ic(opti_dict['geodesic'].sigma_v)
    opti_dict["hamiltonian_integration"] = opti_dict["args"]["hamiltonian_integration"]
    mr = optimizer(**opti_dict)
    mr.to_analyse = opti_dict["to_analyse"]

    return mr


def _load_heavy_optim(opti_dict, verbose):

    flag_JM = False

    _, optimizer = find_meta_optimiser_from_repr(opti_dict["__repr__"])
    if isinstance(optimizer, Weighted_joinedMask_Metamorphosis_Shooting):
        flag_JM = True

    # kernelOp = opti_dict["args"]["kernelOperator"]
    # ic(kernelOp)

    # opti_dict["mp"].kernelOperator = kernelOp
    if verbose:
        print("DT:", opti_dict["data_term"])
    if flag_JM:
        new_optim = optimizer(
            opti_dict["source"][:, 0][None],
            opti_dict["target"][:, 0][None],
            opti_dict["source"][:, 1][None],
            opti_dict["target"][:, 1][None],
            opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            data_term=opti_dict["data_term"],
            optimizer_method=opti_dict["optimizer_method_name"],
            hamiltonian_integration=opti_dict["args"]["hamiltonian_integration"],
        )

    else:
        new_optim = optimizer(
            opti_dict["source"],
            opti_dict["target"],
            opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            hamiltonian_integration=opti_dict["args"]["hamiltonian_integration"],
        )

    for k in FIELD_TO_SAVE[5:]:
        new_optim.__dict__[k] = opti_dict[k]

    return new_optim
