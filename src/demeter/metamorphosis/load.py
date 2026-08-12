"""
In this file you will find everything you need to load a previously saved optimisation.
"""

import pickle
from icecream import ic

from demeter.constants import *
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
from .affine import Affine_Metamorphosis_integrator, Affine_Metamorphosis_Optimizer
from .affine_decoupled import Affine_Decoupled_Metamorphosis_integrator, Affine_Decoupled_Metamorphosis_Optimizer
from .regression import MetamorphosisRegression
from .splines import MetamorphosisSplineIntegrator, MetamorphosisSplineOptimizer
from ..utils.reproducing_kernels import (
    GaussianRKHS,
    VolNormalizedGaussianRKHS,
    Multi_scale_GaussianRKHS, DummyKernel, SobolevFluidOperator,
)


def _find_meta_optimiser_from_repr_(repr_str):
    if "MetamorphosisSplineOptimizer" in repr_str:
        return MetamorphosisSplineIntegrator, MetamorphosisSplineOptimizer
    if "MetamorphosisRegression" in repr_str:
        return Metamorphosis_integrator, MetamorphosisRegression
    if "ConstrainedMetamorphosis_Shooting" in repr_str:
        return ConstrainedMetamorphosis_integrator, ConstrainedMetamorphosis_Shooting
    if "Metamorphosis_Shooting" in repr_str:
        return Metamorphosis_integrator, Metamorphosis_Shooting
    if "Reduce_field_Optim" in repr_str:
        return ConstrainedMetamorphosis_integrator, Reduce_field_Optim
    if "Optimize_weighted_joinedMask" in repr_str:
        return (
            Weighted_joinedMask_Metamorphosis_integrator,
            Weighted_joinedMask_Metamorphosis_Shooting,
        )
    if "Simplex_sqrt_Shooting" in repr_str:
        return Simplex_sqrt_Metamorphosis_integrator, Simplex_sqrt_Shooting
    if "RigidMetamorphosis_Optimizer" in repr_str or "Affine_Decoupled_Metamorphosis_Optimizer" in repr_str:
        return Affine_Decoupled_Metamorphosis_integrator, Affine_Decoupled_Metamorphosis_Optimizer
    if "Affine_Metamorphosis_Optimizer" in repr_str:
        return Affine_Metamorphosis_integrator, Affine_Metamorphosis_Optimizer
    else:
        raise ValueError(f"No class found for the given repr_str : {repr_str}")


def _find_kernelOp_from_repr_(repr_str):
    if "SobolevFluidOperator" in repr_str:
        return SobolevFluidOperator
    if "VolNormalizedGaussianRKHS" in repr_str:
        return VolNormalizedGaussianRKHS
    if "Multi_scale_GaussianRKHS" in repr_str:
        return Multi_scale_GaussianRKHS
    if "GaussianRKHS" in repr_str:
        return GaussianRKHS
    if "DummyKernel" in repr_str:
        return DummyKernel
    else:
        raise ValueError("no existing kernelOperator was found for the given repr_str")


def _extract_analysis_results(opti_dict):
    integration_diverged = False
    if ("optimized_momenta" in opti_dict) or ("loss_stock" in opti_dict):
        optimized_momenta = opti_dict.get("optimized_momenta")
        loss_stock = opti_dict.get("loss_stock")
    elif "to_analyse" in opti_dict:
        analysis = opti_dict["to_analyse"]
        if isinstance(analysis, (tuple, list)) and len(analysis) == 2:
            optimized_momenta, loss_stock = analysis
        else:
            optimized_momenta, loss_stock = None, analysis
    else:
        optimized_momenta, loss_stock = None, None

    if isinstance(loss_stock, str) and loss_stock == "Integration diverged":
        integration_diverged = True
        loss_stock = None

    return optimized_momenta, loss_stock, integration_diverged


def _restore_optimizer_state(new_optim, opti_dict):
    for key in FIELD_TO_SAVE[5:]:
        if key == "data_term":
            continue
        value = opti_dict.get(key)
        if value is not None:
            setattr(new_optim, key, value)

    optimized_momenta, loss_stock, integration_diverged = _extract_analysis_results(
        opti_dict
    )
    if optimized_momenta is not None:
        new_optim.optimized_momenta = optimized_momenta
        new_optim.parameter = optimized_momenta
    if loss_stock is not None:
        new_optim.loss_stock = loss_stock
    if hasattr(new_optim.mp, "id_grid"):
        new_optim.id_grid = new_optim.mp.id_grid
    new_optim.integration_diverged = (
        getattr(new_optim, "integration_diverged", False)
        or integration_diverged
    )
    return new_optim


def _classic_integrator_arguments(args):
    arguments = {
        name: args[name]
        for name in (
            "method",
            "rho",
            "kernelOperator",
            "n_step",
            "dx_convention",
        )
    }
    if "boundary" in args:
        arguments["boundary"] = args["boundary"]
    return arguments


def _saved_optimizer_type(opti_dict):
    optimizer_class = opti_dict.get("optimizer_class")
    if optimizer_class is not None:
        return _find_meta_optimiser_from_repr_(optimizer_class)
    integrator, optimizer = _find_meta_optimiser_from_repr_(opti_dict["__repr__"])
    if optimizer in (MetamorphosisSplineOptimizer, MetamorphosisRegression):
        raise ValueError(
            "spline and regression saves must use the current versioned format"
        )
    return integrator, optimizer


def load_optimize_geodesicShooting(file_name, path=None, verbose=True):
    """
    load previously saved optimisation. Usually the file will be saved in the
     OPTIM_SAVE_DIR witch is by default `/saved_optim` .

    Parameters
    -------------
    file_name : str
        name of the file to load
    path : str, optional
        path to the file, by default OPTIM_SAVE_DIR = `saved_optim`
    verbose : bool, optional
        print the loaded optimiser / integrator _repr_, by default True

    Returns
    -------------
    new_optim : Metamorphosis_Shooting
        the loaded optimiser

    Examples
    -------------
    >>> import demeter.metamorphosis as mt
    >>> # load the optimiser
    >>> mr = mt.load_optimize_geodesicShooting('2D_23_01_2025_simpleToyExample_rho_0.00_000.pk1')


     """

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
        raise FileNotFoundError(f"File {file_name} does not exist in {path}")
    with open(os.path.join(path, file_name), "rb") as f:
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
    if "segmentations" in opti_dict.keys():
        new_optim.compute_DICE(opti_dict["segmentations"][0], opti_dict["segmentations"][1])

    new_optim.loaded_from_file = file_name
    if verbose:
        print(f"New optimiser loaded ({file_name}) :\n", new_optim.__repr__())
    return new_optim


def  _load_light_optim(opti_dict, verbose):

    ## Find with which class we are dealing with
    integrator, optimizer = _saved_optimizer_type(opti_dict)

    # Reinitialize the kernelOperator
    kernel_arguments = dict(opti_dict["args"]["kernelOperator"])
    kernel_name = kernel_arguments.pop("name")
    kernelOp = _find_kernelOp_from_repr_(kernel_name)
    ic(kernelOp)
    ic(kernel_arguments)
    if kernel_name == "DummyKernel":
        kernelOp = kernelOp()
    else:
        kernelOp = kernelOp(**kernel_arguments)

    # and inject it in the args
    opti_dict["args"]["kernelOperator"] = kernelOp
    ## Re-shoot the integration
    if optimizer is MetamorphosisSplineOptimizer:
        integrator_arguments = {
            name: opti_dict["args"][name]
            for name in (
                "rho",
                "control_times",
                "kernelOperator",
                "n_step",
                "cg_eps",
                "dx_convention",
            )
        }
        mp = integrator(**integrator_arguments)
    elif optimizer in (Metamorphosis_Shooting, MetamorphosisRegression):
        mp = integrator(**_classic_integrator_arguments(opti_dict["args"]))
    else:
        mp = integrator(**opti_dict["args"])
    optimized_momenta, _, _ = _extract_analysis_results(opti_dict)
    if (
        optimizer in (MetamorphosisSplineOptimizer, MetamorphosisRegression)
        and optimized_momenta is None
    ):
        raise ValueError(
            "spline and regression saves must contain optimized_momenta"
        )
    shooting_parameter = (
        optimized_momenta
        if optimized_momenta is not None
        else opti_dict.get("parameter")
    )
    if shooting_parameter is None:
        raise ValueError("saved optimization has no shooting parameter")
    print("Light save loaded : Reshooting integrator ...")
    mp.forward(
        opti_dict["source"],
        shooting_parameter,
        save=True,
        plot=0,
        hamiltonian_integration=opti_dict["args"].get(
            "hamiltonian_integration", False
        ),
    )
    # print(mp)

    # inject the shooting in the optimizer
    opti_dict["geodesic"] = mp

    opti_dict["hamiltonian_integration"] = opti_dict["args"]["hamiltonian_integration"]
    if optimizer is MetamorphosisSplineOptimizer:
        mr = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            target_times=opti_dict["args"]["target_times"],
            geodesic=mp,
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            lbfgs_max_iter=opti_dict["args"]["lbfgs_max_iter"],
            lbfgs_history_size=opti_dict["args"]["lbfgs_history_size"],
            temporal_preconditioning=opti_dict["args"][
                "temporal_preconditioning"
            ],
        )
    elif optimizer is MetamorphosisRegression:
        mr = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            target_times=opti_dict["args"]["target_times"],
            geodesic=mp,
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            lbfgs_max_iter=opti_dict["args"]["lbfgs_max_iter"],
            lbfgs_history_size=opti_dict["args"]["lbfgs_history_size"],
            hamiltonian_integration=opti_dict["args"]["hamiltonian_integration"],
            adam_scheduler=opti_dict["args"]["adam_scheduler"],
            adam_grad_clip=opti_dict["args"]["adam_grad_clip"],
        )
    elif optimizer is Metamorphosis_Shooting:
        mr = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            geodesic=mp,
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict.get(
                "optimizer_method_name", "LBFGS_torch"
            ),
            lbfgs_max_iter=opti_dict["args"].get("lbfgs_max_iter", 20),
            lbfgs_history_size=opti_dict["args"].get(
                "lbfgs_history_size", 100
            ),
            hamiltonian_integration=opti_dict["args"].get(
                "hamiltonian_integration", False
            ),
            adam_scheduler=opti_dict["args"].get("adam_scheduler"),
            adam_grad_clip=opti_dict["args"].get("adam_grad_clip"),
        )
    else:
        mr = optimizer(**opti_dict)
    opti_dict["parameter"] = shooting_parameter
    return _restore_optimizer_state(mr, opti_dict)


def _load_heavy_optim(opti_dict, verbose):

    flag_JM = False

    _, optimizer = _saved_optimizer_type(opti_dict)
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

    elif optimizer is MetamorphosisSplineOptimizer:
        new_optim = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            target_times=opti_dict["args"]["target_times"],
            geodesic=opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            lbfgs_max_iter=opti_dict["args"]["lbfgs_max_iter"],
            lbfgs_history_size=opti_dict["args"]["lbfgs_history_size"],
            temporal_preconditioning=opti_dict["args"][
                "temporal_preconditioning"
            ],
        )
    elif optimizer is MetamorphosisRegression:
        new_optim = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            target_times=opti_dict["args"]["target_times"],
            geodesic=opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            lbfgs_max_iter=opti_dict["args"]["lbfgs_max_iter"],
            lbfgs_history_size=opti_dict["args"]["lbfgs_history_size"],
            hamiltonian_integration=opti_dict["args"]["hamiltonian_integration"],
            adam_scheduler=opti_dict["args"]["adam_scheduler"],
            adam_grad_clip=opti_dict["args"]["adam_grad_clip"],
        )
    elif optimizer is Metamorphosis_Shooting:
        new_optim = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            geodesic=opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict.get(
                "optimizer_method_name", "LBFGS_torch"
            ),
            lbfgs_max_iter=opti_dict["args"].get("lbfgs_max_iter", 20),
            lbfgs_history_size=opti_dict["args"].get(
                "lbfgs_history_size", 100
            ),
            hamiltonian_integration=opti_dict["args"].get(
                "hamiltonian_integration", False
            ),
            adam_scheduler=opti_dict["args"].get("adam_scheduler"),
            adam_grad_clip=opti_dict["args"].get("adam_grad_clip"),
        )
    else:
        new_optim = optimizer(
            source=opti_dict["source"],
            target=opti_dict["target"],
            geodesic=opti_dict["mp"],
            cost_cst=opti_dict["cost_cst"],
            optimizer_method=opti_dict["optimizer_method_name"],
            hamiltonian_integration=opti_dict["args"]["hamiltonian_integration"],
        )

    return _restore_optimizer_state(new_optim, opti_dict)
