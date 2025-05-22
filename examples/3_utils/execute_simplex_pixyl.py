import __init__
import os, math, time
import re
import sys, csv
import torch

import  demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
from demeter.constants import ROOT_DIRECTORY
import examples.pixyl_utils as pu
from demeter.utils.toolbox import convert_bytes_size


# open simplex



def perform_simplex_ref(resize_factor, save_gpu):
    slc = tuple( [int(s * r) for s, r in zip(slice, resize_factor)])

    source = pu.path_to_simplex(os.path.join(path,patient,f"{patient}_{source_fol}"),
                                resize_factors=resize_factor,
                                key = 'LB_prob'
                                )
    target = pu.path_to_simplex(
        os.path.join(path,patient,f"{patient}_{target_fol}"),
        resize_factors=resize_factor,
        key = 'LB_prob'
    )
    print("source : ", source.shape, source.min().item(), source.max().item())
    print("target; ", target.shape, target.min().item(), target.max().item())

    # if 'turtlefox' in ROOT_DIRECTORY:
    #     plot_simplex(source, slc)



    # subdiv = 10
    # sigma = rk.get_sigma_from_img_ratio(source.shape,subdiv = subdiv)
    # kernelOperator = rk.GaussianRKHS(sigma, kernel_reach=6, normalized=True)

    # sigma = (.01,.01,.1)
    # dx = tuple([1./(s-1) for s in source.shape[2:]])
    # kernelOperator = rk.VolNormalizedGaussianRKHS(
    #     sigma,
    #     sigma_convention="continuous",
    #     dx=dx,
    #     kernel_reach=6
    # )

    sigma = (.1,.1,.1)
    dx = tuple([1./(s-1) for s in source.shape[2:]])
    # dx = (1, 1, 1)
    k = 3
    kernelOperator = rk.All_Scale_Anisotropic_Normalized_Gaussian_RKHS(
            sigma=sigma,
            k = k,
            dx=dx,
            sigma_convention='continuous'
        )


    print(kernelOperator)
    print(kernelOperator.kernel.max())
    # rk.plot_gaussian_kernel_3d(kernelOperator.kernel, sigma=sigma)
    # plt.show()



    data_cost = mt.Ssd_normalized(target)
    # data_cost = None
    dx_convention = "pixel"
    source = source.to(device)
    target = target.to(device)

    print("source", source.min(), source.max())
    rho = 1
    ic(rho)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    try:
        start = time.time()

        # from torch.profiler import profile, record_function, ProfilerActivity

        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     profile_memory=True
        # ) as prof:
        #     with record_function("simplex_metamorphosis"):
        mr = mt.simplex_metamorphosis(source, target, 0,
                        rho= rho,
                       kernelOperator=kernelOperator,
                        data_term=data_cost,
                       cost_cst=.001,
                       integration_steps=10,
                       n_iter=25,
                       grad_coef=10,
                      dx_convention=dx_convention,
                      save_gpu_memory=save_gpu
        )
        ic(mr)
        torch.cuda.synchronize()
        exec_time = time.time() - start
        mem_usage = torch.cuda.max_memory_allocated()
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

        print('-_'*15)
        print("size : ",source.shape,  "save gpu", save_gpu)
        print("memory used : " ,convert_bytes_size(mem_usage))
        print('-_'*15)
        print("\n")
    except torch.OutOfMemoryError:
        print("OUT OF MEMORY")
        return None, None, None,  source.shape

    return mr, mem_usage, exec_time, source.shape


    #%%


#%%

if __name__ == "__main__":
# if False:
    assert len(sys.argv) == 5, "Usage: python execute_simplex_pixyl.py resize_factor save_gpu save_plot csv_path "
    rf = float(sys.argv[1])
    save_gpu = sys.argv[2].lower() == 'true'
    save_plot = sys.argv[3].lower() == 'true'
    csv_path = sys.argv[4]

    ic(
        rf,
        save_gpu,
        save_plot,
        csv_path,
    )

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "img shape",
                'resize factor',
                "save gpu",
                "mem usage bytes",
                "exec time sec",
            ])

    device = 'cuda:0'
    if 'turtlefox' in ROOT_DIRECTORY:
        path = "/home/turtlefox/Documents/11_metamorphoses/data/pixyl/aligned"
    else:
        path = "/gpfs/workdir/francoisa/data/aligned"

    # rf = .2
    resize_factor = (rf, rf, 1)

    patient, slice  = "PSL_001", (200,270,50)
    # patient, slice  = "PSL_007", (300,180,25)
    source_fol = "M21"
    target_fol = "M30"
    # source_fol = "M10"
    # target_fol = "M14"


#%%
    mr, mem, time_exec, img_shape = perform_simplex_ref(resize_factor, save_gpu)

    if save_plot:
        pu.plot_mr(mr, slice, resize_factor)
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            tuple(img_shape),
            resize_factor,
            save_gpu,
            mem if mem else "OOM",
            time_exec if time_exec else "OOM"
        ])



