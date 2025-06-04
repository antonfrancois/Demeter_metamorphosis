import argparse

import __init__
import os, math, time
import re
import sys, csv
import torch

import  demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
from demeter.constants import ROOT_DIRECTORY
from demeter.utils.toolbox import convert_bytes_size
from demeter.utils.torchbox import resize_image
import demeter.utils.image_3d_plotter as i3p
import matplotlib.pyplot as plt
import nibabel as nib

# open simplex

ORDER  = [
    "necrosis",             #1
    "wm",                   #4
    "basal",                #7
    "ventricles",       #6
    "gm",                   #5
    "gde",                      #3
    "thalamus",         #8
    "wmh_edema",       #2
]

def open_nii(path):
    img = nib.load(path)
    # affine = img.affine
    # return img.get_fdata(), affine
    img_ras = nib.as_closest_canonical(img)
    return img_ras.get_fdata()

def find_open_mri(path):
    """
    Find the open mri file in the path
    :param path: str
    :return: str
    """
    def find():
        for file in os.listdir(path):
            if 'FLAIR' in file:
                return file

    path_flair = os.path.join(path,find())
    return nib.load(path_flair).get_fdata()

def open_probabilities(path, key='prob'):
    """
    Open the probabilities files
    :param path: str
    :return: np.array
    """
    dict_list = []
    for file in os.listdir(path):
        if key not in file:
            continue
        d = {
            'name': file,
            # 'data': nib.load(os.path.join(path,file)).get_fdata()
            'data': open_nii(os.path.join(path,file)),
        }
        dict_list.append(d)
    return dict_list


def probability_to_simplex(probabilities,
                           key : str ='prob',
                           resize_factors : tuple[float] = None,
                           to_shape : tuple[int] = None
                           ):
    d = len(probabilities)

    # simplex = np.zeros((d,) + proba[0]['data'].shape)
    proba_f = []
    pattern = re.compile(fr"{key}_([a-z_]*).nii.gz")
    for o in ORDER:
        for p in probabilities:
            file_id = pattern.search(p["name"])
            if file_id:
                result = file_id.group(1)
            else:
                result = 'non'
                continue

            if o == result:
                # print(f"\tAdding : {p["name"]}")
                proba_f.append(p)
                continue
    # print("\t len(proba_f)",len(proba_f))
    # for pf in proba_f:
    #     print(pf['name'])
    if resize_factors is None and to_shape is None:
        simplex = torch.stack([
            torch.tensor(proba['data'].copy()).clip( 0,1)
            for proba in proba_f
        ],dim=0)[None]
    elif resize_factors is not None and to_shape is not None:
        raise ValueError(f"ResizeFactors and to_shape are mutually exclusive, got resize_factor : {resize_factors} and to_shape : {to_shape}, only one of them has to be provided")
    else:
        # check resize
        img_list = []
        for proba in proba_f:
                img  = torch.tensor(proba['data'].copy())[None, None]
                # ic(img.shape)
                re_img = resize_image(img, scale_factor = resize_factors, to_shape=to_shape)
                re_img = re_img.clip(0, 1)
                # ic(re_img.shape)
                img_list.append( re_img )
        simplex = torch.cat(
            img_list,dim=1
        )

    sum_simplex = simplex.sum(dim=1,keepdim=True).clip(0,1)
    # complete the simplex with a background class
    simplex = torch.cat([simplex, 1 - sum_simplex],dim=1)
    return simplex

def path_to_simplex(path, key='prob', resize_factors = None, to_shape = None):
    probabilities = open_probabilities(path, key)

    simplex = probability_to_simplex(probabilities, key, resize_factors=resize_factors, to_shape=to_shape)
    return simplex

def plot_simplex(image, slc):
    image_rgb =  i3p.SimplexToHSV(image,is_last_background=True).to_rgb()
    # source_rgb = i3p.SimplexToHSV(target,is_last_background=True).to_rgb()

    fig, ax = plt.subplots(1,3, constrained_layout=True, figsize=(15,5))
    ax[0].imshow(image_rgb[0,:,:,slc[2]].transpose(1,0,2), aspect="auto", origin="lower")
    ax[1].imshow(image_rgb[0,:, slc[1], :].transpose(1,0,2), aspect="auto", origin="lower")
    ax[2].imshow(image_rgb[0,slc[0],:,:].transpose(1,0,2), aspect="auto", origin="lower")

    line_color = "red"
    _l_x_v = ax[0].axvline(x=slc[0], color=line_color, alpha=0.6)
    _l_x_h = ax[0].axhline(y=slc[1], color=line_color, alpha=0.6)
    _l_y_v = ax[1].axvline(x=slc[0], color=line_color, alpha=0.6)
    _l_y_h = ax[1].axhline(y=slc[2], color=line_color, alpha=0.6)
    _l_z_v = ax[2].axvline(x=slc[1], color=line_color, alpha=0.6)
    _l_z_h = ax[2].axhline(y=slc[2], color=line_color, alpha=0.6)

    # title = name+ f" t = {t}"
    # fig.suptitle(title, fontsize=20)
    # fig.savefig("imgs/pixyl_reg/"+title+".png")
    plt.show()

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def plot_mr(mr):
    slc = tuple( [int(s * r) for s, r in zip(slice, resize_factor)])
    rho = mr.mp.rho
    name = f"{patient}_{source_fol}_to_{target_fol}_meta_rho{rho}"
    if 'turtlefox' in ROOT_DIRECTORY:
        mr.plot_cost()
        plt.show()
        ic.disable()
        i3p.Visualize_GeodesicOptim_plt(mr, name)
        plt.show()
    else:
        out_path = os.path.join(
            "/gpfs/users/francoisa/RadioAide_Preprocessing",
            "imgs/pixyl_reg"
        )



        t_img = i3p.SimplexToHSV(mr.mp.image_stock.cpu(), is_last_background=True).to_rgb()
        ic(t_img.shape)
        # t = 4
        for t in range(t_img.shape[0]):
            fig, ax = plt.subplots(1,3, constrained_layout=True, figsize=(15,5))
            ax[0].imshow(t_img[t,:,:,slc[2]].transpose(1,0,2), aspect="auto", origin="lower")
            ax[1].imshow(t_img[t,:, slc[1], :].transpose(1,0,2), aspect="auto", origin="lower")
            ax[2].imshow(t_img[t,slc[0],:,:].transpose(1,0,2), aspect="auto", origin="lower")

            line_color = "red"
            _l_x_v = ax[0].axvline(x=slc[0], color=line_color, alpha=0.6)
            _l_x_h = ax[0].axhline(y=slc[1], color=line_color, alpha=0.6)
            _l_y_v = ax[1].axvline(x=slc[0], color=line_color, alpha=0.6)
            _l_y_h = ax[1].axhline(y=slc[2], color=line_color, alpha=0.6)
            _l_z_v = ax[2].axvline(x=slc[1], color=line_color, alpha=0.6)
            _l_z_h = ax[2].axhline(y=slc[2], color=line_color, alpha=0.6)

            title = name+ f" t = {t}"
            fig.suptitle(title, fontsize=20)
            fig.savefig(out_path+'/'+title+"_meso.png")
            plt.show()

    mr.save(name, light_save=True)


def perform_simplex_ref(img_shape, save_gpu,  n_iter, n_step, lbfgs_history_size,  lbfgs_max_iter):

    print("Building source & target ...")
    source = path_to_simplex(os.path.join(path,patient,f"{patient}_{source_fol}"),
                                # resize_factors=resize_factor,
                             # to_shape = img_shape,
                                key = 'LB_prob'
                                )
    target = path_to_simplex(
        os.path.join(path,patient,f"{patient}_{target_fol}"),
        # resize_factors=resize_factor,
        # to_shape = img_shape,
        key = 'LB_prob'
    )
    print("source : ", source.shape, source.min().item(), source.max().item())
    print("target; ", target.shape, target.min().item(), target.max().item())

    # resize
    before_shape =  source.shape
    source = resize_image(source, to_shape=img_shape)
    source = source.clip(0, 1)
    target = resize_image(target, to_shape=img_shape)
    target = target.clip(0, 1)
    print(before_shape,img_shape, source.shape)
    resize_factor  = [a / b for a, b in zip(img_shape, before_shape[2:])]
    print("resize_factor ",resize_factor)
    slc = tuple( [int(s * r) for s, r in zip(slice, resize_factor)])
    print(slc, slice)
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
    bef_mem = torch.cuda.max_memory_allocated()
    source = source.to(device)
    after_mem = torch.cuda.max_memory_allocated()
    target = target.to(device)

    print("source", source.min(), source.max())
    size_of_S = after_mem - bef_mem
    print(f"size of the image : {size_of_S/1024 ** 2:.2f} MB")
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
                       integration_steps=n_step,
                       n_iter=n_iter,
                       grad_coef=10,
                      dx_convention=dx_convention,
                      save_gpu_memory=save_gpu,
                     lbfgs_history_size=lbfgs_history_size,
                    lbfgs_max_iter=lbfgs_max_iter
        )
        ic(mr)
        torch.cuda.synchronize()
        exec_time = time.time() - start
        mem_allocated = torch.cuda.max_memory_allocated()
        mem_reserved = torch.cuda.max_memory_reserved()
        # print(f"In : GPU memory used: {gpus[0].memoryUsed} MB")
        print('-_'*15)
        print("size : ",size, "save gpu", save_gpu)
        print("max memory allocated : " ,convert_bytes_size(mem_allocated))
        print("max memory reserved : " ,convert_bytes_size(mem_reserved))
        print('-_'*15)
        print("\n")
        return  size_of_S, mem_allocated, mem_reserved, exec_time
    except torch.OutOfMemoryError:
        return size_of_S, None, None, None



    #%%


#%%

if __name__ == "__main__":
# if False:
    parser = argparse.ArgumentParser(description="Run GPU Simplex Metamorphosis on pixyl Benchmark")

    parser.add_argument("--width", type=int, default = 200, help="Image width")
    parser.add_argument("--height", type=int, default = 200, help="Image height")
    parser.add_argument("--save_gpu", type=str, default = "False", help="Whether to save GPU info (true/false)")
    parser.add_argument("--n_iter", type=int, default = 10, help="Number of iterations")
    parser.add_argument("--n_step", type=int, default = 10, help="Number of steps")
    parser.add_argument("--lbfgs_history_size", type=int, default = 100, help="L-BFGS history size")
    parser.add_argument("--lbfgs_max_iter", type=int, default = 20, help="L-BFGS max evaluations")
    parser.add_argument("--csv_file", type=str, default = "trash.csv", help="Path to output CSV")

    args = parser.parse_args()

    # Parse booleans
    save_gpu = args.save_gpu.lower() == "true"

    # Now you can use:
    width = args.width
    height = args.height
    n_iter = args.n_iter
    n_step = args.n_step
    lbfgs_history_size = args.lbfgs_history_size
    lbfgs_max_iter = args.lbfgs_max_iter
    csv_path = args.csv_file

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
    size = (width, height, 100)
    # resize_factor = (rf, rf, 1)

    patient, slice  = "PSL_001", (200,270,50)
    # patient, slice  = "PSL_007", (300,180,25)
    source_fol = "M21"
    target_fol = "M30"
    # source_fol = "M10"
    # target_fol = "M14"


#%%
    size_image, mem_allocated, mem_reserved, exec_time = perform_simplex_ref(size, save_gpu,n_iter,n_step,lbfgs_history_size,lbfgs_max_iter)

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            (1,1,width,height),
            size_image,
            save_gpu,
            n_iter,
            n_step,
            lbfgs_history_size,
            lbfgs_max_iter,
            mem_allocated if mem_allocated else "OOM",
            mem_reserved if mem_reserved else "OOM",
            time_exec if time_exec else "OOM",
        ])

