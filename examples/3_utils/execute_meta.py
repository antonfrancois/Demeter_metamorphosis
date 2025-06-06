import argparse
import sys
import os


from time import time

import math
import csv
import torch
import numpy as np
from demeter import *
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
from demeter.utils.decorators import *
from demeter.utils.toolbox import convert_bytes_size

@monitor_gpu
def perform_ref_of_size(size, save_gpu, n_iter, n_step, lbfgs_history_size,  lbfgs_max_iter):
    # print(f"Before putting S,T on GPU : GPU memory used: {gpus[0].memoryUsed} MB")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    bef_mem = torch.cuda.max_memory_allocated()
    S = tb.reg_open('m0t', size=size).to(device)
    after_mem = torch.cuda.max_memory_allocated()
    T = tb.reg_open('m1c', size=size).to(device)

    size_of_S = after_mem - bef_mem
    print(f"size of the image : {size_of_S/1024 ** 2:.2f} MB")

    sigma = rk.get_sigma_from_img_ratio(T.shape, subdiv=10)
    kernelOperator = rk.GaussianRKHS(sigma, kernel_reach=4)
    data_cost = mt.Ssd(T)
    # print(f"After putting S,T on GPU : GPU memory used: {gpus[0].memoryUsed} MB")

    print(S.dtype)

    try:
        torch.cuda.memory._record_memory_history()
        start = time()
        mt.metamorphosis(S, T, 0, 0.05,
                         cost_cst=0.001,
                         kernelOperator=kernelOperator,
                         integration_steps=n_step,
                         n_iter=n_iter,
                         data_term=data_cost,
                         hamiltonian_integration=True,
                         save_gpu_memory=save_gpu,
                         dx_convention='square',
                         grad_coef=1,
                         optimizer_method="LBFGS_torch",
                         lbfgs_history_size= lbfgs_history_size,
                         lbfgs_max_iter=lbfgs_max_iter,
                         )
        torch.cuda.synchronize()
        exec_time = time() - start

        save_at = f'memory_snapshots/classic_s{size[0]}_sg{save_gpu}_ni{n_iter}_ns{n_step}_lh{lbfgs_history_size}_li{lbfgs_max_iter}.pickle'
        torch.cuda.memory._dump_snapshot(save_at)
        print(f"snapshot saved at {save_at}")


        mem_allocated = torch.cuda.max_memory_allocated()
        mem_reserved = torch.cuda.max_memory_reserved()
        # print(f"In : GPU memory used: {gpus[0].memoryUsed} MB")
        print('-_'*15)
        print("size : ",size, "save gpu", save_gpu)
        print("max memory allocated : " ,convert_bytes_size(mem_allocated))
        print("max memory reserved : " ,convert_bytes_size(mem_reserved))
        print('-_'*15)
        print("\n")
        return size_of_S, mem_allocated, mem_reserved, exec_time
    except torch.OutOfMemoryError:
        return size_of_S, None, None, None


# print(f"Before : GPU memory used: {gpus[0].memoryUsed} MB")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPU Metamorphosis Benchmark")

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

    print(f"python bench_execute_meta.py "
            f"\n\twidth = {width}" 
            f"\n\theight = {height}" 
            f"\n\tsave_gpu={save_gpu}"
            f"\n\tn_iter = {n_iter}" 
            f"\n\tn_step = {n_step}" 
            f"\n\tlbfgs_history_size = {lbfgs_history_size}" 
            f"\n\tlbfgs_max_iter = {lbfgs_max_iter}" 
            f"\n\tcsv_path = {csv_path}"
          )
    # print(f"Before : GPU memory used: {gpus[0].memoryUsed} MB")

    # image_mem_size, mem, time_exec = perform_ref_of_size(
    #     (width, height),
    #     save_gpu,
    #     n_iter,
    #     n_step,
    #     lbfgs_history_size,
    #     lbfgs_max_iter
    # )

    image_mem_size, mem_allocated, mem_reserved, time_exec = perform_ref_of_size(
        (width, height),
        save_gpu,
        n_iter,
        n_step,
        lbfgs_history_size,
        lbfgs_max_iter
    )

    # print(f"After : GPU memory used: {gpus[0].memoryUsed} MB")

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            (1,1,width,height),
            image_mem_size,
            save_gpu,
            n_iter,
            n_step,
            lbfgs_history_size,
            lbfgs_max_iter,
            mem_allocated if mem_allocated else "OOM",
            mem_reserved if mem_reserved else "OOM",
            time_exec if time_exec else "OOM",
        ])
