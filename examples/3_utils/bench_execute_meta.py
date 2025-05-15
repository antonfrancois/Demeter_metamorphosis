import argparse
import sys
import os
import time
import math
import csv
import torch
import numpy as np
from demeter import *
import demeter.utils.reproducing_kernels as rk
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
from demeter.utils.decorators import *



def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
import time

@monitor_gpu
def perform_ref_of_size(size, save_gpu, n_iter, n_step):
    # print(f"Before putting S,T on GPU : GPU memory used: {gpus[0].memoryUsed} MB")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    bef_mem = torch.cuda.max_memory_allocated()
    # print(f"Before S,T; Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    # print(f"Before S,T;Peak reserved:  {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    S = tb.reg_open('m0t', size=size).to(device)
    after_mem = torch.cuda.max_memory_allocated()
    T = tb.reg_open('m1c', size=size).to(device)
    # print(f"After S,T; Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    # print(f"After S,T;Peak reserved:  {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

    size_of_S = after_mem - bef_mem
    print(f"size of the image : {size_of_S/1024 ** 2:.2f} MB")

    sigma = rk.get_sigma_from_img_ratio(T.shape, subdiv=10)
    kernelOperator = rk.GaussianRKHS(sigma, kernel_reach=4)
    data_cost = mt.Ssd(T)
    # print(f"After putting S,T on GPU : GPU memory used: {gpus[0].memoryUsed} MB")

    print(S.dtype)

    try:
        start = time.time()
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
                         lbfgs_history_size=20,
                         lbfgs_max_eval=10,
                         )
        torch.cuda.synchronize()
        exec_time = time.time() - start
        mem_usage = torch.cuda.max_memory_allocated()
        # print(f"In : GPU memory used: {gpus[0].memoryUsed} MB")
        print('-_'*15)
        print("size : ",size, "save gpu", save_gpu)
        print("memory used : " ,convert_size(mem_usage))
        print('-_'*15)
        print("\n")
        return size_of_S, mem_usage, exec_time
    except torch.OutOfMemoryError:
        return size_of_S, None, None


# print(f"Before : GPU memory used: {gpus[0].memoryUsed} MB")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPU Metamorphosis Benchmark")

    parser.add_argument("--width", type=int, default = 200, help="Image width")
    parser.add_argument("--height", type=int, default = 200, help="Image height")
    parser.add_argument("--save_gpu", type=str, default = "False", help="Whether to save GPU info (true/false)")
    parser.add_argument("--n_iter", type=int, default = 10, help="Number of iterations")
    parser.add_argument("--n_step", type=int, default = 10, help="Number of steps")
    parser.add_argument("--lbfgs_history_size", type=int, default = 100, help="L-BFGS history size")
    parser.add_argument("--lbfgs_max_eval", type=int, default = 20, help="L-BFGS max evaluations")
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
    lbfgs_max_eval = args.lbfgs_max_eval
    csv_path = args.csv_file

    print(f"python bench_execute_meta.py "
            f"\n\twidth = {width}" 
            f"\n\theight = {height}" 
            f"\n\tsave_gpu={save_gpu}"
            f"\n\tn_iter = {n_iter}" 
            f"\n\tn_step = {n_step}" 
            f"\n\tlbfgs_history_size = {lbfgs_history_size}" 
            f"\n\tlbfgs_max_eval = {lbfgs_max_eval}" 
            f"\n\tcsv_path = {csv_path}"
          )
    # print(f"Before : GPU memory used: {gpus[0].memoryUsed} MB")

    image_mem_size, mem, time_exec = perform_ref_of_size((width, height), save_gpu, n_iter, n_step)

    # print(f"After : GPU memory used: {gpus[0].memoryUsed} MB")


    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([width, height, save_gpu, mem if mem else "OOM", time_exec if time_exec else "OOM"])
