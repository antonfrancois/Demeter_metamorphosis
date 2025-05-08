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

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def perform_ref_of_size(size, save_gpu):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    S = tb.reg_open('m0t', size=size).to(device)
    T = tb.reg_open('m1c', size=size).to(device)
    sigma = rk.get_sigma_from_img_ratio(T.shape, subdiv=10)
    kernelOperator = rk.GaussianRKHS(sigma, kernel_reach=4)
    data_cost = mt.Ssd(T)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    try:
        start = time.time()
        mt.metamorphosis(S, T, 0, 0.05, cost_cst=0.001, kernelOperator=kernelOperator,
                         integration_steps=10, n_iter=15, grad_coef=1,
                         dx_convention='square', data_term=data_cost,
                         hamiltonian_integration=True, save_gpu_memory=save_gpu)
        torch.cuda.synchronize()
        exec_time = time.time() - start
        mem_usage = torch.cuda.max_memory_allocated()

        print('-_'*15)
        print("size : ",size, "save gpu", True)
        print("memory used : " ,convert_size(mem_usage))
        print('-_'*15)
        print("\n")
        return mem_usage, exec_time
    except torch.OutOfMemoryError:
        return None, None

if __name__ == "__main__":
    assert len(sys.argv) == 5, "Usage: python bench_execute_meta.py width height save_gpu csv_path"
    width, height = int(sys.argv[1]), int(sys.argv[2])
    save_gpu = sys.argv[3].lower() == 'true'
    csv_path = sys.argv[4]

    mem, time_exec = perform_ref_of_size((width, height), save_gpu)
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([width, height, save_gpu, mem if mem else "OOM", time_exec if time_exec else "OOM"])
