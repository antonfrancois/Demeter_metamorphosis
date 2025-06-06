"""

"""

import torch
from math import log
from demeter.utils.toolbox import convert_bytes_size
from execute_meta import perform_ref_of_size

def predict_gpu_model_image(
        image,
        integration_steps,
        n_iter,
        lbfgs_max_iter,
        lbfgs_history_size,
        parameters,
):
    a,b,c,_ = parameters.values()
    D = image.nbytes

    M = min(lbfgs_max_iter * n_iter, lbfgs_history_size)

    return (a * M + b*integration_steps) * D + c

def check_gpu():
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        # gpu_name = torch.cuda.get_device_name(device_index)
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
    else:
        total_memory = 0
    return total_memory

# ====================================================
# Estimated parameters using understanding_GPU.py

PARAM_LINREG_200_TRUE = {
    'a':1.95011778, 'b': 1.85643221, 'c':22197766, 'save_gpu': True
}

PARAM_LINREG_200_FALSE = {
    'a':1.94818074, 'b': 34.86387044, 'c':15754640, 'save_gpu': False
}

PARAM_LINREG_ALL_TRUE = {
    'a':0, 'b':0, 'c':0, 'save_gpu': True
}

PARAM_LINREG_ALL_FALSE = {
    'a':0, 'b':0, 'c':0, 'save_gpu': False
}


#======================================================

size = 200
img = torch.rand((size,size),dtype= torch.float32)
integration_steps= 20
n_iter= 3
lbfgs_max_iter= 6
lbfgs_history_size= 10
parameters = PARAM_LINREG_200_TRUE

pred =  predict_gpu_model_image(img,
        integration_steps,
        n_iter,
        lbfgs_max_iter,
        lbfgs_history_size,
        parameters
                        )

total_gpu_mem = check_gpu()

print(f"Prediction: {pred}, {convert_bytes_size(pred)}")
if pred > total_gpu_mem:
    print(f"With current GPU (memory available {convert_bytes_size(total_gpu_mem)})and parameters, the model will be out of memory.")

# =====================================================
# Check if the prediction is correct !

image_mem_size, mem_allocated, mem_reserved, time_exec = perform_ref_of_size(
    img.shape,
    parameters["save_gpu"],
    n_iter,
    integration_steps,
    lbfgs_history_size,
    lbfgs_max_iter
)

print('Parameters:')
print(f"\t image_size :{img.shape}")
print(f"\t n_iter :{n_iter}")
print(f"\t lbfgs_max_iter:{lbfgs_max_iter}")
print(f"\t lbfgs_history_size :{lbfgs_history_size}")
print(f"\t parameters :{parameters}")

print(f"Prediction: {pred}, {convert_bytes_size(pred)}")
if pred > total_gpu_mem:
    print(f"With current GPU (memory available {convert_bytes_size(total_gpu_mem)})and parameters, the model will be out of memory.")

if mem_allocated is not None:
    print(f"Memory allocated: {mem_allocated}, {convert_bytes_size(mem_allocated)}")
    print(f"Difference  abs : {abs(pred - mem_allocated)}, {convert_bytes_size(abs(pred - mem_allocated))}")
    print(f"log difference : {abs(log(pred) - log(mem_allocated))}")
else:
    print(f"Memory allocated: Out Of Memory")


