"""

"""
import matplotlib.pyplot as plt
import torch
from math import log
from demeter.utils.toolbox import convert_bytes_size
# from execute_meta import perform_ref_of_size

def predict_gpu_model_image(
        image,
        integration_steps,
        n_iter,
        lbfgs_max_iter,
        lbfgs_history_size,
        parameters,
):
    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]
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

def max_image_size_for_params(
        gpu_memory,
        integration_steps,
        n_iter,
        lbfgs_max_iter,
        lbfgs_history_size,
        parameters):
    """ gives the predicted maximum image size according to a set of parameters

    D_max =  (y - c) / (aM + bN)

    """

    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]

    M = min(lbfgs_max_iter * n_iter, lbfgs_history_size)

    return  (gpu_memory - c) /  (a * M + b*integration_steps)


# ====================================================
# Estimated parameters using understanding_GPU.py

PARAM_CLASSIC_200_TRUE = {
    'a':1.95011778, 'b': 1.85643221, 'c':22197766, 'save_gpu': True
}

PARAM_CLASSIC_200_FALSE = {
    'a':1.94818074, 'b': 34.86387044, 'c':15754640, 'save_gpu': False
}

PARAM_CLASSIC_ALL_TRUE = {
    # 'a':2.51876377 , 'b':6.57725367, 'c':19109046, 'save_gpu': True
    'a':2.77470059 , 'b':7.25764855, 'c':20496658, 'save_gpu': True
    # 'a': 3.06239906, 'b': 8.69422842, 'c': 23337559.0, 'r2': 0.921863, 'save_gpu': True
}

PARAM_CLASSIC_ALL_FALSE = {
    # 'a':1.65921669, 'b':64.26293817, 'c':438641, 'save_gpu': False
    'a':2.47526598, 'b':38.84920446, 'c':12498300, 'save_gpu': False
    # 'a': 2.61022542, 'b': 39.20531525, 'c': 14595235.0, 'r2': 0.995706, 'save_gpu': False
}

PARAM_SIMPLEX_ALL_TRUE = {
    'a':1.8872569 , 'b':4.46558841, 'c': 1966816249, 'save_gpu': True

}

PARAM_SIMPLEX_ALL_FALSE = {
    'a': 1.87025701,  'b' : 15.73652725, 'c':1216915863, 'save_gpu': False
}



#======================================================
if __name__ == '__main__':
#%%

    size = 2000
    integration_steps= 10
    n_iter= 10
    lbfgs_max_iter= 20
    lbfgs_history_size= 100
    parameters = PARAM_CLASSIC_ALL_TRUE

    img = torch.rand((size,size),dtype= torch.float32)
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

    #%%
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


    #%% Fake benchmark
    # import numpy as np

    size_list = torch.linspace(100,5000, 10)
    pred_true_list = []
    pred_false_list = []
    img_size_list = []
    for size in size_list:
        img = torch.rand((int(size),int(size)),dtype= torch.float32)
        img_size_list.append(int(size)**2)
        pred_false =  predict_gpu_model_image(img,
                integration_steps,
                n_iter,
                lbfgs_max_iter,
                lbfgs_history_size,
                PARAM_CLASSIC_ALL_FALSE
                                )
        pred_true =  predict_gpu_model_image(img,
                integration_steps,
                n_iter,
                lbfgs_max_iter,
                lbfgs_history_size,
                PARAM_CLASSIC_ALL_TRUE
                                )
        pred_true_list.append(pred_true)
        pred_false_list.append(pred_false)

    fig, ax = plt.subplots()
    ax.plot(img_size_list, pred_true_list, 'b')
    ax.plot(img_size_list, pred_false_list, 'r')
    ax.legend()
    ax.grid(True)
    plt.show()
