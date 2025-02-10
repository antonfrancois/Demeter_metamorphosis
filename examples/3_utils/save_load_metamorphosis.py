"""
.. _save_load_metamorphosis:


This file is to test the save and load functions of the Metamorphosis class.
"""
################################################################################
# Import the necessary libraries

try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass


import torch

from demeter.constants import *
# %load_ext autoreload
# %autoreload 2
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

################################################################################
# Load the images and visualize them
print("Performing simple optimisation and saving it")

size = (200,200)

source_name,target_name = 'te_s','te_t'
S = tb.reg_open(source_name,size = size).to(device) #,location='bartlett')
T = tb.reg_open(target_name,size = size).to(device)
seg = tb.reg_open('te_c_seg',size=size).to(device)

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(S[0,0,...].detach().cpu().numpy(),**DLT_KW_IMAGE)
ax[0].set_title('Source')
ax[1].imshow(T[0,0,...].detach().cpu().numpy(),**DLT_KW_IMAGE)
ax[1].set_title('Target')
ax[2].imshow(seg[0,0,...].detach().cpu().numpy(),**DLT_KW_IMAGE)
ax[2].set_title('Segmentation')

# ini_ball,_ = tb.make_ball_at_shape_center(seg,overlap_threshold=.1,verbose=True)
# ini_ball = ini_ball.to(device)
# residuals = torch.zeros(seg.shape[2:],device=device)

mr = mt.metamorphosis(S,T,0,.03,1,5,0.0001,15,10,10)
# mr = mt.weighted_metamorphosis(S,T,0,seg,
#                                mu=1,rho= 10,
#                                rf_method = 'identity',
#                                sigma=15,
#                                cost_cst=0.0001,
#                                n_iter = 15,
#                                grad_coef= 1,
#                                data_term = data_term
#                                )

# mr.plot()

################################################################################
# If you want to save the optimisation along all parameter and integrated
# image, field and momentum, set light_save = True. Recommended if you are testing
# developping a new version of Metamorphosis. Otherwise, light_save = False, save
# only the initial momentum and the parameters used. Then the load function
# re-shoot a single integration.
file_normal,_ = mr.save(source_name,target_name,
                 message = 'Testing save',
                 light_save=False)
print(f"normal file saved at {file_normal}")

file_light,_ = mr.save(source_name,target_name,
                 message = 'Testing light save',
                 light_save=True)

print(f"light file saved at {file_light}")


#%%
################################################################################
# then one can try reloading the file.

# file_light = "some file"
mr_2 = mt.load_optimize_geodesicShooting(file_light)

mr_2.plot()

