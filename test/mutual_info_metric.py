import __init__
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from utils.constants import ROOT_DIRECTORY
from utils.cost_functions import Mutual_Information
from utils.torchbox import imCmp


# - set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'


# Choose dataset
# ----------------------------------------------------------------------
def open_MRI_data():
    inFolder = ROOT_DIRECTORY + '/../data/bratsreg_2022/BraTSReg_Training_Data_v3/BraTSReg_007/'
    t1_name = 'BraTSReg_007_00_0000_flair.nii.gz'

    # t2_name = t1_name
    t2_name =  'BraTSReg_007_00_0000_t1.nii.gz'
    # t2_name = 'BraTSReg_007_01_0373_flair.nii.gz'
    # t2_name = 'BraTSReg_007_01_0373_t1.nii.gz'

    # Load the data
    t1_img = nib.load(os.path.join(inFolder, t1_name))
    t1_data = t1_img.get_fdata()
    t2_img = nib.load(os.path.join(inFolder, t2_name))
    t2_data = t2_img.get_fdata()
    print("t1_data.shape : ",t1_data.shape)
    print("T1 data min: ",t1_data.min()," max: ",t1_data.max())

    # normalize color range
    t1_data = t1_data / np.max(t1_data.ravel())
    t2_data = t2_data / np.max(t2_data.ravel())

    # Convert numpy to torch tensors and add channels dimension
    t1_data_torch = torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0)
    t2_data_torch = torch.tensor(t2_data, dtype=torch.float32).unsqueeze(0)

    # Require gradients for back-prop
    t1_data_torch.requires_grad = True
    t2_data_torch.requires_grad = True
    return t1_data_torch, t2_data_torch

#%%
Nbins = 40
min = 0
max = 1
mutual_info = Mutual_Information(bins=Nbins, min=min, max=max, sigma=100, reduction='sum')


#%% Basic test  ----------------------------------------------------------------------


t1_data_torch, t2_data_torch = open_MRI_data()


tic = time.time()
mi_p = mutual_info(t1_data_torch, t2_data_torch, plot=False)
toc = time.time()
print(f"mutual computation time : {toc - tic}s")
tic = time.time()
mi_p.backward()
toc = time.time()
print(f"backward computation time : {toc - tic}s")

if t1_data_torch.grad is not None:
    print('\t Test backward passed !')
    print('Max gradients:')
    print(t1_data_torch.grad.max())
    print(t2_data_torch.grad.max())

#%% Two balls overlap ----------------------------------------------------------------------


def make_ball(center, radius):
    im = torch.zeros((1,1, 100, 100))
    yy,xx = torch.meshgrid(torch.arange(100),torch.arange(100))
    im[0,0] = (xx-center[0])**2 + (yy-center[1])**2 #< radius**2
    # im[0,0] = (im[0,0].max() - im[0,0])/im[0,0].max()
    im[0,0] = torch.exp(-im[0,0]/(2*radius**2))
    return im

# Create two balls
yy,xx = np.meshgrid(np.arange(100),np.arange(100))
ball_dict = [
    # no overlap
    {
    "name" : "no overlap",
    "center_1" : (25,50),
    "radius_1" : 10,
    "center_2": (75,50),
    "radius_2": 10
    },
    # small overlap
    {
    "name" : "small overlap",
    "center_1" : (50,50),
    "radius_1" : 20,
    "center_2": (70,70),
    "radius_2": 20
    },

    # full overlap
    {
    "name" : "full overlap",
    "center_1" : (50,50),
    "radius_1" : 10,
    "center_2": (50,50),
    "radius_2": 10
    },
]

for coord in ball_dict:
    center_1 = coord["center_1"]
    radius_1 = coord["radius_1"]
    center_2 = coord["center_2"]
    radius_2 = coord["radius_2"]

    print(f"Case : {coord['name']}")

    im1 = make_ball(center_1, radius_1)
    im2 = make_ball(center_2, radius_2)

    fig, ax = plt.subplots(3,1,figsize=(5,15))
    a = ax[0].imshow(im1[0,0])
    fig.colorbar(a, ax=ax[0])
    b = ax[1].imshow(im2[0,0])
    fig.colorbar(b, ax=ax[1])
    ax[2].imshow(imCmp(im1,im2, method = 'seg'))
    plt.show()

    tic = time.time()
    mi_p = mutual_info(im1, im2, plot=False)
    toc = time.time()
    print(f"mutual computation time : {toc - tic}s")
    print(f"MI score : {mi_p}")