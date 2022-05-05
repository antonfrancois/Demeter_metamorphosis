import __init__
import os, sys, time
sys.path.append(os.path.abspath('../'))
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,10)
import torch
import image_3d_visualisation as i3v
from constants import *
import numpy as np
# import my_torchbox as tb
from my_toolbox import update_progress,format_time
# import my_metamorphosis.metamorphosis as mt


# %load_ext autoreload
# %autoreload 2
import metamorphosis as mt
import my_torchbox as tb

cuda = torch.cuda.is_available()
device = 'cpu'
if cuda:
    device = 'cuda:0'
torch.autograd.set_detect_anomaly(False)
print('device used :',device)

path = ROOT_DIRECTORY+"/im3Dbank/"
source_name = "source_3D_toyExample"
target_name = "target_3D_toyExample"
seg_name = "segmentation_3D_toyExample"
source = torch.load(path+source_name+".pt").to(device)
target = torch.load(path+target_name+".pt").to(device)
seg = torch.load(path+seg_name+".pt").to(device)
print(source.shape)
print(seg.shape)

sampler_step = 3
source = source[:,:,::sampler_step,::sampler_step,::sampler_step]
target = target[:,:,::sampler_step,::sampler_step,::sampler_step]
seg = seg[:,:,::sampler_step,::sampler_step,::sampler_step]


if False:
    S = tb.addGrid2im(S,30,method='lines')
    T = tb.addGrid2im(T,30,method='lines')


#%%


print('T :',target.max(),' S :',source.max())
## Construct the target image
ini_ball,_ = tb.make_ball_at_shape_center(seg,overlap_threshold=.06,verbose=True)


# i3v.compare_3D_images_vedo(seg,ini_ball)
ini_ball = ini_ball.to(device)
new_optim = False
if new_optim:
    residuals = torch.zeros(seg.shape[2:],device=device)
    print(residuals.shape)
    residuals.requires_grad = True

    n_step = 15
    lamb = 0.0001
    sigma = 3
    start = time.time()
    mp = mt.Metamorphosis_path(method='semiLagrangian',
                                  mu=0,rho=0,
                                  sigma_v=(sigma,sigma,sigma),
                                  n_step=n_step)
    mr = mt.Optimize_metamorphosis(ini_ball,seg,mp,cost_cst=lamb,
                                           optimizer_method='adadelta')
    mr.forward(residuals,n_iter=2000,grad_coef=1)
    end = time.time()
    print("Computation done in ",format_time(end - start)," s")

    # i3v.Visualize_geodesicOptim(mr)
    mr.save('ini_ball','seg_3D_tE','good mask')
else:
    mr = mt.load_optimize_geodesicShooting("3D_02_05_2022_ini_ball_to_seg_3D_tE_001.pk1")

#%% ======================================================
#     Weighted Metamorphosis
#   =======================================================

## Maintenant au boulot
mask = mr.mp.image_stock
mu = .2
rho = 10
sigma = 4
rf = mt.Residual_norm_identity(mask.to(device),mu,rho)
# rf = mt.Residual_norm_borderBoost(mask.to(device),mu,rho)
print(rf)

residuals = torch.zeros(seg.shape[2:],device=device)
residuals.requires_grad = True
rf = mt.Residual_norm_identity(mask.to(device),mu,rho)
mp_weighted = mt.Weighted_meta_path(
    rf,sigma_v=(sigma,sigma,sigma),n_step=mask.shape[0],
    border_effect = False
)

# mp_weighted.rf.dt_F_mask.shape
# mp_weighted.forward(source,residuals)
# mp_weighted.plot()
mr_weighted = mt.Weighted_optim(
    # target,source,mp_weighted,
    source,target,mp_weighted,
    cost_cst=0.0001,optimizer_method='adadelta'
)
start = time.time()
mr_weighted.forward(residuals,n_iter=3000,grad_coef=1)
end = time.time()
print(f"Computation done in {format_time(end - start)}")
mr_weighted.save('source','playExample','WM test')
