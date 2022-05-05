import __init__

import torch
import my_torchbox as tb
from my_toolbox import update_progress,format_time
import metamorphosis as mt
import image_3d_visualisation as i3v
import time
from constants import *

# cuda = torch.cuda.is_available()
cuda = True
device = 'cpu'
if cuda:
    device = 'cuda:0'
torch.autograd.set_detect_anomaly(False)
print('device used :',device)


path = ROOT_DIRECTORY+"/im3Dbank/"
source_name = "ball_for_hanse"
target_name = "hanse_w_ball"
S = torch.load(path+source_name+".pt").to(device)
T = torch.load(path+target_name+".pt").to(device)

# if the image is too big for your GPU, you can downsample it quite barbarically :
step = 2
if step > 0:
    S = S[:,:,::step,::step,::step]
    T = T[:,:,::step,::step,::step]
_,_,D,H,W = S.shape


residuals= torch.zeros((D,H,W),device = device)
residuals.requires_grad = True

reg_grid = tb.make_regular_grid(S.size(),device=device)

meta_path_method = 'semiLagrangian'
# mu = 0
# mu,rho,lamb = 0, 0, .0001   # LDDMM
mu,rho,lamb = 0.02, 1, .0001  # Metamorphosis

mp_sl = mt.Metamorphosis_path(method=meta_path_method,
                              mu=mu,rho=rho,
                              sigma_v=(4,4,4),
                              n_step=20)

mr_sl = mt.Optimize_metamorphosis(S,T,mp_sl,
                                     cost_cst=0.001,
                                     optimizer_method='adadelta')
start = time.time()
mr_sl.forward(residuals,n_iter=600,grad_coef=1)
end = time.time()

print(f"Computation done in {format_time(end - start)}")


deformation  = mr_sl.mp.get_deformation()
# mr_sl.plot_cost()
# plt.show()
# mr_sl.save(source_name,target_name)

i3v.Visualize_geodesicOptim(mr_sl,alpha=1)
plt_v = i3v.compare_3D_images_vedo(T,mp_sl.image_stock.cpu())
plt_v.show_deformation_flow(deformation,1,step=3)
plt_v.plotter.show(interactive=True).close()