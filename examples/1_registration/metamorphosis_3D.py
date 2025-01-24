"""
In this file we apply the metamorphosis algorithm to 3D images.

"""

try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass


import src.demeter.metamorphosis as mt
import src.demeter.utils.image_3d_visualisation as i3v
from src.demeter.utils import *

# cuda = torch.cuda.is_available()
cuda = True
device = 'cpu'
if cuda:
    device = 'cuda:0'
torch.autograd.set_detect_anomaly(False)
print('device used :',device)


path = ROOT_DIRECTORY+"/examples/im3Dbank/"
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

## Setting residuals to 0 is equivalent to writing the
## following line of code :
# residuals= torch.zeros((D,H,W),device = device)
# residuals.requires_grad = True
residuals = 0

# reg_grid = tb.make_regular_grid(S.size(),device=device)


# mu = 0
# mu,rho,lamb = 0, 0, .0001   # LDDMM

# print("Apply LDDMM")
# mr_lddmm = mt.lddmm(S,T,residuals,
#     sigma=(4,4,4),          #  Kernel size
#     cost_cst=0.001,         # Regularization parameter
#     integration_steps=10,   # Number of integration steps
#     n_iter=600,             # Number of optimization steps
#     grad_coef=1,            # max Gradient coefficient
#     data_term=None,         # Data term (default Ssd)
#     sharp=False,            # Sharp integration toggle
#     safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
#     integration_method='semiLagrangian',  # You should not use Eulerian for real usage
# )
# mr_lddmm.plot_cost()

# you can save the optimization:
# mr_lddmm.save(source_name,target_name)


#%%
mu,rho,lamb = 0.02, 1, .0001  # Metamorphosis
print("\nApply Metamorphosis")
mr_meta = mt.metamorphosis(S,T,residuals,
    mu=mu,                  # intensity addition coef
    rho=rho,                # ratio deformation / intensity addition
    sigma=(4,4,4),          #  Kernel size
    cost_cst=lamb,         # Regularization parameter
    integration_steps=10,   # Number of integration steps
    n_iter=60,             # Number of optimization steps
    grad_coef=1,            # max Gradient coefficient
    data_term=None,         # Data term (default Ssd)
    sharp=False,            # Sharp integration toggle
    safe_mode = False,      # Safe mode toggle (does not crash when nan values are encountered)
    integration_method='semiLagrangian',  # You should not use Eulerian for real usage
)
mr_meta.plot_cost()

# you can get the deformation grid:
deformation  = mr_meta.mp.get_deformation()

# We provide some visualisation tools :

i3v.Visualize_geodesicOptim(mr_meta,alpha=1)
plt_v = i3v.compare_3D_images_vedo(T,mr_meta.mp.image_stock.cpu())
plt_v.show_deformation_flow(deformation,1,step=3)
plt_v.plotter.show(interactive=True).close()