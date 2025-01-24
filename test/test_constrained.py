"""import src.demeter.metamorphosis as mt
import matplotlib.pyplot as plt
import src.demeter.utils.torchbox as tb
import src.demeter.utils.reproducing_kernels as rk

plot =False

size = (100,100)
device = 'cpu'
source_name,target_name = '23','24'
S = tb.reg_open(source_name,size = size).to(device)   # Small oval with gray dots
T = tb.reg_open(target_name,size = size).to(device)   # Big circle with deformed gray dots
seg = tb.reg_open('21_seg',size=size).to(device)      # rounded triangle


print('T :',T.max(),' S :',S.max())
## Construct the target image
ini_ball,_ = tb.make_ball_at_shape_center(seg,overlap_threshold=.1,verbose=True)
ini_ball = ini_ball.to(device)
T[seg>0] = 0.5                                       # Add the rounded triangle to the target

source = S
target = T
# mask = mr.mp.image_stock

source_name = 'oval_w_round'
target_name = 'round_w_triangle_p_rd'

if plot:
    kw_img = dict(cmap='gray',vmin=0,vmax=1)
    plt.rcParams["figure.figsize"] = (20,20)
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(source[0,0,:,:].cpu().numpy(),**kw_img)
    ax[0,0].set_title('source')
    ax[0,1].imshow(target[0,0,:,:].cpu().numpy(),**kw_img)
    ax[0,1].set_title('target')
    ax[1,0].imshow(tb.imCmp(source,target),vmin=0,vmax=1)
    ax[1,1].imshow(seg[0,0].cpu().numpy(),**kw_img)
    ax[1,1].set_title('segmentation')
    plt.show()

ic.disable()
kernelOperator = rk.GaussianRKHS(sigma=(3, 3), normalized=True)
### MAKE MASKs
mr_mask = mt.lddmm(ini_ball,seg,0,kernelOperator,.0001,
                    integration_steps=5,n_iter=10,grad_coef=1,
                    safe_mode=False,
                  dx_convention='pixel',
              optimizer_method='LBFGS_torch'
              )
if plot:
    mr_mask.plot()
    mr_mask.mp.plot()
    plt.show()
#%%
# mask weighted
mask_w = mr_mask.mp.image_stock.clone()
mask_w = 1 - .5*mask_w

# mask oriented
mask_o = mr_mask.mp.image_stock.clone()
field_o = mr_mask.mp.field_stock.clone()


if plot:
    fig,ax =plt.subplots(2,2,figsize=(20,10))
    a =ax[0,0].imshow(mask_w[0,0].cpu().numpy(),cmap='gray',vmin=0,vmax=1)
    fig.colorbar(a,ax=ax[0,0])
    b = ax[0,1].imshow(mask_w[-1,0].cpu().numpy(),cmap='gray',vmin=0,vmax=1)
    fig.colorbar(b,ax=ax[0,1])
    tb.gridDef_plot_2d(field_o[-1][None],ax=ax[1,0],add_grid=True)
    c = ax[1,1].imshow(mask_o[-1,0].cpu().numpy(),cmap='gray',vmin=0,vmax=1)
    fig.colorbar(c,ax=ax[1,1])
    plt.show()
#%%
#  MAKE CM
ic.enable()
n_iter =5
lamb = 0.001
grad_coef = 1
mask_o = mask_o.to(device)
mask_w = mask_w.to(device)
field_o = field_o.to(device)
# residuals = mr_wm.to_analyse[0].clone().to(device)
mr_cm = mt.constrained_metamorphosis(source,target,0,
                    orienting_mask = mask_o,
                    orienting_field = field_o,
                    residual_mask = mask_w,
                    kernelOperator=kernelOperator,
                    cost_cst=lamb,
                    n_iter=n_iter,
                    grad_coef=grad_coef,
                    safe_mode=False,
                    dx_convention='pixel',
                    # optimizer_method='adadelta'
                                  )
mr_cm.plot()
mr_cm.mp.plot()
plt.show()"""