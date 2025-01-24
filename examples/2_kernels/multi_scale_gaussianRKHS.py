import numpy as np

import matplotlib.pyplot as plt
import torch
import src.demeter.utils.reproducing_kernels as rk
import src.demeter.utils.torchbox as tb

plt.rcParams['text.usetex'] = False

def see_kernels_filter_2D(sigma_list,ax=None, force_xticks=None):

    mono_gauss = [ rk.GaussianRKHS(s,'constant') for s in sigma_list]
    multi_RKHS = rk.Multi_scale_GaussianRKHS(sigma_list)
    font_size= 25
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))
    kernel_multi = multi_RKHS.kernel
    _,h,w = kernel_multi.shape
    tick_max = w if force_xticks is None else force_xticks
    ax.plot(kernel_multi[0,h//2],label=sigma_list,c='black')
    style = ['--','-.',':']
    for i,s in enumerate(sigma_list):
        kernel_m = mono_gauss[i].kernel
        _,hh,ww = kernel_m.shape
        ax.plot((torch.arange(ww)+((tick_max - ww)/2)),
                mono_gauss[i].kernel[0,hh//2],
                label=s,linestyle=style[i - 3*(i//3)])
    ax.set_ylabel(r'$K_\sigma(x)$',fontsize=font_size)
    ax.set_xlabel(r'$x$',fontsize=font_size)
    # tick_max = w if force_xticks is None else force_xticks
    x_ticks = np.linspace(0,tick_max,5)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='both',
                   #which='minor',
                   labelsize=font_size)
    ax.legend(loc='upper left',
              fontsize=font_size,
              bbox_to_anchor=(.75, 1.),
              # ncol=len(sigma_list)+1
              )

def see_im_convoled_by_kernel(rk,I,ax):
    _,_,H,W = I.shape
    I_g = rk(I)
    ax.imshow(I_g[0,0], **DLT_KW_IMAGE)

def see_mono_kernels(I,sigma_list):
    fig,ax = plt.subplots(1,len(sigma_list))
    for i,s in enumerate(sigma_list):
        grk = rk.GaussianRKHS(s,'constant')
        see_im_convoled_by_kernel(grk,I,ax[i])



if __name__ == '__main__':

    sigma_list = [
        (1,1),
        (5,5),
        (10,10),
        (20,20),
        # (16,16)
    ]

    I = tb.reg_open('01',size = (300,300))

    see_kernels_filter_2D(sigma_list)

    see_mono_kernels(I,sigma_list)

    multirk = rk.Multi_scale_GaussianRKHS(sigma_list)
    plt.figure()
    plt.imshow(multirk(I)[0,0])
    plt.show()


