"""
.. _make_ball_at_shape_center:

A utility function to create a ball at the center of a shape in an image.
Can be useful to initialise a mask for a guided registration task.
"""


try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass


# import __init__
import torch
from demeter.constants import ROOT_DIRECTORY
from demeter.utils.torchbox import make_ball_at_shape_center,reg_open, imCmp
import matplotlib.pyplot as plt
import demeter.utils.image_3d_plotter as i3p

#%%
print(f"2D Example :")
img = reg_open('m0t')

ball, info = make_ball_at_shape_center(img,
                                       overlap_threshold=.1,
                                       verbose=True)
centre_x,centre_y,r = info

fig,ax = plt.subplots(1,2)
ax[0].imshow(img[0,0],cmap='gray')
ax[0].plot(centre_x,centre_y,'x')
ax[1].imshow(imCmp(img,ball),origin='lower')
plt.show()

#%%
# _ = input("Press for 3D example :")
img = torch.load(ROOT_DIRECTORY+"/examples/im3Dbank/hanse_w_ball.pt")


ball, info = make_ball_at_shape_center(img,
                                       # shape_binarization=img == img.max(),
                                       overlap_threshold=.1,
                                       # force_r=50,
                                       verbose=True)
centre_x,centre_y,centre_z,r = info

img_cmp = imCmp(img,ball)
i3p.imshow_3d_slider(img_cmp)
plt.show()