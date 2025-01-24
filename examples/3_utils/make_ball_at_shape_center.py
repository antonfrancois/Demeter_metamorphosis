"""
Il faut retravailler ça
"""

try:
    import sys, os
    # add the parent directory to the path
    base_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    sys.path.insert(0,base_path)
    import __init__

except NameError:
    pass

import vedo.pyplot

# import __init__
import torch
from src.demeter.utils import make_ball_at_shape_center,reg_open, imCmp
import matplotlib.pyplot as plt
import src.demeter.utils.image_3d_visualisation as i3v

#%%
print(f"2D Example :")
img = reg_open('m0t')

ball, info = make_ball_at_shape_center(img,
                                       shape_value=img.max(),
                                       overlap_threshold=.1,
                                       verbose=True)
centre_x,centre_y,r = info

fig,ax = plt.subplots(1,2)
ax[0].imshow(img[0,0],cmap='gray')
ax[0].plot(centre_x,centre_y,'x')
ax[1].imshow(imCmp(img,ball),origin='lower')
plt.show()

#%%
_ = input("Press for 3D example :")
# img = torch.load("im3Dbank/unmooned.pt")
img = torch.load("im3Dbank/segmentation_3D_toyExample.pt")



ball, info = make_ball_at_shape_center(img,
                                       shape_value=img.max(),
                                       overlap_threshold=.1,
                                       # force_r=50,
                                       verbose=True)
centre_x,centre_y,centre_z,r = info
cmp = i3v.compare_3D_images_vedo(ball,img,close=False)
pt = vedo.Point([centre_x,centre_y,centre_z],r=15).c('red').alpha(1)
cmp.plotter.show(pt,at=0)
cmp.plotter.show(pt,at=1)
cmp.plotter.show(interactive=True).close()