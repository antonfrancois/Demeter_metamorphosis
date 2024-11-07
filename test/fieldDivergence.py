import argparse
import sys

import __init__

import torch
import matplotlib.pyplot as plt

import demeter.utils.bspline as mbs
from demeter.utils.torchbox import Field_divergence, grid2im




#%%

"""field = dd.field()

print(field.shape)


div = tb.Field_divergence(dd.dx_convention)(field)


div_theorique = dd.theoritical_div()

score =  (div_theorique[1:-1,1:-1] - div[0,0,1:-1,1:-1]).abs()
print("Difference in between the theoritical div and the finite difference:")
print("mean absolute difference:", score.mean())
print("std absolute difference:", score.std())
print("median absolute difference:", score.median())
print("min absolute difference:", score.min())
print("max absolute difference:", score.max())

# _,d_ax = plt.subplots()
fig,ax = plt.subplots(1,3,constrained_layout=True)

div_plot = ax[0].imshow(div[0,0,:,:],origin='lower')
ax[0].quiver(field[0,:,:,0],field[0,:,:,1])
ax[0].set_title('sobel Divergence')
fig.colorbar(div_plot, ax =ax[0])
b= ax[1].imshow(div_theorique,origin='lower')
ax[1].set_title('theoretical Divergence')
fig.colorbar(b, ax=ax[1])
c = ax[2].imshow(score,origin='lower')
ax[2].set_title('absolute difference')
fig.colorbar(c, ax=ax[2])
plt.show()
"""


#%%



#%%
"""import demeter.utils.torchbox as tb
import matplotlib.pyplot as plt
rgi = tb.RandomGaussianImage((100, 150), 1, 'pixel')
img =  rgi.image()
der = rgi.derivative()
# print("a",rgi.a)
# print("b",rgi.b)
# print("c",rgi.c)
# print("IMAGE :",img.shape)
#%%
fig,ax = plt.subplots(3,1,constrained_layout=True)
ax[0].imshow(img[0,0],origin = 'lower')
ax[0].set_title('Image')
ax[1].imshow(der[0,0],origin = 'lower')
ax[1].set_title('derivative_x')
ax[2].imshow(der[0,1],origin = 'lower')
ax[2].set_title('derivative_y')
plt.show()
#%%
fgf  = tb.RandomGaussianField((100,150,2),1,'pixel')
print(fgf)
field = fgf.field()
th_div = fgf.divergence()

sob_div = tb.Field_divergence('pixel')(field)

print("field shape:",field.shape)
print("div shape:",th_div.shape)
print("sobel div shape:",sob_div.shape)

#%%
score = (th_div[0,0,1:-1,1:-1] - sob_div[0,0,1:-1,1:-1]).abs()
print("Difference in between the theoritical div and the finite difference:")
print("mean absolute difference:", score.mean())
print("std absolute difference:", score.std())
print("median absolute difference:", score.median())
print("min absolute difference:", score.min())
print("max absolute difference:", score.max())

fig, ax = plt.subplots(3,2, constrained_layout=True, figsize=(10, 10))

a = ax[0,0].imshow(field[0,:,:,0])
plt.colorbar(a,ax=ax[0,0],fraction=0.046, pad=0.04)
ax[0,0].set_title('field_x')
b = ax[0,1].imshow(field[0,:,:,1])
plt.colorbar(b,ax=ax[0,1],fraction=0.046, pad=0.04)
ax[0,1].set_title('field_y')

ax[1,0].imshow(fgf.rgi_list[0].derivative()[0,0])
ax[1,0].set_title('derivative_x')
ax[1,1].imshow(fgf.rgi_list[0].derivative()[0,0])
ax[1,1].set_title('derivative_y')

ax[2,0].imshow(sob_div[0,0])
ax[2,0].set_title('sobel Divergence')
ax[2,1].imshow(th_div[0,0])
ax[2,1].set_title('theoretical Divergence')
plt.show()
"""


#%%

D,H,W = 50,100,165

import torch
import matplotlib.pyplot as plt
import unittest
import demeter.utils.bspline as mbs
import demeter.utils.torchbox as tb

def plot_test_div_2d(cls):
    print(cls.name)

    fig,ax = plt.subplots(1,4,constrained_layout=True,figsize= (15,4))
    for a in ax.ravel():
        a.set_xlim(cls.x.min(),cls.x.max())
        a.set_ylim(cls.y.min(),cls.y.max())
    im_kw = {
        'origin' : 'lower',
        'aspect' : 'auto',
        'extent' : (cls.x.min(),cls.x.max(),cls.y.min(),cls.y.max())

    }
    step = min(max(H,W), int(max(H,W) /30)+1)
    print("shape, max :",max(cls.div.shape),step)
    div_plot = ax[0].imshow(cls.div[0,0,:,:],**im_kw)
    ax[0].quiver(
        cls.x[::step,::step],cls.y[::step,::step],
        cls.field[0,::step,::step,0],cls.field[0,::step,::step,1])
    ax[0].set_title('sobel Divergence')
    fig.colorbar(div_plot, ax =ax[0],fraction=0.046, pad=0.04)
    b = ax[1].imshow(cls.div_theoretical[0,0],**im_kw)
    ax[1].set_title('theoretical Divergence')
    fig.colorbar(b, ax=ax[1],fraction=0.046, pad=0.04)
    c = ax[2].imshow(cls.score,**im_kw)
    ax[2].set_title('absolute difference')

    fig.colorbar(c, ax=ax[2],fraction=0.046, pad=0.04)

    # ax[3].imshow(torch.ones_like(cls.div_theoretical),origin='lower',cmap='gray')
    affin_y = lambda x : x* (cls.y.max() - cls.y.min()) + cls.y.min()
    ax[3].text(cls.x.min()+.1, affin_y(0.1),  f"mean: {cls.score.mean()}\n")
    ax[3].text(cls.x.min()+.1, affin_y(0.3),  f"std: {cls.score.std()}\n")
    ax[3].text(cls.x.min()+.1, affin_y(0.5),  f"median: {cls.score.median()}\n")
    ax[3].text(cls.x.min()+.1, affin_y(0.7),  f"min: {cls.score.min()}\n")
    ax[3].text(cls.x.min()+.1, affin_y(0.9),  f"max: {cls.score.max()}\n")
    ax[3].text(cls.x.min()+.1,affin_y(1),f"absolute difference")

    fig.suptitle(f"Test divergence {cls.name}")
    plt.show()

class Test_exp_pixel(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'exp_2d_pixel'
        cls.H, cls.W = H, W
        rgf = tb.RandomGaussianField((H, W,2), 4, 'pixel',
                                     a= [
                                         [-2,-1,1,2],
                                        [2,-1,1,-2]
                                     ],
                                     b = [
                                            [.1*H,.3*H,.2*H,.5*H],
                                            [.3*W,.1*W,.1*W,.21*W],
                                     ],)
        cls.field =  rgf.field()
        cls.x = rgf.rgi_list[0].X[0,...,0]
        cls.y = rgf.rgi_list[0].X[0,...,1]


        cls.div_theoretical = rgf.divergence()
        cls.div =tb.Field_divergence('pixel')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1]).abs()
        if args.plot:
            plot_test_div_2d(cls)

    def test_divergence(self):
        eps = 1e-3
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")

import napari

class Test_exp_3d_square(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'exp_3d_square'
        cls.H, cls.W = H, W
        rgf = tb.RandomGaussianField((D,H, W,3), 4, 'square',
                                     a= [
                                         [-2,-1,1,2],
                                        [2,-1,1,-2],
                                        [3,.5,-2.5,-2],
                                     ],
                                     b = [
                                            [.1,.3,.2,.5],
                                            [.3,.1,.1,.21],
                                            [.4,.2,.2,.5],
                                     ],
                                     )
        cls.field =  rgf.field()
        cls.x = rgf.rgi_list[0].X[...,0]
        cls.y = rgf.rgi_list[0].X[...,1]
        cls.z = rgf.rgi_list[0].X[...,2]


        cls.div_theoretical = rgf.divergence()
        cls.div =tb.Field_divergence('square')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1,1:-1]).abs()
        if args.plot:
            nv = napari.Viewer()
            nv.add_image(cls.div_theoretical[0,0],name='theoretical')
            nv.add_image(cls.div[0,0],name='sobel')
            nv.add_image(cls.score,name='difference')
            napari.run()

    def test_divergence(self):
        eps = 1
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")

class Test_exp_2d_square(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'exp_2d_square'
        cls.H, cls.W = H, W
        rgf = tb.RandomGaussianField((H, W,2), 4, 'square',
                                     a= [
                                         [-2,-1,1,2],
                                        [2,-1,1,-2]
                                     ],
                                     b = [
                                            [.1,.3,.2,.5],
                                            [.3,.1,.1,.21]
                                     ],
                                     )
        cls.field =  rgf.field()
        cls.x = rgf.rgi_list[0].X[0,...,0]
        cls.y = rgf.rgi_list[0].X[0,...,1]


        cls.div_theoretical = rgf.divergence()
        cls.div =tb.Field_divergence('square')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1]).abs()
        if args.plot:
            plot_test_div_2d(cls)

    def test_divergence(self):
        eps = 1e-1
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")

class Test_exp_2square(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'exp_2d_2square'
        cls.H, cls.W = H, W
        rgf = tb.RandomGaussianField((H, W,2), 4, '2square',
                                     a= [
                                         [-2,-1,1,2],
                                        [2,-1,1,-2]
                                     ],
                                     b = [
                                            [.1,.3,.2,.5],
                                            [.3,.1,.1,.21]
                                     ])
        cls.field =  rgf.field()
        cls.x = rgf.rgi_list[0].X[0,...,0]
        cls.y = rgf.rgi_list[0].X[0,...,1]


        cls.div_theoretical = rgf.divergence()
        cls.div =tb.Field_divergence('2square')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1]).abs()
        if args.plot:
            plot_test_div_2d(cls)

    def test_divergence(self):
        eps = .5
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")



class TestCosSinPoly_2square(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'cos_sin_poly_2square'
        cls.H, cls.W = H, W
        id_grid = tb.make_regular_grid((H,W),dx_convention='2square')
        xx = id_grid[0,...,0]
        yy = id_grid[0,...,1]
        cls.x, cls.y = xx, yy
        field_x = torch.sin(xx) + xx**2
        field_y = torch.cos(yy) - xx * yy
        cls.field = torch.stack([field_x, field_y], dim=-1)[None]
        cls.div_theoretical = torch.cos(xx) + 2 * xx - torch.sin(yy) - xx
        cls.div_theoretical = cls.div_theoretical[None,None]
        cls.div =tb.Field_divergence('2square')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1]).abs()
        if args.plot:
            plot_test_div_2d(cls)

    def test_divergence(self):
        eps = 1e-3
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")




class Test_square(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.name = 'cos_sin_poly_square'
        cls.H, cls.W = H, W
        id_grid = tb.make_regular_grid((H,W),dx_convention='square')
        xx = id_grid[0,...,0]
        yy = id_grid[0,...,1]
        cls.x, cls.y = xx, yy
        cls.field_x = torch.sin(xx) + xx**2
        cls.field_y = torch.cos(yy) - xx * yy
        cls.field = torch.stack([cls.field_x, cls.field_y], dim=-1)[None]
        cls.div_theoretical = torch.cos(xx) + 2 * xx - torch.sin(yy) - xx
        cls.div_theoretical = cls.div_theoretical[None,None]
        cls.div = tb.Field_divergence('square')(cls.field)
        cls.score =  (cls.div_theoretical[0,0,1:-1,1:-1] - cls.div[0,0,1:-1,1:-1]).abs()
        if args.plot:
            plot_test_div_2d(cls)

    def test_divergence(self):
        eps = 1e-3
        self.assertTrue(torch.all(self.score < eps),
                        f"max difference: {self.score.max()} and tolerance: {eps}")


class TestBSplineField(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.cms = torch.tensor([  # control matrices
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, -1, 0, -1, 0, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, -1, 0, +1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, +1, 0, -1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 0, -1, 0, -1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, +1, 0, -1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 0, -1, 0, -1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, -1, 0, -1, 0, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, +1, 0, -1, 0, +1, 0, +1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ], requires_grad=False, dtype=torch.float)
        self.field_size = (20, 20)
        self.field = mbs.field2D_bspline(self.cms, self.field_size, degree=(3, 3), dim_stack=2).unsqueeze(0)
        self.div = tb.Field_divergence('pixel')(self.field)
        if args.plot:
            self.plot()

    @classmethod
    def plot(cls):
        fig,ax = plt.subplots()

        div_plot = ax.imshow(cls.div[0,0,:,:],origin='lower')
        ax.quiver(cls.field[0,:,:,0],cls.field[0,:,:,1])
        fig.colorbar(div_plot)
        fig.suptitle("Jolie figure")
        plt.show()

    def test_divergence(self):
        self.assertIsNotNone(self.div)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test the function spatialGradient found in `utils.torchbox`"
                    "in a variety of situations"
    )
    parser.add_argument('-p','--plot',
                        dest="plot",
                        action='store_true',
                        default=False ,
                        help="Plot")
    args = parser.parse_args()
    # Utilisez args.plot si nécessaire
    if args.plot:
        print("Plotting is enabled")

    unittest.main(argv=[sys.argv[0]])
