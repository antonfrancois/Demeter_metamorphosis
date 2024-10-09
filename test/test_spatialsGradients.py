import sys

from IPython.core.pylabtools import figsize
from skimage.filters.rank import minimum, maximum
from sympy.physics.quantum.gate import normalized

import __init__
import torch
import matplotlib.pyplot as plt
import unittest
import argparse

import demeter.utils.torchbox as tb

# Test in this file:
# 1. TestLinearX_pixel (line 21):
#      Tests spatial derivatives in the x direction for a linear image in x
#      with pixelic coordinates.
# 2. TestLinearX_1square (line 78):
#      Tests spatial derivatives for an image on the [0,1] square area.
# 3. TestLinearX_2square (line 101):
#      Tests spatial derivatives for an image on the [-1,1] square area. (Pytorch convention)
# 4. TestLinearY_pixel (line 124):
#      Tests spatial derivatives in the y direction for a linear image in y.
# 5. TestCustom2D_pixel (line 147):
#      Tests spatial derivatives for an image defined by a custom
#      polynomial function in 2D.
# 6. TestLinear3D_pixel (line 170):
#      Tests spatial derivatives for a linear image in 3D.
# 7. TestCustom3D_pixel (line 194):
#      Tests spatial derivatives for an image defined by a custom
#      function in 3D.




class TestLinearX_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.H,cls.W = (10,15)
        yy,xx = torch.meshgrid(
            torch.arange(0,cls.H,dtype=torch.long),
            torch.arange(0,cls.W,dtype=torch.long),
            indexing='ij'
        )

        cls.image = xx.clone()
        cls.image = cls.image[None,None].to(torch.float64)

        cls.derivative = tb.spatialGradient(cls.image,dx_convention = "pixel")
        if args.plot:
            cls.plot()

    @classmethod
    def plot(cls):
        fig,ax  = plt.subplots(3,2)
        ax[0,0].imshow(cls.image[0,0],cmap="gray")
        ax[1,0].imshow(cls.derivative[0,0,0],cmap="gray")
        ax[2,0].imshow(cls.derivative[0,0,1],cmap="gray")

        ax[0,1].plot(cls.image[0,0,cls.H//2])
        ax[1,1].plot(cls.derivative[0,0,0,:,cls.W//2])
        ax[2,1].plot(cls.derivative[0,0,1,cls.H//2])
        plt.show()

    def test_derivative_constant_direction_x(self):
        self.assertEqual(
            self.derivative[0,0,0,:,self.W//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_x f(x,y) = x was not constant"
        )

    def test_derivative_constant_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_y f(x,y) = x was not constant"
        )

    def test_value_of_derivative_direction_x(self):
        eps = 1e-7
        self.assertTrue(
            (self.derivative[0,0,0,self.H//2,self.W//2] - 1).abs() < eps,
            "The derivative of D_x f(x,y) = x should be "
            f"{1} but was {self.derivative[0,0,0,self.H//2,self.W//2]}"
        )

    def test_value_of_derivative_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2,self.W//2],
            0,
            "The derivative of D_y f(x,y) = x should be "
            f"{0} but was {self.derivative[0,0,1,self.H//2,self.W//2]}"
        )



class TestLinearX_1square(unittest.TestCase):

    def setUp(self):
        self.H,self.W = (10,15)
        yy,xx = torch.meshgrid(
            torch.linspace(0,1,self.H),
            torch.linspace(0,1,self.W),
            indexing='ij'
        )

        self.image = xx.clone()
        self.image = self.image[None,None].to(torch.float64)


        self.derivative = tb.spatialGradient(self.image,dx_convention = "square")

    def test_derivative_constant_direction_x(self):
        self.assertEqual(
            self.derivative[0,0,0,:,self.W//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_x f(x,y) = x was not constant"
        )

    def test_derivative_constant_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_y f(x,y) = x was not constant"
        )

    def test_value_of_derivative_direction_x(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0,0,0,self.H//2,self.W//2] - 1).abs() < eps,
            "The derivative of D_x f(x,y)= x should be 1 "
            f" but was {self.derivative[0,0,0,self.H//2,self.W//2]}"
        )

    def test_value_of_derivative_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2,self.W//2],
            0,
            "The derivative of D_x f(x,y) = x should be "
            f"{0} but was {self.derivative[0,0,1,self.H//2,self.W//2]}"
        )

class TestLinearX_2square(unittest.TestCase):

    def setUp(self):
        self.H,self.W = (10,15)
        yy,xx = torch.meshgrid(
            torch.linspace(-1,1,self.H),
            torch.linspace(-1,1,self.W),
            indexing='ij'
        )

        self.image = xx.clone()
        self.image = self.image[None,None].to(torch.float64)


        self.derivative = tb.spatialGradient(self.image,dx_convention = "2square")

    def test_derivative_constant_direction_x(self):
        self.assertEqual(
            self.derivative[0,0,0,:,self.W//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_x f(x,y) = x was not constant"
        )

    def test_derivative_constant_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2].unique().shape,
            torch.Size([1]),
            "The derivative of D_y f(x,y) = y was not constant"
        )

    def test_value_of_derivative_direction_x(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0,0,0,self.H//2,self.W//2] - 1).abs() < eps,
            "The derivative of D_x f(x,y) = x should be "
            f"{1} but was {self.derivative[0,0,0,self.H//2,self.W//2]}"
        )

    def test_value_of_derivative_direction_y(self):
        self.assertEqual(
            self.derivative[0,0,1,self.H//2,self.W//2],
            0,
            "The derivative of D_x f(x,y) = x should be "
            f"{0} but was {self.derivative[0,0,1,self.H//2,self.W//2]}"
        )

class TestLinearY_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.H, cls.W = (10, 15)
        yy, xx = torch.meshgrid(
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            indexing='ij'
        )

        cls.image = yy.clone()
        cls.image = cls.image[None, None].to(torch.float64)

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        if args.plot:
            cls.plot()

    @classmethod
    def plot(cls):
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(cls.image[0, 0], cmap="gray")
        ax[1, 0].imshow(cls.derivative[0, 0, 0], cmap="gray")
        ax[2, 0].imshow(cls.derivative[0, 0, 1], cmap="gray")

        ax[0, 1].plot(cls.image[0, 0, cls.H // 2])
        ax[1, 1].plot(cls.derivative[0, 0, 0, :, cls.W // 2])
        ax[2, 1].plot(cls.derivative[0, 0, 1, cls.H // 2])
        plt.show()

    def test_derivative_constant_direction_x(self):
        self.assertEqual(
            self.derivative[0, 0, 0, :, self.W // 2].unique().shape,
            torch.Size([1]),
            "The derivative of D_x f(x,y) = y was not constant"
        )

    def test_derivative_constant_direction_y(self):
        self.assertEqual(
            self.derivative[0, 0, 1, self.H // 2].unique().shape,
            torch.Size([1]),
            "The derivative of D_y f(x,y) = y was not constant"
        )

    def test_value_of_derivative_direction_x(self):
        self.assertEqual(
            self.derivative[0, 0, 0, self.H // 2, self.W // 2],
            0,
            "The derivative of D_x f(x,y) = y should be 0"
        )

    def test_value_of_derivative_direction_y(self):
        eps = 1e-7
        self.assertTrue(
            (self.derivative[0, 0, 1, self.H // 2, self.W // 2] - 1).abs() < eps,
            "The derivative of D_y f(x,y) = y should be "
            f"{1} but was {self.derivative[0, 0, 1, self.H // 2, self.W // 2]}"
        )

class TestCustom2D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.H, cls.W = (10, 15)
        yy, xx = torch.meshgrid(
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            indexing='ij'
        )

        cls.image = yy**2  + 2*xx*yy
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = 2 * yy
        cls.theoretical_derivative_y = 2 * yy + 2 * xx

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        if args.plot:
            cls.plot()

    @classmethod
    def plot(cls):
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(cls.image[0, 0], cmap="gray")
        ax[0, 0].set_title("f(x,y) = yy**2  + 2*xx*yy")
        ax[1, 0].imshow(cls.derivative[0, 0, 0], cmap="gray")
        ax[1, 0].set_title('Spatial gradient x')
        ax[2, 0].imshow(cls.derivative[0, 0, 1], cmap="gray")
        ax[2, 0].set_title('Spatial gradient y')

        ax[0, 1].plot(cls.image[0, 0,:, cls.W // 2])
        ax[1, 1].imshow(cls.theoretical_derivative_x, cmap="gray")
        ax[1, 1].set_title("D_x f(x,y) = 2*yy")
        ax[2, 1].imshow(cls.theoretical_derivative_y, cmap="gray")
        ax[2, 1].set_title("D_y f(x,y) = 2*yy + 2*xx")
        plt.show()

    def test_values_of_derivative_direction_x(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 0,1:-1,1:-1] - self.theoretical_derivative_x[1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_x f(x,y) = yy**2  + 2*xx*yy  should be "
            f"{self.theoretical_derivative_x} but was {self.derivative[0, 0, 0]}"
        )

    def test_values_of_derivative_direction_y(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 1,1:-1,1:-1] - self.theoretical_derivative_y[1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y) = yy**2  + 2*xx*yy should be "
            f"{self.theoretical_derivative_y} but was {self.derivative[0, 0, 1]}"
        )

class TestLinear3D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.D, cls.H, cls.W = (5, 10, 20)
        xx,yy,zz = torch.meshgrid(
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            torch.arange(0, cls.D, dtype=torch.long),
            indexing='ij'
        )

        cls.image = xx + yy + zz
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = torch.ones_like(xx)
        cls.theoretical_derivative_y = torch.ones_like(yy)
        cls.theoretical_derivative_z = torch.ones_like(zz)

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        # if args.plot:
        #     cls.plot()

    # @classmethod
    # def plot(cls):
    #     fig, ax = plt.subplots(3, 3)
    #     ax[0, 0].imshow(cls.image[0, 0, cls.D // 2], cmap="gray")
    #     ax[0, 0].set_title("f(x,y,z) = zz**2 + yy**2 + 2*xx*yy")
    #     ax[1, 0].imshow(cls.derivative[0, 0, 0, cls.D // 2], cmap="gray")
    #     ax[1, 0].set_title('Spatial gradient x')
    #     ax[2, 0].imshow(cls.derivative[0, 0, 1, cls.D // 2], cmap="gray")
    #     ax[2, 0].set_title('Spatial gradient y')
    #     ax[2, 1].imshow(cls.derivative[0, 0, 2, cls.D // 2], cmap="gray")
    #     ax[2, 1].set_title('Spatial gradient z')
    #
    #     ax[0, 1].plot(cls.image[0, 0, cls.D // 2, :, cls.W // 2])
    #     ax[1, 1].imshow(cls.theoretical_derivative_x[cls.D // 2], cmap="gray")
    #     ax[1, 1].set_title("D_x f(x,y,z) = 2*yy")
    #     ax[2, 1].imshow(cls.theoretical_derivative_y[cls.D // 2], cmap="gray")
    #     ax[2, 1].set_title("D_y f(x,y,z) = 2*yy + 2*xx")
    #     ax[2, 2].imshow(cls.theoretical_derivative_z[cls.D // 2], cmap="gray")
    #     ax[2, 2].set_title("D_z f(x,y,z) = 2*zz")
    #     plt.show()

    def test_values_of_derivative_direction_x(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_x f(x,y,z) = xx + yy + zz should be "
            f"{self.theoretical_derivative_x} but was {self.derivative[0, 0, 0]}"
        )

    def test_values_of_derivative_direction_y(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y,z) = xx + yy + zz should be "
            f"{self.theoretical_derivative_y} but was {self.derivative[0, 0, 1]}"
        )

    def test_values_of_derivative_direction_z(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_z f(x,y,z) = xx + yy + zz should be "
            f"{self.theoretical_derivative_z} but was {self.derivative[0, 0, 2]}"
        )

class TestCustom3D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.H,cls.W,cls.D = (100, 300, 150)
        zz,yy,xx = torch.meshgrid(
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            torch.arange(0, cls.D, dtype=torch.long),
            indexing='ij'
        )

        c1,c2,c3 = .002,.003,.001
        cls.image = torch.sin((c1 * (xx- cls.H//2)**2 + c2 * (yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = c1 * 2 * (xx - cls.H//2) * torch.cos((c1 * (xx- cls.H//2)**2 + c2 *(yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))
        cls.theoretical_derivative_y = c2 * 2 * (yy - cls.W//2) * torch.cos((c1 * (xx- cls.H//2)**2 + c2 *(yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))
        cls.theoretical_derivative_z = c3 * 2 * (zz - cls.D//2) * torch.cos((c1 * (xx- cls.H//2)**2 + c2 *(yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))


        # cls.image = torch.sin(((xx - cls.H//2)**2 + (yy - cls.W//2)**2) + (zz - cls.D//2)**2)
        # cls.image = cls.image[None, None].to(torch.float64)
        #
        # cls.theoretical_derivative_x = -torch.cos(((xx - cls.H//2)**2 + (yy - cls.W//2)**2) + (zz - cls.D//2)**2) * 2 * (xx - cls.H//2)
        # cls.theoretical_derivative_y = -torch.cos(((xx - cls.H//2)**2 + (yy - cls.W//2)**2) + (zz - cls.D//2)**2) * 2 * (yy - cls.W//2)
        # cls.theoretical_derivative_z = -torch.cos(((xx - cls.H//2)**2 + (yy - cls.W//2)**2) + (zz - cls.D//2)**2) * 2 * (zz - cls.D//2)


        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        if args.plot:
            cls.plot()

    @classmethod
    def plot(cls):
        fix,ax = plt.subplots(3,4, constrained_layout=True)

        im_kw = dict(cmap="gray",vmin=cls.image.min(),vmax=cls.image.max())
        ax[0,0].imshow(cls.image[0,0,cls.H//2],**im_kw)
        ax[0,0].set_title("f(x,y,z) = xx + yy + zz")
        ax[1,0].imshow(cls.image[0,0,:,cls.W//2],**im_kw)
        ax[2,0].imshow(cls.image[0,0,:,:,cls.D//2],**im_kw)


        min_p = min(cls.derivative.min(),
                        float(cls.theoretical_derivative_x.min()),
                        float(cls.theoretical_derivative_y.min()),
                        float(cls.theoretical_derivative_z.min()))
        max_p = max(cls.derivative.max(),
                        float(cls.theoretical_derivative_x.max()),
                        float(cls.theoretical_derivative_y.max()),
                        float(cls.theoretical_derivative_z.max()))
        der_kw = dict(cmap="gray",vmin=min_p,vmax=max_p)
        ax[0,1].set_title("D_x f(x,y,z)")
        ax[0,1].imshow(cls.derivative[0,0,0,cls.H//2],**der_kw)
        ax[1,1].imshow(cls.derivative[0,0,0,:,cls.W//2],**der_kw)
        ax[2,1].imshow(cls.derivative[0,0,0,:,:,cls.D//2],**der_kw)

        ax[0,2].set_title("D_y f(x,y,z)")
        ax[0,2].imshow(cls.derivative[0,0,1,cls.H//2],**der_kw)
        ax[1,2].imshow(cls.derivative[0,0,1,:,cls.W//2],**der_kw)
        ax[2,2].imshow(cls.derivative[0,0,1,:,:,cls.D//2],**der_kw)

        ax[0,3].set_title("D_z f(x,y,z)")
        ax[0,3].imshow(cls.derivative[0,0,2,cls.H//2],**der_kw)
        ax[1,3].imshow(cls.derivative[0,0,2,:,cls.W//2],**der_kw)
        ax[2,3].imshow(cls.derivative[0,0,2,:,:,cls.D//2],**der_kw)


        fix,ax = plt.subplots(3,4, constrained_layout=True)

        im_kw = dict(cmap="gray",vmin=cls.image.min(),vmax=cls.image.max())
        ax[0,0].imshow(cls.image[0,0,cls.H//2],**im_kw)
        ax[0,0].set_title("f(x,y,z) = xx + yy + zz")
        ax[1,0].imshow(cls.image[0,0,:,cls.W//2],**im_kw)
        ax[2,0].imshow(cls.image[0,0,:,:,cls.D//2],**im_kw)

        # der_kw = dict(cmap="gray",vmin=cls.theoretical_derivative_z.min(),vmax=cls.theoretical_derivative_z.max())
        ax[0,1].set_title("D_x f(x,y,z) theorique")
        ax[0,1].imshow(cls.theoretical_derivative_x[cls.H//2],**der_kw)
        ax[1,1].imshow(cls.theoretical_derivative_x[:,cls.W//2],**der_kw)
        ax[2,1].imshow(cls.theoretical_derivative_x[:,:,cls.D//2],**der_kw)

        ax[0,2].set_title("D_y f(x,y,z) theorique")
        ax[0,2].imshow(cls.theoretical_derivative_y[cls.H//2],**der_kw)
        ax[1,2].imshow(cls.theoretical_derivative_y[:,cls.W//2],**der_kw)
        ax[2,2].imshow(cls.theoretical_derivative_y[:,:,cls.D//2],**der_kw)

        ax[0,3].set_title("D_z f(x,y,z) theorique")
        ax[0,3].imshow(cls.theoretical_derivative_z[cls.H//2],**der_kw)
        ax[1,3].imshow(cls.theoretical_derivative_z[:,cls.W//2],**der_kw)
        ax[2,3].imshow(cls.theoretical_derivative_z[:,:,cls.D//2],**der_kw)


        fig,ax = plt.subplots(3,3,constrained_layout=True,figsize = (10,10))
        ax[0,0].plot(cls.derivative[0,0,0,:,cls.W//2,cls.D//2],label="spatialGradient")
        ax[0,0].plot(cls.theoretical_derivative_x[:,cls.W//2,cls.D//2],'--',label="theoretical")
        ax[0,0].set_title("D_x f(x,y,z)")
        ax[0,0].set_ylabel("X")

        ax[0,1].plot(cls.derivative[0,0,1,:,cls.W//2,cls.D//2],label="spatialGradient")
        ax[0,1].plot(cls.theoretical_derivative_y[:,cls.W//2,cls.D//2],'--',label="theoretical")
        ax[0,1].set_title("D_y f(x,y,z)")
        ax[0,1].set_ylabel("X")


        ax[0,2].plot(cls.derivative[0,0,2,:,cls.W//2,cls.D//2],label="spatialGradient")
        ax[0,2].plot(cls.theoretical_derivative_z[:,cls.W//2,cls.D//2],'--',label="theoretical")
        ax[0,2].set_title("D_z f(x,y,z)")
        ax[0,1].set_ylabel("X")

        ax[1,0].plot(cls.derivative[0,0,0,cls.H//2,:,cls.D//2],label="spatialGradient")
        ax[1,0].plot(cls.theoretical_derivative_x[cls.H//2,:,cls.D//2],'--',label="theoretical")
        ax[1,0].set_title("D_x f(x,y,z)")
        ax[1,0].set_ylabel("Y")

        ax[1,1].plot(cls.derivative[0,0,1,cls.H//2,:,cls.D//2],label="spatialGradient")
        ax[1,1].plot(cls.theoretical_derivative_y[cls.H//2,:,cls.D//2],'--',label="theoretical")
        ax[1,1].set_title("D_y f(x,y,z)")
        ax[1,1].set_ylabel("Y")


        ax[1,2].plot(cls.derivative[0,0,2,cls.H//2,:,cls.D//2],label="spatialGradient")
        ax[1,2].plot(cls.theoretical_derivative_z[cls.H//2,:,cls.D//2],'--',label="theoretical")
        ax[1,2].set_title("D_z f(x,y,z)")
        ax[1,1].set_ylabel("Y")


        ax[2,0].plot(cls.derivative[0,0,0,cls.H//2,cls.W//2],label="spatialGradient")
        ax[2,0].plot(cls.theoretical_derivative_x[cls.H//2,cls.W//2],'--',label="theoretical")
        ax[2,0].set_title("D_x f(x,y,z)")
        ax[2,0].set_ylabel("Z")

        ax[2,1].plot(cls.derivative[0,0,1,cls.H//2,cls.W//2],label="spatialGradient")
        ax[2,1].plot(cls.theoretical_derivative_y[cls.H//2,cls.W//2],'--',label="theoretical")
        ax[2,1].set_title("D_y f(x,y,z)")
        ax[2,1].set_ylabel("Z")

        # Last
        ax[2,2].plot(cls.derivative[0,0,2,cls.H//2,cls.W//2],label="spatialGradient")
        ax[2,2].plot(cls.theoretical_derivative_z[cls.H//2,cls.W//2],'--',label="theoretical")
        ax[2,2].set_title("D_z f(x,y,z)")
        ax[2,1].set_ylabel("Z")

        plt.legend()
        plt.show()

    def test_values_of_derivative_direction_x(self):
        eps = 3e-1
        self.assertTrue(
            (self.derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y,z) = torch.sin((c1 * (xx- cls.H//2)**2 + c2 * (yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))\n "
            f"max abs diffence is {(self.derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs().max()}"
            f"tolereance is {eps}"
        )

    def test_values_of_derivative_direction_y(self):
        eps = 3e-1
        self.assertTrue(
            (self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y,z) = torch.sin((c1 * (xx- cls.H//2)**2 + c2 * (yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))\n "
            f"max abs diffence is {(self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max()}"
            f"tolereance is {eps}"
        )

    def test_values_of_derivative_direction_z(self):
        eps = 3e-1
        self.assertTrue(
            (self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_z f(x,y,z) = torch.sin((c1 * (xx- cls.H//2)**2 + c2 * (yy - cls.W//2)**2 + c3 * (zz - cls.D//2)**2))\n "
            f"max abs diffence is {(self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max()} tolereance is {eps}"
        )

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

"""
 

import torch
import matplotlib.pyplot as plt
import demeter.utils.torchbox as tb

H,W,D = (100, 300, 150)
zz,yy,xx = torch.meshgrid(
    torch.arange(0, H, dtype=torch.long),
    torch.arange(0, W, dtype=torch.long),
    torch.arange(0, D, dtype=torch.long),
    indexing='ij'
)

# self.image = xx + yy + zz
# self.image = image[None, None].to(torch.float64)
#
# theoretical_derivative_x = xx
# theoretical_derivative_y = torch.zeros_like(theoretical_derivative_x)
# theoretical_derivative_z = torch.zeros_like(theoretical_derivative_x)

# image = xx**2 * zz**3 - xx * yy**2 + torch.sin(zz) + 2*xx*yy
c1,c2,c3 = .005,.003,.001
image = torch.sin((c1 * (xx- H//2)**2 + c2 * (yy - W//2)**2 + c3 * (zz - D//2)**2))
image = image[None, None].to(torch.float64)

theoretical_derivative_x = c1 * 2 * (xx - H//2) * torch.cos((c1 * (xx- H//2)**2 + c2 *(yy - W//2)**2 + c3 * (zz - D//2)**2))
theoretical_derivative_y = c2 * 2 * (yy - W//2) * torch.cos((c1 * (xx- H//2)**2 + c2 *(yy - W//2)**2 + c3 * (zz - D//2)**2))
theoretical_derivative_z = c3 * 2 * (zz - D//2) * torch.cos((c1 * (xx- H//2)**2 + c2 *(yy - W//2)**2 + c3 * (zz - D//2)**2))


derivative = tb.spatialGradient(image, dx_convention="pixel")
# derivative[0,0,0] = derivative[0,0,0] /
print("diff : \n",
(derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max()
)
#%%
fix,ax = plt.subplots(3,4, constrained_layout=True)

im_kw = dict(cmap="gray",vmin=image.min(),vmax=image.max())
ax[0,0].imshow(image[0,0,H//2],**im_kw)
ax[0,0].set_title("f(x,y,z) = xx + yy + zz")
ax[1,0].imshow(image[0,0,:,W//2],**im_kw)
ax[2,0].imshow(image[0,0,:,:,D//2],**im_kw)

der_kw = dict(cmap="gray",vmin=derivative.min(),vmax=derivative.max())
ax[0,1].set_title("D_x f(x,y,z)")
ax[0,1].imshow(derivative[0,0,0,H//2],**der_kw)
ax[1,1].imshow(derivative[0,0,0,:,W//2],**der_kw)
ax[2,1].imshow(derivative[0,0,0,:,:,D//2],**der_kw)

ax[0,2].set_title("D_y f(x,y,z)")
ax[0,2].imshow(derivative[0,0,1,H//2],**der_kw)
ax[1,2].imshow(derivative[0,0,1,:,W//2],**der_kw)
ax[2,2].imshow(derivative[0,0,1,:,:,D//2],**der_kw)

ax[0,3].set_title("D_z f(x,y,z)")
ax[0,3].imshow(derivative[0,0,2,H//2],**der_kw)
ax[1,3].imshow(derivative[0,0,2,:,W//2],**der_kw)
ax[2,3].imshow(derivative[0,0,2,:,:,D//2],**der_kw)
plt.show()

fix,ax = plt.subplots(3,4, constrained_layout=True)

im_kw = dict(cmap="gray",vmin=image.min(),vmax=image.max())
ax[0,0].imshow(image[0,0,H//2],**im_kw)
ax[0,0].set_title("f(x,y,z) = xx + yy + zz")
ax[1,0].imshow(image[0,0,:,W//2],**im_kw)
ax[2,0].imshow(image[0,0,:,:,D//2],**im_kw)

der_kw = dict(cmap="gray",vmin=derivative.min(),vmax=derivative.max())
ax[0,1].set_title("D_x f(x,y,z) theorique")
ax[0,1].imshow(theoretical_derivative_x[H//2],**der_kw)
ax[1,1].imshow(theoretical_derivative_x[:,W//2],**der_kw)
ax[2,1].imshow(theoretical_derivative_x[:,:,D//2],**der_kw)

ax[0,2].set_title("D_y f(x,y,z) theorique")
ax[0,2].imshow(theoretical_derivative_y[H//2],**der_kw)
ax[1,2].imshow(theoretical_derivative_y[:,W//2],**der_kw)
ax[2,2].imshow(theoretical_derivative_y[:,:,D//2],**der_kw)

ax[0,3].set_title("D_z f(x,y,z) theorique")
ax[0,3].imshow(theoretical_derivative_z[H//2],**der_kw)
ax[1,3].imshow(theoretical_derivative_z[:,W//2],**der_kw)
ax[2,3].imshow(theoretical_derivative_z[:,:,D//2],**der_kw)

plt.show()
#%%
fig,ax = plt.subplots(3,3,constrained_layout=True,figsize = (10,10))
ax[0,0].plot(derivative[0,0,0,:,W//2,D//2],label="spatialGradient")
ax[0,0].plot(theoretical_derivative_x[:,W//2,D//2],'--',label="theoretical")
ax[0,0].set_title("D_x f(x,y,z)")
ax[0,0].set_ylabel("X")

ax[0,1].plot(derivative[0,0,1,:,W//2,D//2],label="spatialGradient")
ax[0,1].plot(theoretical_derivative_y[:,W//2,D//2],'--',label="theoretical")
ax[0,1].set_title("D_y f(x,y,z)")
ax[0,1].set_ylabel("X")


ax[0,2].plot(derivative[0,0,2,:,W//2,D//2],label="spatialGradient")
ax[0,2].plot(theoretical_derivative_z[:,W//2,D//2],'--',label="theoretical")
ax[0,2].set_title("D_z f(x,y,z)")
ax[0,1].set_ylabel("X")

ax[1,0].plot(derivative[0,0,0,H//2,:,D//2],label="spatialGradient")
ax[1,0].plot(theoretical_derivative_x[H//2,:,D//2],'--',label="theoretical")
ax[1,0].set_title("D_x f(x,y,z)")
ax[1,0].set_ylabel("Y")

ax[1,1].plot(derivative[0,0,1,H//2,:,D//2],label="spatialGradient")
ax[1,1].plot(theoretical_derivative_y[H//2,:,D//2],'--',label="theoretical")
ax[1,1].set_title("D_y f(x,y,z)")
ax[1,1].set_ylabel("Y")


ax[1,2].plot(derivative[0,0,2,H//2,:,D//2],label="spatialGradient")
ax[1,2].plot(theoretical_derivative_z[H//2,:,D//2],'--',label="theoretical")
ax[1,2].set_title("D_z f(x,y,z)")
ax[1,1].set_ylabel("Y")




ax[2,0].plot(derivative[0,0,0,H//2,W//2],label="spatialGradient")
ax[2,0].plot(theoretical_derivative_x[H//2,W//2],'--',label="theoretical")
ax[2,0].set_title("D_x f(x,y,z)")
ax[2,0].set_ylabel("Z")

ax[2,1].plot(derivative[0,0,1,H//2,W//2],label="spatialGradient")
ax[2,1].plot(theoretical_derivative_y[H//2,W//2],'--',label="theoretical")
ax[2,1].set_title("D_y f(x,y,z)")
ax[2,1].set_ylabel("Z")


ax[2,2].plot(derivative[0,0,2,H//2,W//2],label="spatialGradient")
ax[2,2].plot(theoretical_derivative_z[H//2,W//2],'--',label="theoretical")
ax[2,2].set_title("D_z f(x,y,z)")
ax[2,1].set_ylabel("Z")



plt.legend()
plt.show()
"""