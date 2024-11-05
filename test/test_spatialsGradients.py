import sys

import __init__
import torch
import matplotlib.pyplot as plt
import unittest
import argparse
import napari


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
        fig.suptitle(r"spatial convention on pixelic $[0,H-1]\times [0,W-1]$")
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
        fig.suptitle(r"spatial convention on pixelic $[0,H-1]\times [0,W-1]$")

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
        fig.suptitle(r"spatial convention on pixels $[0,H-1]\times [0,W-1]$")

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


class TestCustom2D_1square(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.H, cls.W = (150, 198)
        yy, xx = torch.meshgrid(
            torch.linspace(0,1, cls.H, dtype=torch.float),
            torch.linspace(0,1, cls.W, dtype=torch.float),
            indexing='ij'
        )

        a,b = 8.7, 5.7
        cls.image = torch.cos(a * xx) + torch.sin(b * xx) * torch.cos (a * yy)
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = -a * torch.sin(a * xx) + b * torch.cos(b * xx) * torch.cos (a * yy)
        cls.theoretical_derivative_y = torch.sin(b * xx) * -a * torch.sin(a * yy)

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="square")
        cls.diff_x  = (cls.theoretical_derivative_x[1:-1,1:-1] - cls.derivative[0, 0, 0,1:-1,1:-1]).abs()
        cls.diff_y  = (cls.theoretical_derivative_y[1:-1,1:-1] - cls.derivative[0, 0, 1,1:-1,1:-1]).abs()
        if args.plot:
            print(f"TestCustom2D_1square, the difference between theoretical and finite difference derivative :")
            print(f"\tdiff_x : max { cls.diff_x.max()};\n\t min {cls.diff_x.min()};\n\t mean {cls.diff_x.mean()};\n\t sum {cls.diff_x.sum()}")
            print(f"\tdiff_y : max { cls.diff_y.max()};\n\t min {cls.diff_y.min()};\n\t mean {cls.diff_y.mean()};\n\t sum {cls.diff_y.sum()}")
            cls.plot()


    @classmethod
    def plot(cls):
        fig, ax = plt.subplots(3, 3, constrained_layout=True)
        ax[0, 0].imshow(cls.image[0, 0], cmap="gray")
        ax[0, 0].set_title("f(x,y) = cos(a * xx) + sin(b * xx) * cos (a * yy)")
        ax[1, 0].imshow(cls.derivative[0, 0, 0], cmap="gray")
        ax[1, 0].set_title('Spatial gradient x')
        ax[2, 0].imshow(cls.derivative[0, 0, 1], cmap="gray")
        ax[2, 0].set_title('Spatial gradient y')

        ax[0, 1].plot(cls.image[0, 0,:, cls.W // 2])
        ax[1, 1].imshow(cls.theoretical_derivative_x, cmap="gray")
        ax[1, 1].set_title("D_x f(x,y) = -a * sin(a * xx) + b * cos(b * xx) * cos (a * yy)")
        ax[2, 1].imshow(cls.theoretical_derivative_y, cmap="gray")
        ax[2, 1].set_title("D_y f(x,y) = sin(b * xx) * -a * sin(a * yy)")
        fig.suptitle("dx convention on the square $[0,1]^2$")

        a = ax[1, 2].imshow(cls.diff_x, cmap="RdYlGn_r")
        ax[1, 2].set_title("difference D_x")
        fig.colorbar(a, ax=ax[1, 2])

        b = ax[2, 2].imshow(cls.diff_y, cmap="RdYlGn_r")
        ax[2, 2].set_title("difference D_y")
        fig.colorbar(b, ax=ax[2, 2])
        fig.suptitle("spatial convention on the square $[0,1]^2$")
        plt.show()


    def test_values_of_derivative_direction_x_mean(self):
        eps = 5e-3
        self.assertTrue(
            self.diff_x.mean() < eps,
            "The mean difference between theoretical and finite difference "
            "derivative of D_x f(x,y) should be <"
            f"{eps} but was {self.diff_x.mean()}"
        )

    def test_values_of_derivative_direction_x_max(self):
        eps = 1e-2
        self.assertTrue(
            self.diff_x.max() < eps,
            "The max difference between theoretical and finite difference "
            "derivative of D_x f(x,y) should be <"
            f"{eps} but was {self.diff_x.max()}"
        )

    def test_values_of_derivative_direction_y_mean(self):
        eps = 5e-3
        self.assertTrue(
            self.diff_y.mean() < eps,
            "The mean difference between theoretical and finite difference "
            "derivative of D_y f(x,y) should be <"
            f"{eps} but was {self.diff_y.mean()}"
        )

    def test_values_of_derivative_direction_y_mean(self):
        eps = 1e-2
        self.assertTrue(
            self.diff_y.mean() < eps,
            "The max difference between theoretical and finite difference "
            "derivative of D_y f(x,y) should be <"
            f"{eps} but was {self.diff_y.max()}"
        )

class TestLinear3D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("TestLinear3D_pixel setup")
        cls.D, cls.H, cls.W = (5, 10, 20)
        zz,yy,xx = torch.meshgrid(
            torch.arange(0, cls.D, dtype=torch.long),
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            indexing='ij'
        )

        cls.image = xx + 2 * yy + 3 *  zz
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = torch.ones_like(xx)
        cls.theoretical_derivative_y = torch.ones_like(yy) * 2
        cls.theoretical_derivative_z = torch.ones_like(zz) * 3

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        # print(f"Values x,y,z of the central pixel in derivative: "
        #       f"{cls.derivative[0, 0, :, cls.D // 2, cls.H // 2, cls.W // 2]}")
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
            f"{self.theoretical_derivative_x[self.D//2,self.H//2,self.W//2]}"
            f" but was {self.derivative[0, 0, 0,self.D//2,self.H//2,self.W//2]}"
        )

    def test_values_of_derivative_direction_y(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y,z) = xx + yy + zz should be "
            f"{self.theoretical_derivative_y[self.D//2,self.H//2,self.W//2]}"
            f" but was {self.derivative[0, 0, 1,self.D//2,self.H//2,self.W//2]}"
        )

    def test_values_of_derivative_direction_z(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_z f(x,y,z) = xx + yy + zz should be "
            f"{self.theoretical_derivative_z[self.D//2,self.H//2,self.W//2]}"
            f" but was {self.derivative[0, 0, 2,self.D//2,self.H//2,self.W//2]}"
        )

"""class TestCustom3D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.H,cls.W,cls.D = (100, 300, 150)
        zz,yy,xx = torch.meshgrid(
            torch.arange(0, cls.D, dtype=torch.long),
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
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
        fig_1,ax = plt.subplots(3,4, constrained_layout=True)

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
        fig_1.suptitle(r"spatial convention on 3D voxels : finite difference")



        fig_2,ax = plt.subplots(3,4, constrained_layout=True)

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
        fig_2.suptitle(r"spatial convention on 3D voxels : theoretical")

        fig_3,ax = plt.subplots(3,3,constrained_layout=True,figsize = (10,10))
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
        fig_3.suptitle(r"spatial convention on 3D voxels : finite difference vs theoretical")

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

class TestCustom3D_1square(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("TestCustom3D_1square setup")
        cls.D, cls.H, cls.W = (200, 600, 300)
        zz, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, cls.D, dtype=torch.float),
            torch.linspace(0, 1, cls.H, dtype=torch.float),
            torch.linspace(0, 1, cls.W, dtype=torch.float),
            indexing='ij'
        )

        a1, a2 = 3, 3
        b1 = .7
        c1, c2 = 1.5, 3

        cls.image = (a1 * torch.exp(-((a2*(xx - 0.5))**2 + (a2*(yy - 0.5))**2 + (a2*(zz - 0.5))**2))
                     + b1 * (xx + yy + zz)
                     + c1 * torch.sin(c2 * torch.pi * xx) * torch.cos(c2 * torch.pi * yy))
        cls.image = cls.image[None, None].to(torch.float64)

        exp_term = a1 * torch.exp(-((a2*(xx - 0.5))**2 + (a2*(yy - 0.5))**2 + (a2*(zz - 0.5))**2))
        sin_term_x = c1 * c2 * torch.pi * torch.cos(c2 * torch.pi * xx) * torch.cos(c2 * torch.pi * yy)
        sin_term_y = -c1 * c2 * torch.pi * torch.sin(c2 * torch.pi * xx) * torch.sin(c2 * torch.pi * yy)

        cls.theoretical_derivative_x = exp_term * (-2 * a2**2 * (xx - 0.5)) + b1 + sin_term_x
        cls.theoretical_derivative_y = exp_term * (-2 * a2**2 * (yy - 0.5)) + b1 + sin_term_y
        cls.theoretical_derivative_z = exp_term * (-2 * a2**2 * (zz - 0.5)) + b1

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="square")
        if args.plot:
            cls.plot()

    def test_derivative_x(self):
        eps = 1e-2
        diff_x = (self.derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs()
        self.assertTrue(torch.all(diff_x < eps),
                        f"max abs differance is {diff_x.max()} tolerance is {eps}")

    def test_derivative_y(self):
        eps = 1e-2
        diff_y = (self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs()
        self.assertTrue(torch.all(diff_y < eps),
                        f"max abs differance is {diff_y.max()} tolerance is {eps}")

    def test_derivative_z(self):
        eps = 1e-2
        diff_z = (self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs()
        self.assertTrue(torch.all(diff_z < eps),
                        f"max abs differance is {diff_z.max()} tolerance is {eps}")

    @classmethod
    def plot(cls):
        fig, ax = plt.subplots(1, 3, constrained_layout=True)
        ax[0].plot(cls.image[0, 0, cls.D//2, cls.H//2, :], label="image")
        ax[0].plot(cls.theoretical_derivative_x[cls.D//2, cls.H//2, :], label="theoretical")
        ax[0].plot(cls.derivative[0, 0, 0, cls.D//2, cls.H//2, :], "--", label="spatialGradient")

        ax[1].plot(cls.image[0, 0, :, cls.H//2, cls.W//2], label="image")
        ax[1].plot(cls.theoretical_derivative_y[:, cls.H//2, cls.W//2], label="theoretical")
        ax[1].plot(cls.derivative[0, 0, 1, :, cls.H//2, cls.W//2], "--", label="spatialGradient")

        ax[2].plot(cls.image[0, 0, :, cls.H//2, cls.W//2], label="image")
        ax[2].plot(cls.theoretical_derivative_z[:, cls.H//2, cls.W//2], label="theoretical")
        ax[2].plot(cls.derivative[0, 0, 2, :, cls.H//2, cls.W//2], "--", label="spatialGradient")
        plt.legend()
        plt.show()
"""

class TestExp2D_pixel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n TestExp2D_pixel")
        rgi = tb.RandomGaussianImage((300, 400), 2, 'pixel',
                                     a=[-1, 1],
                                     b=[15, 25],
                                     c=[[.3*400 , .3*300], [.7*400, .7*300]])
        cls.image =  rgi.image()
        cls.theoretical_derivative = rgi.derivative()
        print(f"image shape : {cls.image.shape}")
        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        print(f"theoretical_derivative shape : {cls.theoretical_derivative.shape}")
        print(f"derivative shape : {cls.derivative.shape}")
        cls.score = (cls.derivative[...,1:-1,1:-1] - cls.theoretical_derivative[...,1:-1,1:-1]).abs()
        if args.plot:
            print(f"TestExp2D_pixel, the difference between theoretical and finite difference derivative :")
            print(f"\tscore : max { cls.score.max()};\n\t min {cls.score.min()};\n\t mean {cls.score.mean()};\n\t")
            cls.plot()

    @classmethod
    def plot(cls):
        fig,ax = plt.subplots(3,2,constrained_layout=True)
        ax[0,0].imshow(cls.image[0,0], origin="lower")
        ax[0,0].set_title('Image')
        a = ax[1,0].imshow(cls.theoretical_derivative[0,0], origin="lower")
        plt.colorbar(a,ax=ax[1,0],fraction=0.046, pad=0.04)
        ax[1,0].set_title('theoretical derivative_x')
        b = ax[2,0].imshow(cls.theoretical_derivative[0,1], origin="lower")
        plt.colorbar(b,ax=ax[2,0],fraction=0.046, pad=0.04)
        ax[2,0].set_title('theoretical derivative_y')

        c = ax[1,1].imshow(cls.derivative[0,0,0], origin="lower")
        plt.colorbar(c,ax=ax[1,1],fraction=0.046, pad=0.04)
        ax[1,1].set_title('sobel derivative_x')
        d = ax[2,1].imshow(cls.derivative[0,0,1], origin="lower")
        plt.colorbar(d,ax=ax[2,1],fraction=0.046, pad=0.04)
        ax[2,1].set_title('sobel derivative_y')
        plt.show()

    def test_values_of_derivative(self):
        eps = 1e-2
        self.assertTrue(
            self.score.max() < eps,
            "The max difference between theoretical and finite difference "
            "derivative of D_x f(x,y) should be <"
            f"{eps} but was {self.score.max()}"
        )

class TestExp2D_square(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n TestExp2D_square")
        rgi = tb.RandomGaussianImage((300, 400), 2, 'square',
                                     a =[-1,1],
                                     b = [.1,.1],
                                     c = [[.5,.3],[.5,.7]])
        cls.image =  rgi.image()
        cls.theoretical_derivative = rgi.derivative()
        print(f"image shape : {cls.image.shape}")
        cls.derivative = tb.spatialGradient(cls.image, dx_convention="square")
        print(f"theoretical_derivative shape : {cls.theoretical_derivative.shape}")
        print(f"derivative shape : {cls.derivative.shape}")
        cls.score = (cls.derivative[...,1:-1,1:-1] - cls.theoretical_derivative[...,1:-1,1:-1]).abs()
        if args.plot:
            print(f"TestExp2D_square, the difference between theoretical and finite difference derivative :")
            print(f"\tscore : max { cls.score.max()};\n\t min {cls.score.min()};\n\t mean {cls.score.mean()};\n\t")
            cls.plot()

    @classmethod
    def plot(cls):
        fig,ax = plt.subplots(3,2,constrained_layout=True)
        ax[0,0].imshow(cls.image[0,0], origin="lower")
        ax[0,0].set_title('Image')
        a = ax[1,0].imshow(cls.theoretical_derivative[0,0], origin="lower")
        plt.colorbar(a,ax=ax[1,0],fraction=0.046, pad=0.04)
        ax[1,0].set_title('theoretical derivative_x')
        b = ax[2,0].imshow(cls.theoretical_derivative[0,1], origin="lower")
        plt.colorbar(b,ax=ax[2,0],fraction=0.046, pad=0.04)
        ax[2,0].set_title('theoretical derivative_y')

        c = ax[1,1].imshow(cls.derivative[0,0,0], origin="lower")
        plt.colorbar(c,ax=ax[1,1],fraction=0.046, pad=0.04)
        ax[1,1].set_title('sobel derivative_x')
        d = ax[2,1].imshow(cls.derivative[0,0,1], origin="lower")
        plt.colorbar(d,ax=ax[2,1],fraction=0.046, pad=0.04)
        ax[2,1].set_title('sobel derivative_y')
        plt.show()

    def test_values_of_derivative(self):
        eps = 1e-2
        self.assertTrue(
            self.score.max() < eps,
            "The max difference between theoretical and finite difference "
            "derivative of D_x f(x,y) should be <"
            f"{eps} but was {self.score.max()}"
        )

class TestExp3D_square(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n TestExp3D_square setup")
        rgi = tb.RandomGaussianImage((100, 80,50), 10, 'square',
                                     # a =[-1,1],
                                     # b = [.1,.1],
                                     # c = [[.5,.3],[.5,.7]]
                                     )
        cls.image =  rgi.image()
        cls.theoretical_derivative = rgi.derivative()
        print(f"image shape : {cls.image.shape}")
        cls.derivative = tb.spatialGradient(cls.image, dx_convention="square")
        print(f"theoretical_derivative shape : {cls.theoretical_derivative.shape}")
        print(f"derivative shape : {cls.derivative.shape}")
        cls.score = (cls.derivative[...,1:-1,1:-1,1:-1]
                     - cls.theoretical_derivative[...,1:-1,1:-1,1:-1]).abs()

        if args.plot:
            cls.plot()


    @classmethod
    def plot(cls):
        print(f"TestExp3d_square, the difference between theoretical and finite difference derivative :")
        print(f"\tscore : max { cls.score.max()};\n\t min {cls.score.min()};\n\t mean {cls.score.mean()};\n\t std {cls.score.std()}\n\t")
        nv = napari.Viewer()
        nv.add_image(cls.image[0,0],name="image")
        nv.add_image(cls.theoretical_derivative[0,0],name="X theoretical_derivative")
        nv.add_image(cls.theoretical_derivative[0,1],name="Y theoretical_derivative")
        nv.add_image(cls.theoretical_derivative[0,2],name="Z theoretical_derivative")
        nv.add_image(cls.derivative[0,0,0],name="X derivative")
        nv.add_image(cls.derivative[0,0,1],name="Y derivative")
        nv.add_image(cls.derivative[0,0,2],name="Z derivative")
        nv.add_image(cls.score[0,0,0],name="X score")
        nv.add_image(cls.score[0,0,1],name="Y score")
        nv.add_image(cls.score[0,0,2],name="Z score")
        napari.run()


    def test_values_of_derivative(self):
        eps = .1
        self.assertTrue(
            self.score.mean() + self.score.std() < eps,
            "The max difference between theoretical and finite difference "
            "derivative of D_x f(x,y) should have mean + std <"
            f"{eps} but was: max {self.score.max()} mean {self.score.mean()} and std {self.score.std()}"
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



#%%
# import torch
# import matplotlib.pyplot as plt
# import demeter.utils.torchbox as tb
#
# H, W = (300, 500)
# yy, xx = torch.meshgrid(
#     torch.linspace(0,1, H, dtype=torch.float),
#     torch.linspace(0,1, W, dtype=torch.float),
#     indexing='ij'
# )
#
# a,b = 8.7, 5.7
# image = torch.cos(a * xx) + torch.sin(b * xx) * torch.cos (a * yy)
# image = image[None, None].to(torch.float64)
#
# theoretical_derivative_x = -a * torch.sin(a * xx) + b * torch.cos(b * xx) * torch.cos (a * yy)
# theoretical_derivative_y = torch.sin(b * xx) * -a * torch.sin(a * yy)
#
# derivative = tb.spatialGradient(image, dx_convention="pixel")
# derivative[:,0,0] *= W -1
# derivative[:,0,1] *= H -1
#
# diff_x  = (theoretical_derivative_x[1:-1,1:-1] - derivative[0, 0, 0,1:-1,1:-1]).abs()
# diff_y  = (theoretical_derivative_y[1:-1,1:-1] - derivative[0, 0, 1,1:-1,1:-1]).abs()
#
# print(f"diff_x : max { diff_x.max()}; min {diff_x.min()}; mean {diff_x.mean()}, sum {diff_x.sum()}")
# print(f"diff_y : max { diff_y.max()}; min {diff_y.min()}; mean {diff_y.mean()}, sum {diff_y.sum()}")
# #%%
# fig, ax = plt.subplots(3, 3, constrained_layout=True)
# ax[0, 0].imshow(image[0, 0], cmap="gray")
# ax[0, 0].set_title("f(x,y) = torch.cos(a * xx) + torch.sin(b * yy)")
# ax[1, 0].imshow(derivative[0, 0, 0], cmap="gray")
# ax[1, 0].set_title('Spatial gradient x')
# ax[2, 0].imshow(derivative[0, 0, 1], cmap="gray")
# ax[2, 0].set_title('Spatial gradient y')
#
# ax[0, 1].plot(image[0, 0,:, W // 2])
# ax[1, 1].imshow(theoretical_derivative_x, cmap="gray")
# ax[1, 1].set_title("D_x f(x,y) = -a * sin(a * xx) + b * cos(b * yy)")
# ax[2, 1].imshow(theoretical_derivative_y, cmap="gray")
# ax[2, 1].set_title("D_y f(x,y) = sin(b * xx) * -b * sin(b * yy)")
#
# a = ax[1, 2].imshow(diff_x, cmap="RdYlGn_r")
# ax[1, 2].set_title("difference D_x")
# fig.colorbar(a, ax=ax[1, 2],)
#
# b = ax[2, 2].imshow(diff_y, cmap="RdYlGn_r")
# ax[2, 2].set_title("difference D_y")
# fig.colorbar(b, ax=ax[2, 2])
# fig.suptitle("dx convention on the square $[0,1]^2$")
# plt.show()

#%%




 

"""
import torch
import matplotlib.pyplot as plt
import demeter.utils.torchbox as tb

D,H,W = (100, 300, 150)
zz,yy,xx = torch.meshgrid(
    torch.linspace(0, 1,D, dtype=torch.float),
    torch.linspace(0, 1,H, dtype=torch.float),
    torch.linspace(0, 1,W, dtype=torch.float),
    indexing='ij'
)

# self.image = xx + yy + zz
# self.image = image[None, None].to(torch.float64)
#
# theoretical_derivative_x = xx
# theoretical_derivative_y = torch.zeros_like(theoretical_derivative_x)
# theoretical_derivative_z = torch.zeros_like(theoretical_derivative_x)

a1,a2 = 3,3
b1 = .7
c1,c2 = 1.5,3

image = (a1 * torch.exp(-((a2*(xx - 0.5))**2 + (a2*(yy - 0.5))**2 + (a2*(zz - 0.5))**2))
         + b1 * (xx + yy + zz)
         + c1 * torch.sin(c2 * torch.pi *xx)
              * torch.cos(c2 * torch.pi *yy)
         )
# image = xx + 2*yy + 3*zz
image = image[None, None].to(torch.float64)
_,_,Dw,Hw,Ww, = image.size()
print(f"Dw {Dw} Hw {Hw} Ww {Ww}")
# Terms in derivatives
exp_term = a1 * torch.exp(-((a2*(xx - 0.5))**2 + (a2*(yy - 0.5))**2 + (a2*(zz - 0.5))**2))
sin_term_x = c1 * c2 * torch.pi * torch.cos(c2 * torch.pi * xx) * torch.cos(c2 * torch.pi * yy)
sin_term_y = -c1 * c2 * torch.pi * torch.sin(c2 * torch.pi * xx) * torch.sin(c2 * torch.pi * yy)

# Derivatives
theoretical_derivative_x = exp_term * (-2 * a2**2 * (xx - 0.5)) + b1 + sin_term_x
theoretical_derivative_y = exp_term * (-2 * a2**2 * (yy - 0.5)) + b1 + sin_term_y
theoretical_derivative_z = exp_term * (-2 * a2**2 * (zz - 0.5)) + b1

# theoretical_derivative_x = torch.ones_like(xx) * 1/(D - 1)
# theoretical_derivative_y = torch.ones_like(yy) * 1/(H - 1) * 2
# theoretical_derivative_z = torch.ones_like(zz) * 1/(W - 1) * 3

derivative = tb.spatialGradient(image, dx_convention="square")

# derivative[0,0,0] *= (Ww-1)
# derivative[0,0,1] *= (Hw-1)
# derivative[0,0,2] *= (Dw-1)

print(f"derivative min {derivative.min()} max {derivative.max()} mean {derivative.mean()}")
print("diff x: \n",
(derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs().max()
)
print("diff y: \n",
(derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max()
)
print("diff z: \n",
(derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max()
)

print(f"derivative "
      f"\n\tmin {derivative.min()} max {derivative.max()} mean {derivative.mean()}"
      f"\n\tmidle value {derivative[0,0,:,D//2,H//2,W//2]}")
print(f"theoretical_derivative_x"
      f"\n\t min {theoretical_derivative_x.min()} max {theoretical_derivative_x.max()} mean {theoretical_derivative_x.mean()}"
      f"\n\tmidle value {theoretical_derivative_x[D//2,H//2,W//2]}, {theoretical_derivative_y[D//2,H//2,W//2]}, {theoretical_derivative_z[D//2,H//2,W//2]}")



fig,ax = plt.subplots(1,3, constrained_layout=True)
ax[0].plot(image[0,0,D//2,H//2,:],label="image")
ax[0].plot(theoretical_derivative_x[D//2,H//2,:],label="theoretical")
ax[0].plot(derivative[0,0,0,D//2,H//2,:],"--",label="spatialGradient")

ax[1].plot(image[0,0,:,H//2,W//2],label="image")
ax[1].plot(theoretical_derivative_y[:,H//2,W//2],label="theoretical")
ax[1].plot(derivative[0,0,1,:,H//2,W//2],"--",label="spatialGradient")

ax[2].plot(image[0,0,:,H//2,W//2],label="image")
ax[2].plot(theoretical_derivative_z[:,H//2,W//2],label="theoretical")
ax[2].plot(derivative[0,0,2,:,H//2,W//2],"--",label="spatialGradient")
plt.legend()
plt.show()



#%%


min_p = min(derivative.min(),
                        float(theoretical_derivative_x.min()),
                        float(theoretical_derivative_y.min()),
                        float(theoretical_derivative_z.min())
            )
max_p = max(derivative.max(),
                float(theoretical_derivative_x.max()),
                float(theoretical_derivative_y.max()),
                float(theoretical_derivative_z.max())
            )

fix,ax = plt.subplots(3,4, constrained_layout=True)

im_kw = dict(cmap="gray",vmin=image.min(),vmax=image.max())
ax[0,0].imshow(image[0,0,D//2],**im_kw)
ax[0,0].set_title("f(x,y,z) = custom")
ax[1,0].imshow(image[0,0,:,H//2],**im_kw)
ax[2,0].imshow(image[0,0,:,:,W//2],**im_kw)

der_kw = dict(cmap="gray",vmin=min_p,vmax=max_p)
ax[0,1].set_title("D_x f(x,y,z)")
ax[0,1].imshow(derivative[0,0,0,D//2],**der_kw)
ax[1,1].imshow(derivative[0,0,0,:,H//2],**der_kw)
ax[2,1].imshow(derivative[0,0,0,:,:,W//2],**der_kw)

ax[0,2].set_title("D_y f(x,y,z)")
ax[0,2].imshow(derivative[0,0,1,D//2],**der_kw)
ax[1,2].imshow(derivative[0,0,1,:,H//2],**der_kw)
ax[2,2].imshow(derivative[0,0,1,:,:,W//2],**der_kw)

ax[0,3].set_title("D_z f(x,y,z)")
ax[0,3].imshow(derivative[0,0,2,D//2],**der_kw)
ax[1,3].imshow(derivative[0,0,2,:,H//2],**der_kw)
ax[2,3].imshow(derivative[0,0,2,:,:,W//2],**der_kw)
plt.show()
#%%
fix,ax = plt.subplots(3,4, constrained_layout=True)

im_kw = dict(cmap="gray",vmin=image.min(),vmax=image.max())

ax[0,0].imshow(image[0,0,D//2],**im_kw)
ax[0,0].set_title("f(x,y,z) = xx + yy + zz")
ax[1,0].imshow(image[0,0,:,H//2],**im_kw)
ax[2,0].imshow(image[0,0,:,:,W//2],**im_kw)

ax[0,1].set_title("D_x f(x,y,z) theorique")
ax[0,1].imshow(theoretical_derivative_x[D//2],**der_kw)
ax[1,1].imshow(theoretical_derivative_x[:,H//2],**der_kw)
ax[2,1].imshow(theoretical_derivative_x[:,:,W//2],**der_kw)

ax[0,2].set_title("D_y f(x,y,z) theorique")
ax[0,2].imshow(theoretical_derivative_y[D//2],**der_kw)
ax[1,2].imshow(theoretical_derivative_y[:,H//2],**der_kw)
ax[2,2].imshow(theoretical_derivative_y[:,:,W//2],**der_kw)

ax[0,3].set_title("D_z f(x,y,z) theorique")
ax[0,3].imshow(theoretical_derivative_z[D//2],**der_kw)
ax[1,3].imshow(theoretical_derivative_z[:,H//2],**der_kw)
ax[2,3].imshow(theoretical_derivative_z[:,:,W//2],**der_kw)

plt.show()
#%%
fig,ax = plt.subplots(3,3,constrained_layout=True,figsize = (10,10))
ax[0,0].plot(derivative[0,0,0,:,H//2,W//2],label="spatialGradient")
ax[0,0].plot(theoretical_derivative_x[:,H//2,W//2],'--',label="theoretical")
ax[0,0].set_title("D_x f(x,y,z)")
ax[0,0].set_ylabel("X")

ax[0,1].plot(derivative[0,0,1,:,H//2,W//2],label="spatialGradient")
ax[0,1].plot(theoretical_derivative_y[:,H//2,W//2],'--',label="theoretical")
ax[0,1].set_title("D_y f(x,y,z)")
ax[0,1].set_ylabel("X")


ax[0,2].plot(derivative[0,0,2,:,H//2,W//2],label="spatialGradient")
ax[0,2].plot(theoretical_derivative_z[:,H//2,W//2],'--',label="theoretical")
ax[0,2].set_title("D_z f(x,y,z)")
ax[0,1].set_ylabel("X")

ax[1,0].plot(derivative[0,0,0,D//2,:,W//2],label="spatialGradient")
ax[1,0].plot(theoretical_derivative_x[D//2,:,W//2],'--',label="theoretical")
ax[1,0].set_title("D_x f(x,y,z)")
ax[1,0].set_ylabel("Y")

ax[1,1].plot(derivative[0,0,1,D//2,:,W//2],label="spatialGradient")
ax[1,1].plot(theoretical_derivative_y[D//2,:,W//2],'--',label="theoretical")
ax[1,1].set_title("D_y f(x,y,z)")
ax[1,1].set_ylabel("Y")


ax[1,2].plot(derivative[0,0,2,D//2,:,W//2],label="spatialGradient")
ax[1,2].plot(theoretical_derivative_z[D//2,:,W//2],'--',label="theoretical")
ax[1,2].set_title("D_z f(x,y,z)")
ax[1,1].set_ylabel("Y")




ax[2,0].plot(derivative[0,0,0,D//2,W//2],label="spatialGradient")
ax[2,0].plot(theoretical_derivative_x[D//2,W//2],'--',label="theoretical")
ax[2,0].set_title("D_x f(x,y,z)")
ax[2,0].set_ylabel("Z")

ax[2,1].plot(derivative[0,0,1,D//2,W//2],label="spatialGradient")
ax[2,1].plot(theoretical_derivative_y[D//2,W//2],'--',label="theoretical")
ax[2,1].set_title("D_y f(x,y,z)")
ax[2,1].set_ylabel("Z")


ax[2,2].plot(derivative[0,0,2,D//2,W//2],label="spatialGradient")
ax[2,2].plot(theoretical_derivative_z[D//2,W//2],'--',label="theoretical")
ax[2,2].set_title("D_z f(x,y,z)")
ax[2,1].set_ylabel("Z")



plt.legend()
plt.show()

#%%
import torch
import matplotlib.pyplot as plt
import demeter.utils.torchbox as tb
from demeter.utils.constants import *
from kornia.geometry.transform import resize
import random

def generate_random_positions(img, num_positions):
    # 
    # Génère une liste de tuples contenant des positions aléatoires dans une image donnée,
    # en s'assurant que ces positions correspondent à des endroits où l'image n'est pas nulle.
    # 
    # :param img: Image tensor (2D ou 3D) de type torch.Tensor
    # :param num_positions: Nombre de positions aléatoires à générer
    # :return: Liste de tuples contenant les positions aléatoires
    # 
    if not torch.is_tensor(img):
        raise ValueError("L'image doit être un tensor PyTorch")

    # Trouver les indices où l'image n'est pas nulle
    non_zero_indices = torch.nonzero(img, as_tuple=False)

    if len(non_zero_indices) == 0:
        raise ValueError("L'image ne contient aucune valeur non nulle")

    # Sélectionner des indices aléatoires parmi les indices non nuls
    random_indices = random.sample(range(len(non_zero_indices)), num_positions)
    random_positions = [tuple(non_zero_indices[i].tolist()) for i in random_indices]

    return random_positions

views = generate_random_positions(I[0,0], 10)
#%%

# open the image /home/turtlefox/Documents/11_metamorphoses/Demeter_metamorphosis/examples/im2Dbank/BraTS2021_00147_80_.png
path = ROOT_DIRECTORY+ '/examples/im2Dbank/'

I = tb.rgb2gray(plt.imread(path+'BraTS2021_00147_80_.png'))
original_shape = I.shape
I = torch.tensor(I[None,None,:],dtype=torch.float)
n_obs = 40
factor = (2,2)
# views = generate_random_positions(I[0,0], n_obs)
new_size = (int(original_shape[0]*factor[0]),int(original_shape[1]*factor[1]))
print(f"new_size {new_size}")
I_p = resize(I,new_size)
print(f"I.shape {I.shape}")

def grad_pix2square(grad_pic):
    _,_,_,H,W = grad_pic.shape
    grad_pic_2 = grad_pic.clone()
    grad_pic_2[:,0,0] *= (H - 1)
    grad_pic_2[:,0,1] *= (W - 1)
    return grad_pic_2


I_grad_pixel = tb.spatialGradient(I, dx_convention="pixel")
# I_grad_square = tb.spatialGradient(I, dx_convention="square")
I_grad_square = grad_pix2square(I_grad_pixel)

I_p_grad_pixel = tb.spatialGradient(I_p, dx_convention="pixel")
# I_p_grad_square = tb.spatialGradient(I_p, dx_convention="square")
I_p_grad_square = grad_pix2square(I_p_grad_pixel)


# views = [(100,100),(150,70),(150,150),(134,50),(80,50),(120,120)]

view_p = [(round(factor[0] * v[0]),round(factor[1] * v[1]))
          for v in views]
print(view_p)
ratio_pix_stock = torch.zeros((n_obs,2))
ratio_sq_stock = torch.zeros((n_obs,2))
for i,(v,v_p) in enumerate(zip(views,view_p)):
    print(f"v {v} v_p {v_p}")
    Ig_pix = I_grad_pixel[0,0,:,v[0],v[1]]
    Ig_sq = I_grad_square[0,0,:,v[0],v[1]]
    Ig_p_pix = I_p_grad_pixel[0,0,:,v_p[0],v_p[1]]
    Ig_p_sq = I_p_grad_square[0,0,:,v_p[0],v_p[1]]
    ratio_pix = Ig_pix / Ig_p_pix
    ratio_sq = Ig_sq / Ig_p_sq
    ratio_pix_stock[i] = ratio_pix
    ratio_sq_stock[i] = ratio_sq
    print(f"\tI {I[0,0,v[0],v[1]]} vs I_p {I_p[0,0,v_p[0],v_p[1]]}")
    print(f"\tI_grad_pixel {Ig_pix} vs "
          f"\tI_p_grad_pixel {Ig_p_pix}"
          f"\n\t\t abs diff {abs(Ig_pix 
                             - Ig_p_pix)}"
          f"\n\t\t ratio {Ig_pix / Ig_p_pix} or {Ig_p_pix / Ig_pix}"
            )
    print(f"\tI_grad_square {Ig_sq} vs "
            f"\tI_p_grad_square {Ig_p_sq}"
          f"\n\t\t abs diff {abs(Ig_sq 
                             - Ig_p_sq )}"
          f"\n\t\t ratio {Ig_sq / Ig_p_sq} or {Ig_p_sq / Ig_sq}"
            )
print(f"ratio_pix_stock "
      f"\n\tmean: {ratio_pix_stock.mean(dim=0)}"
      f"\n\tstd: {ratio_pix_stock.std(dim=0)}"
      f"\n\tmin: {ratio_pix_stock.min(dim=0).values}"
      f"\n\tmax: {ratio_pix_stock.max(dim=0).values}")
print(f"ratio_sq_stock "
        f"\n\tmean: {ratio_sq_stock.mean(dim=0)}"
        f"\n\tstd: {ratio_sq_stock.std(dim=0)}"
        f"\n\tmin: {ratio_sq_stock.min(dim=0).values}"
        f"\n\tmax: {ratio_sq_stock.max(dim=0).values}")

def norm2(x):
    return x.sum()
norm_I_pix = norm2(I_grad_pixel)
norm_I_sq = norm2(I_grad_square)
norm_I_p_pix = norm2(I_p_grad_pixel)
norm_I_p_sq = norm2(I_p_grad_square)
print(f"norm_I_pix {norm_I_pix} norm_I_sq {norm_I_sq} \nnorm_I_p_pix {norm_I_p_pix} norm_I_p_sq {norm_I_p_sq}")
print(f"ratio norms pix: {norm_I_pix / norm_I_p_pix}, sq: {norm_I_sq / norm_I_p_sq}")


fig, ax = plt.subplots(2, 3, constrained_layout=True)
ax[0,0].imshow(I[0, 0], cmap="gray",origin="lower")
# ax[0,0].plot(view[1],view[0],'rx')
ax[0,0].set_title("f(x,y) = image")
ax[0,1].imshow(I_grad_pixel[0, 0, 0], cmap="gray",origin="lower")
# ax[0,1].plot(view[1],view[0],'rx')
ax[0,1].set_title('Spatial gradient x pixel')
ax[0,2].imshow(I_grad_pixel[0, 0, 0], cmap="gray",origin="lower")
# ax[0,2].plot(view[1],view[0],'rx')
ax[0,2].set_title('Spatial gradient x square')
for a in ax[0]:
    for v in views:
        a.plot(v[1],v[0],'rx')

ax[1,0].imshow(I_p[0, 0], cmap="gray",origin="lower")
# ax[1,0].plot(view_p[1],view_p[0],'rx')
ax[1,0].set_title("f(x,y) = image")
ax[1,1].imshow(I_p_grad_pixel[0, 0, 0], cmap="gray",origin="lower")
# ax[1,1].plot(view_p[1],view_p[0],'rx')
ax[1,1].set_title('Spatial gradient x pixel')
ax[1,2].imshow(I_p_grad_pixel[0, 0, 0], cmap="gray",origin="lower")
# ax[1,2].plot(view_p[1],view_p[0],'rx')
ax[1,2].set_title('Spatial gradient x square')
for a in ax[1]:
    for v in view_p:
        a.plot(v[1],v[0],'rx')

plt.show()


"""