import sys

from sympy.physics.quantum.gate import normalized

import __init__
import torch
import matplotlib.pyplot as plt
import unittest
import argparse

import demeter.utils.torchbox as tb
from demeter.utils.constants import color




#%% Coordonées pixeliques.


#%%

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
        cls.D, cls.H, cls.W = (6, 10, 5)
        zz, yy, xx = torch.meshgrid(
            torch.arange(0, cls.D, dtype=torch.long),
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            indexing='ij'
        )

        cls.image = xx + yy + zz
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = xx
        cls.theoretical_derivative_y = yy
        cls.theoretical_derivative_z = zz

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
        cls.D, cls.H, cls.W = (6, 10, 5)
        zz, yy, xx = torch.meshgrid(
            torch.arange(0, cls.D, dtype=torch.long),
            torch.arange(0, cls.H, dtype=torch.long),
            torch.arange(0, cls.W, dtype=torch.long),
            indexing='ij'
        )

        cls.image = xx**2 * zz**3 - xx * yy**2 + torch.sin(zz) + 2*xx*yy
        cls.image = cls.image[None, None].to(torch.float64)

        cls.theoretical_derivative_x = 2 * xx * zz**3 - yy**2 + 2*yy
        cls.theoretical_derivative_y = 2 * xx * (1 - yy)
        cls.theoretical_derivative_z = 3 * xx**2 * zz**3 + torch.cos(zz)

        cls.derivative = tb.spatialGradient(cls.image, dx_convention="pixel")
        if args.plot:
            cls.plot()

    @classmethod
    def plot(cls):
        fig, ax = plt.subplots(3, 3)
        ax[0, 0].imshow(cls.image[0, 0, cls.D // 2], cmap="gray")
        ax[0, 0].set_title("f(x,y,z) = zz**2 + yy**2 + 2*xx*yy")
        ax[1, 0].imshow(cls.derivative[0, 0, 0, cls.D // 2], cmap="gray")
        ax[1, 0].set_title('Spatial gradient x')
        ax[2, 0].imshow(cls.derivative[0, 0, 1, cls.D // 2], cmap="gray")
        ax[2, 0].set_title('Spatial gradient y')
        ax[2, 1].imshow(cls.derivative[0, 0, 2, cls.D // 2], cmap="gray")
        ax[2, 1].set_title('Spatial gradient z')

        ax[0, 1].plot(cls.image[0, 0, cls.D // 2, :, cls.W // 2])
        ax[1, 1].imshow(cls.theoretical_derivative_x[cls.D // 2], cmap="gray")
        ax[1, 1].set_title("D_x f(x,y,z) = 2*yy")
        ax[2, 1].imshow(cls.theoretical_derivative_y[cls.D // 2], cmap="gray")
        ax[2, 1].set_title("D_y f(x,y,z) = 2*yy + 2*xx")
        ax[2, 2].imshow(cls.theoretical_derivative_z[cls.D // 2], cmap="gray")
        ax[2, 2].set_title("D_z f(x,y,z) = 2*zz")
        plt.show()

    def test_values_of_derivative_direction_x(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 0, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_x[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_x f(x,y,z) = xx**2 * zz**3 - xx * yy**2 + torch.sin(zz) + 2*xx*yy should be "
            f"{self.theoretical_derivative_x} but was {self.derivative[0, 0, 0]}"
        )

    def test_values_of_derivative_direction_y(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 1, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_y[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_y f(x,y,z) = zz**2 + yy**2 + 2*xx*yy should be "
            f"{self.theoretical_derivative_y} but was {self.derivative[0, 0, 1]}"
        )

    def test_values_of_derivative_direction_z(self):
        eps = 1e-6
        self.assertTrue(
            (self.derivative[0, 0, 2, 1:-1, 1:-1, 1:-1] - self.theoretical_derivative_z[1:-1, 1:-1, 1:-1]).abs().max() < eps,
            "The derivative of D_z f(x,y,z) = zz**2 + yy**2 + 2*xx*yy should be "
            f"{self.theoretical_derivative_z} but was {self.derivative[0, 0, 2]}"
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


 #%% Coordonées 2squares

import torch
import matplotlib.pyplot as plt
import demeter.utils.torchbox as tb

D, H, W = (6, 5, 10)
yy,zz, xx = torch.meshgrid(
    torch.arange(0, D, dtype=torch.long),
    torch.arange(0, H, dtype=torch.long),
    torch.arange(0, W, dtype=torch.long),
    indexing='ij'
)

image = xx + yy + zz
image = image[None, None].to(torch.float64)

theoretical_derivative_x = xx
theoretical_derivative_y = torch.zeros_like(theoretical_derivative_x)
theoretical_derivative_z = torch.zeros_like(theoretical_derivative_x)

derivative = tb.spatialGradient(image, dx_convention="pixel")
#%%
print("0 ",derivative[0,0,0,4])
print("1 ",derivative[0,0,1,4])
print("2 ",derivative[0,0,2,4])

print("th ",theoretical_derivative_x[4])




