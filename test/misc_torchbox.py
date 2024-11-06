import __init__
import torch
import demeter.utils.torchbox as tb
import unittest
from parameterized import parameterized

#%% Test regular grid

# dx_conventions_list = ['pixel','2square','square']
# shape_list = [(1,4,6,2),(4,6),(1,5,7,9,3),(5,7,9)]
class Test_make_regular_grid(unittest.TestCase):

    @parameterized.expand([
        ["3d_grid",(1,5,7,9,3),(1,5,7,9,3)],
        ["3d_ltl",(5,7,9),(1,5,7,9,3)],
        ["2d_grid",(1,4,6,2),(1,4,6,2)],
        ["3d_grid",(4,6),(1,4,6,2)]
        ]
    )
    def test_size(self,name,input,output):
        mesh = tb.make_regular_grid(input)
        self.assertEqual(mesh.shape,output)

    @parameterized.expand([
        ["pixel_2d",(1,4,6,2),'pixel',0],
        ["pixel_3d",(1,5,7,9,3),'pixel',0],
        ["2square_2d",(1,4,6,2),'2square',-1],
        ["2square_2d",(1,4,6,2),'2square',-1],
        ["square",(1,4,6,2),'square',0],
        ["square",(1,5,7,9,3),'square',0],

    ])
    def test_min(self,name,size,dx,mini):
        mesh = tb.make_regular_grid(size,dx_convention=dx)
        self.assertEqual(mesh.min(),mini)

    @parameterized.expand([
        ["pixel_2d",(1,4,6,2),'pixel',5],
        ["pixel_3d",(1,5,7,9,3),'pixel',8],
        ["2square_2d",(1,4,6,2),'2square',1],
        ["2square_2d",(1,4,6,2),'2square',1],
        ["square",(1,4,6,2),'square',1],
        ["square",(1,5,7,9,3),'square',1],
    ])
    def test_max(self,name,size,dx,maxi):
        mesh = tb.make_regular_grid(size,dx_convention=dx)
        self.assertEqual(mesh.max(),maxi)

if __name__ == '__main__':
    unittest.main()