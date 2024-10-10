import unittest
from math import sqrt, log
import torch

import __init__
import demeter.utils.reproducing_kernels as rk

class TestGetSigmaFromImgRatio(unittest.TestCase):

    def setUp(self):
        self.image_2d = torch.rand(1, 1, 100, 150)
        self.image_3d = torch.rand(1, 1, 50, 100, 150)

    def test_single_subdiv_2d(self):
        result = rk.get_sigma_from_img_ratio(self.image_2d.shape, 5)
        expected = rk.get_sigma_from_img_ratio((100, 150), 5)
        self.assertEqual(result, expected)

    def test_tuple_subdiv_2d(self):
        result = rk.get_sigma_from_img_ratio(self.image_2d.shape, (5, 10), c=.01)
        expected = rk.get_sigma_from_img_ratio((100, 150), (5, 10), c=.01)
        self.assertEqual(result, expected)

    def test_list_of_tuples_subdiv_2d(self):
        result = rk.get_sigma_from_img_ratio(self.image_2d.shape, [(5, 10), (10, 20)])
        expected = rk.get_sigma_from_img_ratio((100, 150), [(5, 10), (10, 20)])
        self.assertEqual(result, expected)

    def test_invalid_subdiv_length_2d(self):
        with self.assertRaises(ValueError):
            rk.get_sigma_from_img_ratio(self.image_2d.shape, (5, 10, 4))

    def test_invalid_subdiv_list_length_2d(self):
        with self.assertRaises(ValueError):
            rk.get_sigma_from_img_ratio(self.image_2d.shape, [(5, 10), (10, 20, 4)])

    def test_single_subdiv_3d(self):
        result = rk.get_sigma_from_img_ratio(self.image_3d.shape, 5)
        expected = rk.get_sigma_from_img_ratio((50, 100, 150), 5)
        self.assertEqual(result, expected)

    def test_tuple_subdiv_3d(self):
        result = rk.get_sigma_from_img_ratio(self.image_3d.shape, (5, 10, 15), c=.01)
        expected = rk.get_sigma_from_img_ratio((50, 100, 150), (5, 10, 15), c=.01)
        self.assertEqual(result, expected)

    def test_list_of_tuples_subdiv_3d(self):
        result = rk.get_sigma_from_img_ratio(self.image_3d.shape, [(5, 10, 15), (10, 20, 30)])
        expected = rk.get_sigma_from_img_ratio((50, 100, 150), [(5, 10, 15), (10, 20, 30)])
        self.assertEqual(result, expected)

    def test_invalid_subdiv_length_3d(self):
        with self.assertRaises(ValueError):
            rk.get_sigma_from_img_ratio(self.image_3d.shape, (5, 10))

    def test_invalid_subdiv_list_length_3d(self):
        with self.assertRaises(ValueError):
            rk.get_sigma_from_img_ratio(self.image_3d.shape, [(5, 10, 15), (10, 20)])

    def test_list_of_values_2d(self):
         result = rk.get_sigma_from_img_ratio(
             self.image_2d.shape,
             [[5, 10],
                    [1],
                    [10, 20,30]]
         )
         expected = [
             [
                rk.get_sigma_from_img_ratio(self.image_2d,5),
                rk.get_sigma_from_img_ratio(self.image_2d,10)
             ],
             [rk.get_sigma_from_img_ratio(self.image_2d,1)],
             [
                rk.get_sigma_from_img_ratio(self.image_2d,10),
                rk.get_sigma_from_img_ratio(self.image_2d,20),
                rk.get_sigma_from_img_ratio(self.image_2d,30)
             ]
         ]
         self.assertEqual(result, expected,
                          f"Expected: {expected}\nGot: {result}")

if __name__ == '__main__':
    unittest.main()