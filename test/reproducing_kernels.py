import unittest
from math import sqrt, log
import torch

import __init__
import demeter.utils.reproducing_kernels as rk
from demeter.utils.decorators import time_it
from demeter.utils.reproducing_kernels import plot_gaussian_kernel_1d


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

class TestGetGaussianKernel1D(unittest.TestCase):

    def test_gaussian_kernel(self):
        for sigma in [1, 3, 10, 100, 500, 1000]:
            with self.subTest(sigma=sigma):
                kernel = rk.get_gaussian_kernel1d(sigma)
                kernel_size = kernel.shape[1]
                dist_from_center = float(kernel_size) / 2
                x = torch.linspace(-dist_from_center, dist_from_center, kernel_size)

                # Vérifier que la taille du kernel est correcte
                # self.assertEqual(kernel_size, 2 * int(3 * sigma) + 1)

                # Vérifier que la valeur maximale du kernel est 1
                self.assertAlmostEqual(kernel.max().item(), 1.0, places=6,
                                       msg=f"for sigma={sigma} max was not 1, kernel.max()={kernel.max().item()}")

                # Vérifier que la valeur minimale du kernel est inférieure à 0.1
                self.assertLess(kernel.min().item(), 0.1,
                                f"for sigma={sigma} min was not < 0.1, kernel.min()={kernel.min().item()}")

                # Vérifier que la valeur à la position -sigma est supérieure à 0.6
                self.assertGreater(kernel[0, x > -sigma][0].item(), 0.6,
                                   f"for sigma={sigma} value at position -sigma was not > 0.6, kernel[0, x > -sigma][0]={kernel[0, x > -sigma][0].item()}")

                # Vérifier que la valeur à la position sigma est supérieure à 0.6
                self.assertGreater(kernel[0, x < sigma][-1].item(), 0.6,
                                   f"for sigma={sigma} value at position sigma was not > 0.6, kernel[0, x < sigma][-1]={kernel[0, x < sigma][-1].item()}")



if __name__ == '__main__':
    unittest.main()

#%%
import torch
import demeter.utils.reproducing_kernels as rk
import matplotlib.pyplot as plt
#%%
img = torch.rand(1,1,100,150,200)
result = rk.get_sigma_from_img_ratio(
         img.shape,
    (11,16,21)
)

print("result",result)
strr = rk.GaussianRKHS(result)
print(strr)

#%%
import matplotlib.pyplot as plt
vv = rk.GaussianRKHS((5,5))
fig, ax = plt.subplots()
ax.imshow(vv.kernel[0])
plt.show()

#%%
big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)

sigmas = torch.linspace(0,50,100)
fig, ax = plt.subplots(1,2,figsize=(7,4),constrained_layout=True)
ax[0].plot(sigmas,[big_odd(i) for i in sigmas])
ax[0].set_ylabel('kernel size')
ax[0].set_xlabel('sigma')
ax[0].grid(linestyle='--')


subdivs = torch.arange(5,100)
ax[1].plot(subdivs,[rk._get_sigma_monodim(100,i) for i in subdivs])
ax[1].set_xlabel('subdiv')
ax[1].set_ylabel('sigma')
ax[1].set_title('sigma for a 100 pixel strip')
ax[1].grid(linestyle='--')

plt.show()

#%%
rk.get_sigma_from_img_ratio()

#%%
import torch
%load_ext autoreload
%autoreload 3
import demeter.utils.reproducing_kernels as rk
import matplotlib.pyplot as plt
#%%
%reload_ext autoreload
#%%
sigma = (5,15)
dx_convention = (1./100,1./100)
kernelOp = rk.GaussianRKHS(sigma,dx_convention=dx_convention)
kernelOp.plot()
# fig, axes = plt.subplots(2, 2, figsize=(10, 5))
plt.show()

#%%


#%%
import torch
import demeter.utils.bspline as bs
import demeter.utils.torchbox as tb
from math import prod, log10
import demeter.utils.reproducing_kernels as rk
import matplotlib.pyplot as plt
from demeter.utils.decorators import time_it

@time_it
def compute_V_norm(momentum,image,kernelOperator,dx_convention,dx):
    grad_source = tb.spatialGradient(image,dx_convention=dx_convention)
    print("grad_source",grad_source.shape)
    grad_source_resi = (grad_source * momentum).sum(dim=1) #/ C
    print("grad_source_resi",grad_source_resi.shape)
    K_grad_source_resi = kernelOperator(grad_source_resi)
    print("K_grad_source_resi",K_grad_source_resi.shape)
    norm_V = (grad_source_resi * K_grad_source_resi).sum() /(prod(dx))
    print("norm_V",norm_V)
    return norm_V, K_grad_source_resi

def compute_z_norm(momentum):
    return (momentum**2).sum() / prod(momentum.shape)

def log_diff(a,b):
    return abs(log10(a) - log10(b))

#%%

dx_convention = 'pixel'
# Construction d'une iamge et de moments  bsplines aléatoires à partir d'une matrice
# de points de controle
P_im = torch.rand((10,10),dtype=torch.float)
P_mom = torch.rand((10,10),dtype=torch.float)*2 - 1
s = 4
size = (150,175)
size_s = tuple([int(i*s) for i in size])
# On défini les noyaux
sigma = (5,20)
sigma_s = tuple([i*s for i in sigma]) if dx_convention == 'pixel' else tuple([i for i in sigma])
dx = (1,1) if dx_convention == 'pixel' else (1./(size[0]-1),1./(size[1]-1))
dx_s = (1,1) if dx_convention == 'pixel' else (1./(size_s[0]-1),1./(size_s[1]-1))

img = bs.surf_bspline(P_im,size,degree=(2,2))[None,None]
img_s = bs.surf_bspline(P_im,size_s,degree=(2,2))[None,None]
moments = bs.surf_bspline(P_mom,size,degree=(2,2))[None,None]
moments_s = bs.surf_bspline(P_mom,size_s,degree=(2,2))[None,None]
# field_s = bs.field2D_bspline(P,size_s,dim_stack=-1)
id_grid = tb.make_regular_grid(size,dx_convention=dx_convention)
id_grid_s = tb.make_regular_grid(size_s,dx_convention=dx_convention)

kernelOp = rk.GaussianRKHS(sigma,dx_convention=dx)
kernelOp_s = rk.GaussianRKHS(sigma_s,dx_convention=dx_s)
# calcul de la norme de V (et en même temps du champ)
norm_V, field = compute_V_norm(moments,img,kernelOp,dx_convention,dx)
norm_V_s, field_s = compute_V_norm(moments_s,img_s,kernelOp_s,dx_convention,dx_s)
field = tb.im2grid(field)
field_s = tb.im2grid(field_s)

print(f"norm_V = {norm_V:.4e}, norm_V_s = {norm_V_s:.4e}, diff = {log_diff(norm_V, norm_V_s):.4e}")
print(f"norm2 on q {compute_z_norm(moments):.4e}, on q_s {compute_z_norm(moments_s):.4e}, diff = {log_diff(compute_z_norm(moments), compute_z_norm(moments_s)):.4e}")

# fig, ax = plt.subplots(1,2,figsize=(10,5))
# # ax[0].imshow(moments[0,0])
# # ax[1].imshow(moments_s[0,0])
# step = 5
# ax[0].imshow(img[0,0].T )
# ax[0].quiver(id_grid[0,::step,::step,0],id_grid[0,::step,::step,1],
#           field[0,::step,::step,0],field[0,::step,::step,1])
# step = int(step*s)
# ax[1].imshow(img_s[0,0].T)
# ax[1].quiver(id_grid_s[0,::step,::step,0],id_grid_s[0,::step,::step,1],
#             field_s[0,::step,::step,0],field_s[0,::step,::step,1])
# plt.show()

#%% 3D  !!!

dx_convention = 'square'
# Construction d'une iamge et de moments  bsplines aléatoires à partir d'une matrice
# de points de controle
P_im = torch.rand((20,20,20),dtype=torch.float)
P_mom = torch.rand((20,20,20),dtype=torch.float)*2 - 1
s = 2
size = (201,151,41)
size_s = tuple([int(i*s) for i in size])
sigma = (4,4,4)
sigma_s = tuple([i*s for i in sigma]) if dx_convention == 'pixel' else tuple([i for i in sigma])
dx = (1,1,1) if dx_convention == 'pixel' else (1./size[0],1./size[1],1./size[2])
dx_s = (1,1,1) if dx_convention == 'pixel' else (1./size_s[0],1./size_s[1],1./size_s[2])

img = bs.surf_bspline_3D(P_im,size,degree=(2,2,2))[None,None]
img_s = bs.surf_bspline_3D(P_im,size_s,degree=(2,2,2))[None,None]
moments = bs.surf_bspline_3D(P_mom,size,degree=(2,2,2))[None,None]
moments_s = bs.surf_bspline_3D(P_mom,size_s,degree=(2,2,2))[None,None]
# field_s = bs.field2D_bspline(P,size_s,dim_stack=-1)
id_grid = tb.make_regular_grid(size,dx_convention=dx_convention)
id_grid_s = tb.make_regular_grid(size_s,dx_convention=dx_convention)

# On défini les noyaux
kernelOp = rk.GaussianRKHS(sigma,dx_convention=dx)
kernelOp_s = rk.GaussianRKHS(sigma_s,dx_convention=dx_s)
# calcul de la norme de V (et en même temps du champ)
norm_V, field = compute_V_norm(moments,img,kernelOp,dx_convention,dx)
norm_V_s, field_s = compute_V_norm(moments_s,img_s,kernelOp_s,dx_convention,dx_s)
field = tb.im2grid(field)
field_s = tb.im2grid(field_s)

print(f"norm_V = {norm_V:.4e}, norm_V_s = {norm_V_s:.4e}, diff = {log_diff(norm_V, norm_V_s):.4e}")
print(f"norm2 on q {compute_z_norm(moments):.4e}, on q_s {compute_z_norm(moments_s):.4e}, diff = {log_diff(compute_z_norm(moments), compute_z_norm(moments_s)):.4e}")

#%%

a,b = 1e-3,2e-3
print(log_diff(a,b))

#%%
# def surf_bspline_3D(cm,n_pts, degree = (1,1,1)):
#     """ Generate a 3D surface from a control matrix
#
#     :param cm     = 3D matrix Control point Matrix
#     :param n_pts  = (tuple), number of points on the curve.
#     :param degree = (tuple), degree of the spline in each direction
#     :return:
#     """
#
#     p,q,r = cm.shape
#     d,h,w = n_pts
#     print("p,q,r",p,q,r)
#     print("d,h,w",d,h,w)
#
#     b_p = bs.bspline_basis(p,d,degree[0])
#     b_q = bs.bspline_basis(q,h,degree[1])
#     b_r = bs.bspline_basis(r,w,degree[2])
#
#
#     Q_i = torch.einsum('ij,jkl->ikl', b_p, cm)
#     Q_ij = torch.einsum('ij,jkl->ikl', b_q, Q_i.transpose(0, 1)).transpose(0, 1)
#     surf_3d = torch.einsum('ij,jkl->ikl', b_r, Q_ij.transpose(0, 2)).transpose(0, 2)
#     return surf_3d



P_im = torch.rand((5,6,7),dtype=torch.float)
img = surf_bspline_3D(P_im,(100,200,300))