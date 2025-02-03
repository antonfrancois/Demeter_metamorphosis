
# TODO : faire une fonction de composition 'simple'
# TODO : rendre les fonctions compatibles pour des images en 3D

import torch
from torch.nn.functional import grid_sample
from math import log,ceil

from . import torchbox as tb


 # TODO : Faire docstring
class FieldIntegrator:
    def __init__(self,method,save=False,N=None,dx_convention = 'pixel'):
        r"""

        :param method:
            'fast_exp' :
            'slow_exp' :
            'temporal' :
        :param save:
        :param N:   division number of the vector field
            it must be of the form $\forall n \in \mathbb Z, 2^n$
            default : choose automaticaly the good N
        """
        self.save = save
        self.N = N
        self.flag = False
        self.dx_convention = dx_convention
        self.kwargs = dict(padding_mode="border", align_corners=True)
        if method == 'fast_exp':
            self.integrator = self._fast_exp_integrator
        elif method  == 'slow_exp':
            self.integrator = self._slow_exp_integrator
        elif method == 'temporal':
            self.N = -1 # we don't need to have N
            self.integrator = self._temporal_field_integrator


    def __call__(self,in_vectField,forward=True,verbose = False):
        self.shape = in_vectField.shape
        if self.dx_convention == 'pixel':
            self.in_vectField = tb.pixel_to_2square_convention(in_vectField, is_grid=False)
        elif self.dx_convention == 'square':
            self.in_vectField = tb.square_to_2square_convention(in_vectField,is_grid=False)
        elif self.dx_convention == '2square':
            self.in_vectField = in_vectField.clone()

        if not torch.is_tensor(in_vectField):
            raise TypeError("field2diffeo has been written for torch objects")
        device = 'cuda' if in_vectField.is_cuda else 'cpu'


        if self.N is None:
            # set N to the optimal N
            self._find_optimal_N_exp_(verbose)
            self.flag = True

        # choose forward or backward integration
        self.sign = 1 if forward else -1

        self.id_grid = tb.make_regular_grid(in_vectField.shape,
                                            device=device,
                                            dx_convention='2square')

        integrated = self.integrator()

        if self.dx_convention == 'pixel':
            integrated = tb.square2_to_pixel_convention(integrated)
        elif self.dx_convention == 'square':
            integrated = tb.square2_to_square_convention(integrated)
        if self.flag:
            self.N = None
        return integrated
    # ==================================
    #
    #         Exponentials
    #
    # =================================
    def _find_optimal_N_exp_(self,verbose):
        field_max = float(tb.fieldNorm2(self.in_vectField).max())
        # TODO : implement for 3D !
        if field_max > 0:# if x <0, log(x) throw an error
            max_size = max(self.shape)
            self.N = max(1,ceil((log(field_max) - log(1/max_size) )/log(2)))
        else:
            self.N = 1
        if verbose:
            print('N = ',self.N,'=> n_step = ',2**self.N)

    def _fast_exp_integrator(self):

        self.in_vectField *= 1/(2**self.N)

        grid_def = self.id_grid + self.sign*self.in_vectField

        if self.save:
            vectField_stock = torch.zeros((self.N,)+self.shape)
            vectField_stock[0] = grid_def.detach().to('cpu')

        # Composition du champs divisé 2**N fois
        slave_grid = torch.zeros(grid_def.shape,
                                 device=self.in_vectField.device)

        for n in range(1,self.N+1):
            field = tb.grid2im(grid_def-self.id_grid)
            slave_grid.data = grid_def.data
            interp_vectField = grid_sample(field,slave_grid,**self.kwargs)

            grid_def += tb.im2grid(interp_vectField)

            if self.save:
                vectField_stock[n] = grid_def.detach().to('cpu')

        if self.save:
            return vectField_stock
        else:
            return grid_def

    def _slow_exp_integrator(self):
        self.in_vectField *= 1/(2**self.N)

        grid_def = self.id_grid + self.sign*self.in_vectField

        if self.save:
            vectField_stock = torch.zeros((2**self.N,)+self.shape)
            vectField_stock[0] = grid_def.detach().to('cpu')

        in_vectField_im = tb.grid2im(self.in_vectField)
        slave_grid = torch.zeros(grid_def.shape,
                                 device=self.in_vectField.device)
        for n in range(1,2**self.N+1):
            slave_grid.data = grid_def.data
            interp_vectField = grid_sample(in_vectField_im,slave_grid,**self.kwargs)

            grid_def = grid_def + self.sign*tb.im2grid(interp_vectField)

            if self.save:
                vectField_stock[n] = grid_def.detach().to('cpu')

        if self.save:
            return vectField_stock
        else:
            return grid_def

    # ==================================
    #
    #         Temporal
    #
    # =================================

    def _temporal_field_integrator(self):
        if self.sign == 1:
            return self._temporal_field_integrator_forward()
        elif self.sign == -1:
            self.in_vectField = -self.in_vectField.flip(0)
            return self._temporal_field_integrator_forward()

    def _temporal_field_integrator_forward(self):

        grid_def = self.id_grid + self.in_vectField[0].unsqueeze(0)
        if self.save:
            vectField_stock = torch.zeros(self.in_vectField.shape)
            vectField_stock[0,...] = grid_def.detach().to('cpu')

        in_vectField_im = tb.grid2im(self.in_vectField)
        # slave_grid = torch.zeros(grid_def.shape,
        #                          device=self.in_vectField.device)

        for t in range(1,self.in_vectField.shape[0]):
            slave_grid = grid_def.detach()
            interp_vectField = grid_sample(in_vectField_im[t,...].unsqueeze(0)
                                           ,slave_grid,**self.kwargs)

            grid_def += tb.im2grid(interp_vectField)

            if self.save:
                vectField_stock[t,...] = grid_def.detach().to('cpu')

        if self.save:
            return vectField_stock
        else:
            return grid_def


"""
import matplotlib.pyplot as plt
import my_torchbox as tb
from math import pi
import time as time
import my_bspline.my_bspline as mbs
Tau,H,W = (3,200,200)
step = 10
grid = tb.make_regular_grid((1,H,W,2))

# # construction du champ de vecteur temporel
r = .5
y_mod = torch.linspace(0,pi/2,Tau)
v = torch.zeros((Tau,H,W,2))
coef_mat = torch.exp(-5*(grid[0,...,0]**2 +grid[0,...,1]**2)).unsqueeze(-1)

for t in range(Tau):
    v[t,:,:,:] = (coef_mat*torch.stack((torch.sin(grid[0,...,1]),
                             torch.cos(grid[0,...,0]+pi/2)),dim=2)).unsqueeze(0)
v = v[0].unsqueeze(0)

# vector field generation
# cms = mbs.getCMS_turn()
# v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)
# v *= 0.5

#%%
start = time.time()
integrated = FieldIntegrator(method='temporal',save=True)(v.clone(),forward=True)
end = time.time()
#%%
t_plot = Tau
for t_i in range(t_plot):
    tb.vectField_show(v[t_i].unsqueeze(0),step=step,title='v['+str(t_i)+']')
    tb.deformation_show(integrated[t_i].unsqueeze(0),step=step,title='$\int_0^'+str(t_i)+'v(t) dt$')
plt.show()
#%%
I = tb.reg_open('02',(H,W))
T = tb.imgDeform(I,integrated[-1].unsqueeze(0))

plt.figure()
plt.imshow(T[0,0,:,:],cmap='gray')
plt.show()
#%%
temporal_integrator = FieldIntegrator(method='temporal',save=False)
integrated = temporal_integrator(v.clone(),forward=True)
rev_interagrated = temporal_integrator(v.clone(),forward = False)

tb.deformation_show(integrated,step=step)
tb.deformation_show(rev_interagrated,step=step)
tb.vectField_show(integrated -rev_interagrated,step=step)

exp_v = FieldIntegrator(method='fast_exp',save=False)(v.clone(),forward=True)
I = tb.reg_open('grid',size=(H,W))
I_v = torch.nn.functional.grid_sample(I,exp_v)
# I_v = tb.imgDeform(I,rev_interagrated)
# I_v_mv = tb.imgDeform(I_v,v)
# I_v = grid_sample(I,v)

# tb.gridDef_plot(inv_v,add_grid=True,step =10)
plt.figure()
plt.imshow(I[0,0,:,:],cmap='gray',origin='lower')
plt.figure()
plt.imshow(I_v[0,0,:,:],cmap='gray',origin='lower')
# plt.figure()
# plt.imshow(I_v_mv[0,0,:,:],cmap='gray')
plt.show()

"""
#%%
# ===================
#   TEST
# ===================
"""
import my_bspline.my_bspline as mbs
import matplotlib.pyplot as plt
import time as t

H,W = (100,100)

cms = mbs.getCMS_allcombinaision()

# vector field generation
v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)

grid = tb.make_regular_grid(v.shape)
yy = grid[0,:,:,0]
xx = grid[0,:,:,1]
# xx, yy = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H))

tb.vectField_show(v,step=2,check_diffeo=True,title='original vector field')


start = t.time()
fast_integrator  = vff.FieldIntegrator(method='fast_exp',save= False)
diff = fast_integrator(v.clone(),forward=True)
print('N =',fast_integrator.N)
# diff = vff.FieldIntegrator(method='fast_exp',save=False)(v.clone(),forward=True)
end = t.time()
print('fast exp executed in ',end - start,' s')
tb.deformation_show(diff,step=2,check_diffeo=True,title='fast_exp')


start = t.time()
diff_s = vff.FieldIntegrator(method='slow_exp',save=False)(v.clone(),forward=True)
end = t.time()
print('slow exp executed in ',end - start,' s')

tb.deformation_show(diff_s,step=2,check_diffeo=True,title='slow_exp')
"""



