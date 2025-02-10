"""
.. _semiLagrangian_scheme:

Why we should use semi-Lagrangian scheme?
=========================================


In this example I will try to demonstrate to you why we should use the semi-Lagrangian
scheme during geodesic integration rather than Eulerian. Obviously, we could
use even more developed schemes like Runge-Kutta, but semi-Lagrangian is a good
compromise between speed and accuracy.

"""

######################################################################
# Importing the necessary libraries
import matplotlib.pyplot as plt

import demeter.metamorphosis as mt
import demeter.utils.reproducing_kernels as rk
import demeter.utils.torchbox as tb
from demeter.constants import DLT_KW_IMAGE


######################################################################
# Open images

size = (200,200)
S = tb.reg_open('08',size=size)
T = tb.reg_open('m0',size=size)

fig,ax = plt.subplots(1,3)
ax[0].imshow(S[0,0].cpu(),**DLT_KW_IMAGE)
ax[0].set_title('source')
ax[1].imshow(T[0,0].cpu(),**DLT_KW_IMAGE)
ax[1].set_title('target')
ax[2].imshow(tb.imCmp(S,T,'seg'),origin='lower')
ax[2].set_title('superposition of S and T')

######################################################################
# Now we try to register the source to the target using the Eulerian scheme.
kernel_op = rk.GaussianRKHS(sigma=(6, 6))

momentum_ini = 0
mr = mt.lddmm(S, T, momentum_ini,
      kernelOperator = kernel_op,
      cost_cst=.0001,
      integration_steps=100,
      n_iter=15,
      grad_coef=1,
      integration_method="Eulerian",
      dx_convention='pixel',
)
mr.plot()
mr.plot_deform()
mr.mp.plot()
plt.show()