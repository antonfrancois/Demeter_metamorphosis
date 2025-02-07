"""
.. data_cost_mutualInformation:

Data cost variation example:: Mutual Information
===============================================

This example demonstrates how to change the data cost for the registration.
We will build a simple toy example mimicking situations we can encounter in
medical imaging where the shapes to match are clode in geometry but have
different intensity distributions. There are plenty other data costs in the literature
for every specific purpose, and we will focus on the mutual information.

Mutual information measures the amount of information shared between two images. It is effective for multi-modal image registration.

$$I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \left(\frac{p(x,y)}{p(x)p(y)}\right)$$

Where:

- $X$ and $Y$ are the images being registered.
- $p(x,y)$ is the joint probability distribution of the intensities.
- $p(x)$ and $p(y)$ are the marginal probability distributions of the intensities.

"""

######################################################################
# Importing the necessary libraries
from demeter.constants import *
import demeter.utils.torchbox as tb
import demeter.metamorphosis as mt
import demeter.utils.reproducing_kernels as rk

######################################################################
# Openning the source and target images


size = (200,200)
S = tb.reg_open('01',size=size)
T = 1 -tb.reg_open('17',size=size)

fig,ax = plt.subplots(1,2)
ax[0].imshow(S[0,0].cpu(),**DLT_KW_IMAGE)
ax[0].set_title('source')
ax[1].imshow(T[0,0].cpu(),**DLT_KW_IMAGE)
ax[1].set_title('target')
set_ticks_off(ax)
plt.show()

######################################################################
# As you see in the previous plot, using the Ssd as a data cost will lead to have the ball
# badly registered. (Try predicting the result before running the code !). However, we can
# use the mutual information as a data cost to get a better registration.
#
#
# Now we will create a mutual information data cost object and use it in the registration.

data_term = mt.Mutual_Information(T,mult=10)
# data_term = mt.Ssd(T)

momentum_ini = 0
# momentum_ini = mr.to_analyse[0]
# momentum_ini.requires_grad = True
kernelOp = rk.GaussianRKHS(sigma=(10,10))

mr = mt.lddmm(S,T,momentum_ini,
                kernelOperator = kernelOp,
                cost_cst=.0001,
                integration_steps=5,
                n_iter=150,
                grad_coef=100,
                data_term=data_term
)
mr.plot()
mr.plot_deform()
