import torch
import __init__
import sys
from icecream import ic
ic(sys.path)

import metamorphosis as mt
import utils.torchbox as tb
from utils.decorators import time_it

@time_it
def lddmm(source,target,residuals,
          sigma,cost_cst,
          integration_steps,n_iter,grad_coef,
          data_term =None,
          sharp=False,
          safe_mode = False,
          integration_method='semiLagrangian',
          multiScale_average = False
          ):
    if type(residuals) == int: residuals = torch.zeros(source.shape,device=source.device)
    residuals.requires_grad = True
    if sharp: integration_method = 'sharp'

    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = mt.Metamorphosis_integrator(method=integration_method,
                        mu=0, rho=0,
                        sigma_v=sigma,
                        n_step=integration_steps,
                        multiScale_average=multiScale_average
                        )
    mr = mt.Metamorphosis_Shooting(source,target,mp,
                                cost_cst=cost_cst,
                                # optimizer_method='LBFGS_torch',
                                optimizer_method='adadelta',
                                data_term=data_term
                               )
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr

@time_it
def metamorphosis(source,target,residuals,
                  mu,rho,sigma,cost_cst,
                  integration_steps,n_iter,grad_coef,
                  data_term=None,
                  sharp=False,
                  safe_mode = True,
                  integration_method='semiLagrangian'
                  ):
    if type(residuals) == int: residuals = torch.zeros(source.shape,device=source.device)
    # residuals = torch.zeros(source.size()[2:],device=device)
    residuals.requires_grad = True
    if sharp: integration_method= 'sharp'
    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = mt.Metamorphosis_integrator(method=integration_method,
                        mu=mu, rho=rho,
                        sigma_v=sigma,
                        n_step=integration_steps,
                        )
    mr = mt.Metamorphosis_Shooting(source,target,mp,
                                cost_cst=cost_cst,
                                data_term=data_term,
                                # optimizer_method='LBFGS_torch')
                                optimizer_method='adadelta')
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr