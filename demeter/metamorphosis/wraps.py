import torch
import __init__
import sys
from icecream import ic



# import metamorphosis as mt
from . import classic as cl
from . import constrained as cn

from  ..utils import torchbox as tb
from ..utils.decorators import time_it

@time_it
def lddmm(source,target,residuals,
          sigma,cost_cst,
          integration_steps,n_iter,grad_coef,
          data_term =None,
          sharp=False,
          safe_mode = False,
          integration_method='semiLagrangian',
          multiScale_average = False,
          dx_convention = 'pixel',
          optimizer_method='adadelta',
          ):
    if type(residuals) in [int,float]:
        residuals = residuals * torch.ones(source.shape,device=source.device)
    residuals.requires_grad = True
    if sharp: integration_method = 'sharp'

    # sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = cl.Metamorphosis_integrator(method=integration_method,
                        mu=0, rho=0,
                        n_step=integration_steps,
                        sigma_v=sigma,
                        multiScale_average=multiScale_average,
                        dx_convention=dx_convention
                        )
    mr = cl.Metamorphosis_Shooting(source,target,mp,
                                cost_cst=cost_cst,
                                # optimizer_method='LBFGS_torch',
                                optimizer_method=optimizer_method,
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
                  integration_method='semiLagrangian',
                  optimizer_method='adadelta',
                  dx_convention = 'pixel'):
    if type(residuals) == int: residuals = torch.zeros(source.shape,device=source.device)
    # residuals = torch.zeros(source.size()[2:],device=device)
    residuals.requires_grad = True
    if sharp: integration_method= 'sharp'
    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = cl.Metamorphosis_integrator(method=integration_method,
                        mu=mu, rho=rho,
                        sigma_v=sigma,
                        n_step=integration_steps,
                        dx_convention=dx_convention
                        )
    mr = cl.Metamorphosis_Shooting(source,target,mp,
                                cost_cst=cost_cst,
                                data_term=data_term,
                                # optimizer_method='LBFGS_torch')
                                optimizer_method=optimizer_method)
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr

# ==================================================
#  From contrained.py

@time_it
def weighted_metamorphosis(source,target,residual,mask,
                           mu,rho,rf_method,sigma,cost_cst,
                           n_iter,grad_coef,data_term=None,sharp=False,
                           safe_mode=True,
                           optimizer_method='adadelta',
                           dx_convention = 'pixel'):
    device = source.device
#     sigma = tb.format_sigmas(sigma,len(source.shape[2:]))
    if rf_method == 'identity':
        rf = cn.Residual_norm_identity(mask.to(device),mu,rho)
    elif rf_method == 'borderBoost':
        rf = cn.Residual_norm_borderBoost(mask.to(device),mu,rho)
    else:
        raise ValueError(f"rf_method must be 'identity' or 'borderBoost'")
    if type(residual) == int: residual = torch.zeros(source.shape,device=device)
    residual.requires_grad = True

    mp_weighted = cn.ConstrainedMetamorphosis_integrator(
        residual_function=rf,sigma_v=sigma,
        sharp=sharp,
        dx_convention=dx_convention
    )
    mr_weighted = cn.ConstrainedMetamorphosis_Shooting(
        source,target,mp_weighted,
        cost_cst=cost_cst,
        optimizer_method=optimizer_method,
        data_term=data_term
    )
    if not safe_mode:
        mr_weighted.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr_weighted.forward_safe_mode(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_weighted

@time_it
def oriented_metamorphosis(source,target,residual,mp_orienting,
                           mu,rho,gamma,sigma,cost_cst,
                           n_iter,grad_coef,
                           dx_convention = 'pixel'):
    mask = mp_orienting.image_stock.to(source.device)
    orienting_field = mp_orienting.field_stock.to(source.device)
    if type(residual) == int: residual = torch.zeros(source.shape)
    residual.requires_grad = True

    # start = time.time()
    mp_orient = cn.ConstrainedMetamorphosis_integrator(orienting_mask=mask,
                                      orienting_field=orienting_field,
                                mu=mu,rho=rho,gamma=gamma,
                                sigma_v=(sigma,)*len(residual.shape),
                                dx_convention=dx_convention
                                # n_step=20 # n_step is defined from mask.shape[0]
                                )
    mr_orient = cn.ConstrainedMetamorphosis_Shooting(source,target,mp_orient,
                                       cost_cst=cost_cst,
                                       # optimizer_method='LBFGS_torch')
                                       optimizer_method='adadelta')
    mr_orient.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_orient

@time_it
def constrained_metamorphosis(source,target,residual,
                           rf_method,mu,rho,mask_w,
                           mp_orienting,gamma,mask_o,
                           sigma,cost_cst,sharp,
                           n_iter,grad_coef,
                              dx_convention = 'pixel'):
    mask = mp_orienting.image_stock.to(source.device)
    orienting_field = mp_orienting.field_stock.to(source.device)
#     sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    if rf_method == 'identity':
        rf_method = cn.Residual_norm_identity(mask,mu,rho)
    elif rf_method == 'borderBoost':
        rf_method = cn.Residual_norm_borderBoost(mask,mu,rho)
    else:
        raise ValueError(f"rf_method must be 'identity' or 'borderBoost'")
    if type(residual) == int: residual = torch.zeros(source.shape,device=source.device)
    residual.requires_grad = True

    # start = time.time()
    mp_constr = cn.ConstrainedMetamorphosis_integrator(orienting_mask=mask,
                                      orienting_field=orienting_field,
                                      residual_function=rf_method,
                                mu=mu,rho=rho,gamma=gamma,
                                sigma_v=sigma,
                                sharp=sharp,
                                dx_convention=dx_convention
                                # n_step=20 # n_step is defined from mask.shape[0]
                                )
    mr_constr = cn.ConstrainedMetamorphosis_Shooting(source,target,mp_constr,
                                       cost_cst=cost_cst,
                                       # optimizer_method='LBFGS_torch')
                                       optimizer_method='adadelta')
    mr_constr.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_constr
