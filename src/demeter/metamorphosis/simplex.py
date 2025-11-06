"""


"""

from math import prod, sqrt

from demeter.metamorphosis import Geodesic_integrator, Optimize_geodesicShooting

import demeter.utils.torchbox as tb
from demeter.utils.toolbox import update_progress
import torch




class Simplex_sqrt_Metamorphosis_integrator(Geodesic_integrator):
    # TODO : Add equations in the docstring
    """
    Class integrating over a geodesic shooting for images with values in the simplex.

    ..note::
        In this class we integrate on the sphere using the square root function and
        reproject images on the simplex after.
        Thus, when accessing the image attribute, one should put it to the square
        to get the image in the simplex.
        >>> mp = Simplex_sqrt_Metamorphosis_integrator(rho=1, kernelOperator=kernelOperator, n_step=10)
        >>> mp.forward(image, momentum_ini,)
        >>> image_on_simplex  = mp.image**2
        >>> print( image_on_simplex.sum(dim=1) )

    Parameters
    ----------
    rho : float
        Control parameter for geodesic shooting intensities changes vs deformations.
        rho = 1 is LDDMM (pure deformation), rho = 0 is pure photometric changes.
        Any value in between is a mix of both.
    kernelOperator : callable
        The reproducing kernel operator $K$ that one must define to compute the field.
    n_step : int
        Number of time step in the segment [0,1] over the geodesic integration.
    dx_convention : str
        Convention for the pixel size. Default is 'pixel'.


    """
    def __init__(self, rho, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho

    def _get_mu_(self):
        return 1

    def _get_rho_(self):
        return self.rho

    def __repr__(self):
        return (
            f"Simplex_srqt_Metamorphosis_integrator("
            f"\n\trho={self.rho},"
            f"\n\tkernelOperator={self.kernelOperator},"
            f"\n\tn_step={self.n_step}\n)"
        )

    def step(self, image, momentum):

        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences

        field = self._update_field_(momentum, image)

        try:
            volDelta = prod(self.kernelOperator.dx)
        except AttributeError:
            volDelta = 1

        ## 2. Compute the residuals
        ## 2.1 Compute the scalar product of the momentum and the image,
        # It will be used later as well
        pi_q = (momentum * image).sum(dim=1, keepdim=True) / (
            image**2
        ).sum(dim=1, keepdim=True)
        residuals = (
            (1 - self.rho) * (momentum - pi_q * image) / volDelta
        )

        assert not residuals.isnan().any(), "Residuals is Nan"

        if self.flag_hamiltonian_integration:
            # self.norm_v_i = 0.5 * (field_momentum * field).sum()
            self.norm_z_i = 0.5 * (residuals**2).sum() * volDelta
            # self.ham_value = norm_l2_on_z + norm_v_2

        # self.residuals = (self.momentum - pi_q * self.image) / self.rho

        # ic(self.residuals.min(),self.residuals.max(),self.residuals[0,:,15,80])

        ## 3. Compute the image update
        deform = self.id_grid - self.rho * field / self.n_step
        image = (
            tb.imgDeform(image, deform, dx_convention=self.dx_convention)
            +  residuals / self.n_step
        )
        # cette ligne remet l'image sur la sphere, c'est un test pour
        # voir la stabilité du shémas numérique.
        image = image / (image**2).sum(dim=1, keepdim=True).sqrt()
        assert not image.isnan().any(), "Image is Nan"

        # ic(self.image.min(),self.image.max(),self.image[0,:,15,80])

        ## 4. Compute the momentum update
        div_v_times_p = (
            momentum
            * tb.Field_divergence(dx_convention=self.dx_convention)(field)[0, 0]
        )
        momentum = (
            tb.imgDeform(momentum, deform, dx_convention=self.dx_convention)
            + (
                sqrt(1 - self.rho) * residuals * pi_q  # * volDelta
                - sqrt(self.rho) * div_v_times_p
            )
            / self.n_step
        )

        assert not momentum.isnan().any(), "Momentum is Nan"

        # return self.image, self.field,self.momentum, self.residuals
        return (
            momentum,
            image,
            self.rho * field,
            residuals,
        )



class Simplex_sqrt_Shooting(Optimize_geodesicShooting):
    # TODO: add docstring
    def __init__(
        self,
        source,
        target,
        integrator,
        cost_cst,
        hamiltonian_integration=False,
        **kwargs,
    ):
        if source.min() < 0 or target.min() < 0:
            raise ValueError(f"Provided images must be positive ! Got source.min() = {source.min()} and target.min() = {target.min()}")
        source = torch.sqrt(source)
        target = torch.sqrt(target)
        self.flag_hamiltonian_integration = hamiltonian_integration
        super().__init__(source, target, integrator, cost_cst, **kwargs)
        # self._cost_saving_ = self._simplex_cost_saving_

    # def _get_mu_(self):
    #     return self.mp._get_mu_()

    def _get_rho_(self):
        return self.mp._get_rho_()

    def get_all_arguments(self):
        # TODO:  use super for kernelOp, n_step ....
        return {
            "rho": self.mp.rho,
            "kernelOperator": self.mp.kernelOperator,
            "n_step": self.mp.n_step,
            "cost_cst": self.cost_cst,
        }

    def cost(self, momentum_ini: torch.Tensor):

        rho = self._get_rho_()
        # Geodesic shooting.
        self.mp.forward(
            self.source,
            momentum_ini,
            save=False,
            plot=0,
            hamiltonian_integration=self.flag_hamiltonian_integration,
        )


        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()


        # ic(float(self.norm_v_2),float(self.norm_l2_on_z))
        if self.flag_hamiltonian_integration:
            self.total_cost = self.data_loss + (self.cost_cst) * self.mp.ham_integration
        else:
            self.norm_v_2 = 0.5 * rho * self._compute_V_norm_(momentum_ini, self.source)
            volDelta = prod(self.dx)

            pi_q = ((momentum_ini / volDelta) * self.source).sum(dim=1, keepdim=True) / (
                self.source**2
            ).sum(dim=1, keepdim=True)
            z = sqrt(1 - rho) * (momentum_ini / volDelta - pi_q * self.source)
            # self.norm_l2_on_z = .5 * (z ** 2).sum() * prod(self.dx) # /prod(self.source.shape[2:])
            self.norm_l2_on_z = 0.5 * (z**2).sum() * volDelta

            self.total_cost = self.data_loss + (self.cost_cst) * (
                self.norm_v_2 + self.norm_l2_on_z
            )

        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost
