"""


"""

from math import prod, sqrt

from demeter.metamorphosis import Geodesic_integrator, Optimize_geodesicShooting

import demeter.utils.torchbox as tb
from demeter.utils.decorators import monitor_gpu
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

    def _update_residuals_simplex(self, momentum, image):
        try:
            volDelta = prod(self.kernelOperator.dx)
        except AttributeError:
            volDelta = 1
         ## 2.1 Compute the scalar product of the momentum and the image,
        # It will be used later as well
        pi_q = (momentum * image).sum(dim=1, keepdim=True) / (
            image**2
        ).sum(dim=1, keepdim=True)
        ic(volDelta, self.rho, )
        if self.rho < 1:
            residuals = (
                (1 - self.rho) * (momentum - pi_q * image) / volDelta
            )
        else:
            residuals = torch.zeros_like(image)

        assert not residuals.isnan().any(), "Residuals is Nan"

        if self.flag_hamiltonian_integration:
            # self.norm_v_i = 0.5 * (field_momentum * field).sum()
            self.norm_z_i = 0.5 * (residuals**2).sum() * volDelta
            # self.ham_value = norm_l2_on_z + norm_v_2

        return residuals, pi_q


    def step(self, image, momentum):

        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences

        field = self._update_field_(momentum, image)

        ## 2. Compute the residuals
        residuals, pi_q = self._update_residuals_simplex(momentum, image)

        ## 3. Compute the image update
        deform = self.id_grid - self.rho * field / self.n_step
        ic(momentum.shape,
           image.shape,
           deform.shape,
            residuals.shape,
           )
        image = self._update_image_semiLagrangian_(momentum, image, deform, residuals)

        # cette ligne remet l'image sur la sphere, c'est un test pour
        # voir la stabilité du shémas numérique.
        image = image / (image**2).sum(dim=1, keepdim=True).sqrt()
        assert not image.isnan().any(), "Image is Nan"


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



class Simplex_sqrt_Metamorphosis_integrator_bis(Geodesic_integrator):
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
            f"Simplex_srqt_Metamorphosis_integrator_bis("
            f"\n\trho={self.rho},"
            f"\n\tkernelOperator={self.kernelOperator},"
            f"\n\tn_step={self.n_step}\n)"
        )

    def _update_residuals_simplex(self, momentum, image):
        try:
            volDelta = prod(self.kernelOperator.dx)
        except AttributeError:
            volDelta = 1
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

        return residuals, pi_q



    def step(self, image, momentum):

        ## 1. Compute the vector field
        field = self._update_field_(momentum, image)

        ## 2. Compute the residuals
        residuals, pi_q = self._update_residuals_simplex(momentum, image)

        ## 3. Compute the image update
        deform = self.id_grid - self.rho * field / self.n_step
        image = self._update_image_semiLagrangian_(momentum, image, deform, residuals)

        # cette ligne remet l'image sur la sphere, c'est un test pour
        # voir la stabilité du shémas numérique.
        image = image / (image**2).sum(dim=1, keepdim=True).sqrt()
        assert not image.isnan().any(), "Image is Nan"


        assert not momentum.isnan().any(), "Momentum is Nan"

        # return self.image, self.field,self.momentum, self.residuals
        return (
            image,
            sqrt(self.rho) * field,
            residuals,
        )

    def forward(
        self,
        image,
        momentum,
        save=True,
        plot=0,
        t_max=1,
        verbose=False,
        sharp=None,
        debug=False,
        hamiltonian_integration=False,
    ):
        r"""This method is doing the temporal loop using the good method `_step_`

        Parameters
        ----------
        image : tensor array of shape [1,1,H,W]
            Source image ($I_0$)
        momentum : tensor array of shape [T,1,H,W]
            Momentum ($p_0$) or residual ($z_0$)
        save : bool, optional
            Option to save the integration intermediary steps, by default True
            it saves the image, field and momentum at each step in the attributes
            `image_stock`, `field_stock`, `residuals_stock` and `momentum_stock`.
        plot : int, optional
            Positive int lower than `self.n_step` to plot the indicated number of
            intermediary steps, by default 0
        t_max : int, optional
            The integration will be made on [0,t_max], by default 1
        verbose : bool, optional
            Option to print the progress of the integration, by default False
        sharp : bool, optional
            Option to use the sharp integration, by default None
        debug : bool, optional
            Option to print debug information, by default False
        hamiltonian_integration : bool, optional
            Choose to integrate over first time step only or whole hamiltonian, in
            practice when True, the Regulation norms of the Hamiltonian are computed
            and saved in the good attributes (usually `norm_v` and `norm_z`),
             by default False

        """
        if len(momentum.shape) not in [4, 5]:
            raise ValueError(
                f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {momentum.shape}"
            )
        device = momentum.device
        # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
        self._init_sharp_(sharp)
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.momentum = momentum
        self.debug = debug
        self.flag_hamiltonian_integration = hamiltonian_integration
        try:
            self.save = True if self._force_save else save
        except AttributeError:
            self.save = save

        self.id_grid = tb.make_regular_grid(
            momentum.shape[2:], dx_convention=self.dx_convention, device=device
        )
        assert self.id_grid != None

        # field initialization to a regular grid
        field = self.id_grid.clone().to(device)

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((t_max * self.n_step,) + image.shape[1:])
            self.field_stock = torch.zeros(
                (t_max * self.n_step,) + field.shape[1:]
            )
            self.momentum_stock = torch.zeros(
                (t_max * self.n_step,) + momentum.shape[1:]
            )
            self.residuals_stock = torch.zeros_like(self.momentum_stock)

        if self.flag_hamiltonian_integration:
            self.norm_v = 0
            self.norm_z = 0

        for i, t in enumerate(torch.linspace(0, t_max, t_max * self.n_step)):
            self._i = i
            if self.save_gpu_memory:
                self.image , field, residuals = torch.utils.checkpoint.checkpoint(
                    self.step,
                    self.image,
                    self.momentum[i][None],
                    use_reentrant = True,
                )
            else:
                self.image, field, residuals = self.step(self.image, self.momentum[i][None])

            if self.flag_hamiltonian_integration:
                self.norm_v += self.norm_v_i / self.n_step
                self.norm_z += self.norm_z_i / self.n_step
                # self.ham_integration += self.ham_value / self.n_step
            # ic(self._i,self.field.min().item(),self.field.max().item(),
            #    self.momentum.min().item(),self.momentum.max().item(),
            #     self.image.min().item(),self.image.max().item())

            if self.image.isnan().any() or self.momentum.isnan().any():
                raise OverflowError(
                    "Some nan where produced ! the integration diverged",
                    "changing the parameters is needed. "
                    "You can try:"
                    "\n- increasing n_step (deformation more complex"
                    "\n- decreasing grad_coef (convergence slower but more stable)"
                    "\n- increasing sigma_v (catching less details)",
                )

            if self.save:
                if self._detach_image:
                    self.image_stock[i] = self.image[0].detach().to("cpu")
                else:
                    self.image_stock[i] = self.image[0]
                self.field_stock[i] = field[0].detach().to("cpu")
                # self.momentum_stock[i] = self.momentum.detach().to("cpu")
                self.residuals_stock[i] = residuals[0].detach().to("cpu")

            # save some gpu memory:
            if device != 'cpu':
                del field, residuals
                torch.cuda.empty_cache()

            if verbose:
                update_progress(i / (t_max * self.n_step))
                if self.flag_hamilt_integration:
                    print('ham :', self.ham_value.detach().cpu().item(),
                      self.norm_v.detach().cpu().item(),
                      self.norm_z.detach().cpu().item())

        # try:
        #     _d_ = device if self._force_save else 'cpu'
        #     self.field_stock = self.field_stock.to(device)
        # except AttributeError: pass

        if plot > 0:
            self.plot(n_figs=plot)



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
