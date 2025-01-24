from math import prod, sqrt

from demeter.metamorphosis import Geodesic_integrator, Optimize_geodesicShooting

import demeter.utils.torchbox as tb
from demeter.utils.toolbox import update_progress
import torch


class Simplex_sqrt_Metamorphosis_integrator(Geodesic_integrator):

    def __init__(self, rho, kernelOperator, n_step=None, **kwargs):
        super().__init__(kernelOperator, **kwargs)
        self.rho = rho
        self.n_step = n_step
    #
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

    def step(self):

        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences

        grad_simplex = tb.spatialGradient(self.image, dx_convention="pixel")

        field_momentum = (grad_simplex * self.momentum.unsqueeze(2)).sum(dim=1)  # / C
        field = self.kernelOperator(field_momentum)
        self.field = -sqrt(self.rho) * tb.im2grid(field)

        try:
            volDelta = prod(self.kernelOperator.dx)
        except AttributeError:
            volDelta = 1

        ## 2. Compute the residuals
        ## 2.1 Compute the scalar product of the momentum and the image,
        # It will be used later as well
        pi_q = (self.momentum * self.image).sum(dim=1, keepdim=True) / (
            self.image**2
        ).sum(dim=1, keepdim=True)
        self.residuals = (
            sqrt(1 - self.rho) * (self.momentum - pi_q * self.image) / volDelta
        )

        if self.flag_hamilt_integration:
            norm_v_2 = 0.5 * self.rho * (field_momentum * field).sum()
            norm_l2_on_z = 0.5 * (self.residuals**2).sum() * volDelta
            self.ham_value = norm_l2_on_z + norm_v_2

        # self.residuals = (self.momentum - pi_q * self.image) / self.rho

        # ic(self.residuals.min(),self.residuals.max(),self.residuals[0,:,15,80])

        ## 3. Compute the image update
        deform = self.id_grid - sqrt(self.rho) * self.field / self.n_step
        self.image = (
            tb.imgDeform(self.image, deform, dx_convention=self.dx_convention)
            + sqrt(1 - self.rho) * self.residuals / self.n_step
        )
        # cette ligne remet l'image sur la sphere, c'est un test pour
        # voir la stabilité du shémas numérique.
        self.image = self.image / (self.image**2).sum(dim=1, keepdim=True).sqrt()

        # ic(self.image.min(),self.image.max(),self.image[0,:,15,80])

        ## 4. Compute the momentum update
        div_v_times_p = (
            self.momentum
            * tb.Field_divergence(dx_convention=self.dx_convention)(self.field)[0, 0]
        )
        self.momentum = (
            tb.imgDeform(self.momentum, deform, dx_convention=self.dx_convention)
            + (
                sqrt(1 - self.rho) * self.residuals * pi_q  # * volDelta
                - sqrt(self.rho) * div_v_times_p
            )
            / self.n_step
        )

        # return self.image, self.field,self.momentum, self.residuals
        return (
            self.image,
            sqrt(self.rho) * self.field,
            self.momentum,
            sqrt(1 - self.rho) * self.residuals,
        )

    # def hamiltonian(self):
    #     # compute the hamiltonian from the current values of image, and momentum
    #     rho = self.rho
    #
    #     # Computation of |v|^2/2
    #     grad_image = tb.spatialGradient(self.image, dx_convention=self.dx_convention)
    #     field_momentum = (grad_image * self.momentum.unsqueeze(2)).sum(dim=1) #/ C
    #     field = self.kernelOperator(field_momentum)
    #     self.norm_v_2 = .5 * rho  * (field_momentum * field).sum()
    #
    #     # Computation of |z|2/2
    #     volDelta = prod(self.kernelOperator.dx)
    #     pi_q = (self.momentum * self.image).sum(dim=1,keepdim=True) / (self.image ** 2).sum(dim=1,keepdim=True)
    #     z = sqrt(1 - rho) * (self.momentum - pi_q * self.image)/volDelta
    #     #self.norm_l2_on_z = .5 * (z ** 2).sum() * prod(self.dx) # /prod(self.source.shape[2:])
    #     self.norm_l2_on_z = .5 * (z ** 2).sum() * volDelta
    #     # ic(float(self.norm_v_2),float(self.norm_l2_on_z))
    #     return (self.norm_v_2 + self.norm_l2_on_z)

    # TODO : create a default parameter saving update to match
    # with classical metamorphosis methods,
    # s'inspirer de Constrained_Optim._oriented_cost_saving_()
    def forward(
        self,
        image,
        momentum_ini,
        field_ini=None,
        save=True,
        plot=0,
        t_max=1,
        verbose=False,
        sharp=None,
        debug=False,
        hamiltonian_integration=False,
    ):
        r"""This method is doing the temporal loop using the good method `_step_`

        :param image: (tensor array) of shape [1,1,H,W]. Source image ($I_0$)
        :param field_ini: to be deprecated, field_ini is id_grid
        :param momentum_ini: (tensor array) of shape [H,W]. initial residual ($z_0$)
        :param save: (bool) option to save the integration intermediary steps.
        :param plot: (int) positive int lower than `self.n_step` to plot the indicated
                         number of intermediary steps. (if plot>0, save is set to True)

        """
        if len(momentum_ini.shape) not in [4, 5]:
            raise ValueError(
                f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {momentum_ini.shape}"
            )
        device = momentum_ini.device
        # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
        self._init_sharp_(sharp)
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.momentum = momentum_ini
        self.residuals = torch.zeros_like(momentum_ini, device=device)
        self.debug = debug
        self.flag_hamilt_integration = hamiltonian_integration

        try:
            self.save = True if self._force_save else save
        except AttributeError:
            self.save = save

        self.id_grid = tb.make_regular_grid(
            momentum_ini.shape[2:],
            device=device,
            dx_convention=self.dx_convention,
        ).to(torch.double)
        assert self.id_grid != None

        if field_ini is None:
            self.field = self.id_grid.clone()
        else:
            self.field = field_ini  # /self.n_step

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((t_max * self.n_step,) + image.shape[1:])
            self.field_stock = torch.zeros(
                (t_max * self.n_step,) + self.field.shape[1:]
            )
            self.residuals_stock = torch.zeros(
                (t_max * self.n_step,) + momentum_ini.shape[1:]
            )
            self.momentum_stock = torch.zeros(
                (t_max * self.n_step,) + momentum_ini.shape[1:]
            )

        if self.flag_hamilt_integration:
            self.ham_integration = 0.0

        for i, t in enumerate(torch.linspace(0, t_max, t_max * self.n_step)):
            self._i = i

            _, field_to_stock, momentum_dt, residuals_dt = self.step()

            if self.flag_hamilt_integration:
                self.ham_integration += self.ham_value / self.n_step

            if self.image.isnan().any() or self.residuals.isnan().any():
                raise OverflowError(
                    "Some nan where produced !e the integration diverged",
                    "changing the parameters is needed (increasing n_step can help) ",
                )

            if self.save:
                self.image_stock[i] = self.image[0].detach().to("cpu")
                self.field_stock[i] = field_to_stock[0].detach().to("cpu")
                self.momentum_stock[i] = momentum_dt[0].detach().to("cpu")
                self.residuals_stock[i] = residuals_dt.detach().to("cpu")
            # elif self.save is False and i == 0:
            #     self.momentum_ini = residuals_dt.detach().to('cpu')

            if verbose:
                update_progress(i / (t_max * self.n_step))
                if self.flag_hamilt_integration:
                    print(
                        "ham :",
                        self.ham_value.detach().cpu().item(),
                        self.norm_v_2.detach().cpu().item(),
                        self.norm_l2_on_z.detach().cpu().item(),
                    )

        # print(f"image max : {self.image.max()}")
        if plot > 0:
            self.plot(n_figs=plot)


class Simplex_sqrt_Shooting(Optimize_geodesicShooting):

    def __init__(
        self,
        source,
        target,
        integrator,
        cost_cst,
        hamiltonian_integration=False,
        **kwargs,
    ):
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
            "sigma_v": self.mp.sigma_v,
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
