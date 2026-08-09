"""Geodesic metamorphosis regression for timed 2D observations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .classic import Metamorphosis_Shooting, Metamorphosis_integrator
from .data_cost import SplineSsd
from .var_classes import Momenta
from ..utils import torchbox as tb
from ..utils.spline_data import validate_timed_observations


class _RegressionSsd(SplineSsd):
    """SSD over the differentiable observation images captured while shooting."""

    def __call__(self, at_step=None, **kwargs):
        if not hasattr(self, "optimizer"):
            raise AttributeError(
                "optimizer has not been initialized; call set_optimizer first"
            )
        images = torch.cat(self.optimizer._observation_images, dim=0)
        return 0.5 * (images - self.target).square().sum()


class MetamorphosisRegression(Metamorphosis_Shooting):
    r"""Fit one classical geodesic to images observed at fixed times.

    The objective is

    .. math::
        E(p_0) = \frac12\sum_i \|I(t_i)-J_i\|_2^2
          + \lambda E_{\mathrm{geo}}(p_0),

    where ``E_geo`` is exactly the regularization used by
    :class:`Metamorphosis_Shooting`. Observation times must lie on the uniform
    integration mesh.
    """

    def __init__(
        self,
        source: Tensor,
        target: Tensor,
        target_times: Sequence[float] | Tensor,
        geodesic: Metamorphosis_integrator,
        **kwargs,
    ) -> None:
        if not isinstance(geodesic, Metamorphosis_integrator):
            raise TypeError(
                "geodesic must be a Metamorphosis_integrator, "
                f"got {type(geodesic)}"
            )
        if geodesic.save_gpu_memory:
            raise NotImplementedError(
                "checkpointed geodesic regression is not implemented"
            )
        self.target_times, self.target_steps = validate_timed_observations(
            source, target, target_times, geodesic.n_step
        )
        self._target_step_set = frozenset(self.target_steps)
        source = source.detach()
        target = target.detach()
        super().__init__(
            source,
            target,
            geodesic,
            data_term=_RegressionSsd(target, self.target_steps),
            **kwargs,
        )

    def _forward_and_data_loss(self, momentum_ini: Momenta) -> Tensor:
        self._observation_images = []

        def capture_observation(completed_steps: int, _total_steps: int) -> None:
            if completed_steps in self._target_step_set:
                self._observation_images.append(self.mp.image)

        try:
            self.mp.forward(
                self.source,
                momentum_ini,
                save=False,
                plot=0,
                hamiltonian_integration=self.flag_hamiltonian_integration,
                progress_callback=capture_observation,
            )
            return self.data_term()
        finally:
            del self._observation_images

    @property
    def observation_images(self) -> Tensor:
        """Return fitted images at observation times after a saved shooting."""
        if not hasattr(self.mp, "image_stock"):
            raise ValueError("no saved shooting is available")
        indices = [step - 1 for step in self.target_steps]
        return self.mp.image_stock[indices]

    def get_ssd_def(self) -> float:
        return float(0.5 * (self.observation_images - self.target).square().sum())

    def plot_imgCmp(self, origin="lower", cmp_method="compose"):
        """Compare each fitted observation image with its matching target."""
        fitted = self.observation_images.detach().cpu()
        target = self.target.detach().cpu()
        figure, axes = plt.subplots(
            len(self.target_times),
            3,
            figsize=(12, 4 * len(self.target_times)),
            squeeze=False,
            constrained_layout=True,
        )
        for row, time in enumerate(self.target_times):
            axes[row, 0].imshow(target[row, 0], cmap="gray", origin=origin)
            axes[row, 0].set_title(f"target at t={time:g}")
            axes[row, 1].imshow(fitted[row, 0], cmap="gray", origin=origin)
            axes[row, 1].set_title("fitted geodesic")
            comparison = tb.imCmp(
                target[row:row + 1],
                fitted[row:row + 1],
                method=cmp_method,
            )[0]
            axes[row, 2].imshow(comparison, origin=origin)
            axes[row, 2].set_title("comparison")
            for axis in axes[row]:
                axis.set_axis_off()
        return figure, axes

    def plot_deform(self, temporal_nfigs=0):
        raise NotImplementedError(
            "endpoint deformation plotting is not defined for timed regression"
        )

    def get_all_arguments(self) -> dict:
        return {
            **super().get_all_arguments(),
            "target_times": self.target_times,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_history_size": self.lbfgs_history_size,
            "adam_scheduler": self._adam_scheduler_type,
            "adam_grad_clip": self._adam_grad_clip,
        }
