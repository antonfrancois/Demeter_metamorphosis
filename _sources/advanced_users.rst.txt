For advanced users
================================================

Welcome, if you are reading this, you have probably already tested the basics
features of **Demeter**. This document is intended for advanced users who
want to contribute or design a new metamorphosis model based on the **Demeter** library.
Do not hesitate to post an issue on the github repository if you have any questions.
If you are confused about something, other might be too !



Theory and Implementation choices in Metamorphoses in Demeter
****************************************************************************

Minimal theory to understand before using Metamorphoses in Demeter
-----------------------------------------------------------------------------------------------

Our implementation of the metamorphoses algorithm is based on a geodesic
shooting method following a Hamiltonian trajectory. Broadly speaking, before
delving into the details, and taking LDDMM as an example, this means that the
algorithm is divided into two fundamental components: an `integrator` and an `optimizer`.

The `integrator` computes a "registered" or transported image via a metamorphic
flow starting from an initial condition, referred to as the momentum, denoted $p$
(and $p_0$ at the initial time). In other words, the integrator performs the following action:

.. math::

    I_1 = \mathrm{Integrator}(p_0)

The `optimizer`, on the other hand, is tasked with calculating $p_0$ to produce
the final image $I_1$ that is as close as possible to the target image. This is a
standard optimization problem, which is solved by minimizing a cost function of the form:

.. math::

    E(p_{0}) = \mathrm{DataCost}(\mathrm{Integrator}(p_{0}),T) + \lambda\mathrm{Reg}(p_{0})

where DataCost $\mathrm{DataCost}$ is a data fidelity term, $T$ is the target
image, $\mathrm{Reg}$ is a regularization term, and $\lambda$ is a constant.

The `integrator` and `optimizer` are two separate classes in the code. Integrators
inherit from `Geodesic_integrator` and optimizers inherit from
`Optimize_geodesicShooting`. As you might have guessed, the `integrator` and
`optimizer` are tightly coupled. The `optimizer` uses the `integrator` to compute
the registered image, and therefore the `integrator` is a required input and attribute
to the `optimizer`. In the `Optimize_geodesicShooting` class one can access
the `integrator` by calling the attribute `Optimize_geodesicShooting.mp`, mp
standing for metamorphosis path. This choice was made for conciseness.

More precise explanations : The classic Metamorphic case
-----------------------------------------------------------------------------------------------

As you might know, the LDDMM algorithm can be seen as a particular case of the
metamorphoses one. The implementation of Metamorphoses in **Demeter** is based on the minimization of a Hamiltonian:

.. math::

    H(q,p,v,z) =  (p|\dot q) - R(v,z)

where :math:`q : (\Omega, [0,1]) \mapsto \mathcal M` is the temporal image valued in :math:`\mathcal M`, :math:`R` is a regularization function, :math:`v` is a vector field, and :math:`z` is a control on the photometric part.

In the case of LDDMM and considering :math:`\mathcal M = \mathbb R`, the Hamiltonian is:

.. math::

    H(q,p,v,z) =  (p| \dot q) - \frac 12\|v\|_V^2 -\frac 12\|z\|_Z^2

An optimal trajectory or geodesic under the conditions given by :math:`H` is:

.. math::

    \left\{\begin{array}{rl} \dot q_t &= - \nabla q_t \cdot v_t + z_t\\ \dot z_t &= - \mathrm{div}(z_t  v_t) \\
    p_t &= z_t\\
    v_t &= -K_V\left( z_t\nabla q_t \right)  \end{array}\right.

These equations are written in the continuous case. In this document, all discretisation choices made during the implementation are detailed.

To solve the registration problem, a geodesic shooting strategy is used. For this, a relaxed version of :math:`H` is minimized:

.. math::

    E(p_0) = D_T(I_1) + \frac \lambda2 \left( \|v_0\|_V^2 +\|z_0\|_Z^2  \right)

Where :math:`D_T` is a data attachment term and :math:`T` is a target image, :math:`I_1` is the image at the end of the geodesic integration, and :math:`p_0` is the initial momentum. Note that in the case of Metamorphoses valued in images, :math:`p = z`.

You may have noticed that in the above equation :math:`E(p_{0})` depends only on the initial momentum. Indeed, thanks to a conservation property of norms during the calculation of optimal trajectories in a Hamiltonian which states: Let :math:`v` and :math:`z` follow a geodesic given by :math:`H`, then

.. math::

    \forall t \in [0,1], \|v_{0}\|^2_{V} = \|v_{t}\|^2_{V}; \|z_{0}\|^2_{2} = \|z_{t}\|^2_{2}.

This property is used to save computation time. In practice, due to numerical scheme choices, norm conservation may not be achieved. In this case, it is possible to optimize over the set of norms and :math:`E` becomes:

.. math::

    E(p_0) = D_T(I_1) + \frac \lambda2 \int_{0}^1 \left( \|v_t\|_V^2 +\|z_t\|_Z^2  \right) dt.

The :math:`I_{t},v_t,z_{t}` are still deduced from :math:`p_0`. It is possible to switch between the two in the code using the `hamiltonian_integration` option in the children of `Optimize_geodesicShooting`.


Contributing to Demeter:
-----------------------------------------------------------------------------------------------

Posting an issue:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have any questions, need help with the library or you have an incredible ideas that you want to share, do not hesitate to post an issue on the github repository. The issue can be about a bug, a feature request, or a question about the code. The more precise you are in your issue, the more likely you are to get a quick answer. Here are some guidelines to help you write a good issue:
 `Guides lines on Issues <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue>`_

Contributing to the code:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to contribute to the code, you can fork the repository and make a pull request. The pull request will be reviewed by the maintainers and merged if it is in line with the project's objectives. Here are some guidelines to help you write a good pull request:
 `Guides lines on Pull requests <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_

