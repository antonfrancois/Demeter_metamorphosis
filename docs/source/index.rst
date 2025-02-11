.. Demeter_metamorphosis documentation master file, created by
sphinx, last updated |today|


Welcome to Demeter_metamorphosis's documentation!
=================================================

**In short:**

- Efficient and flexible optimization of metamorphoses in images
- 2D and 3D images registration
- Free software: MIT license
- Based on Pytorch with GPU support

Contents:
-----------------------------------------------------------

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    installation
    auto_examples/index
    modules
    advanced_users

More about Demeter_metamorphosis
-----------------------------------------------------

Demeter_metamorphosis is a library intended to perform Metamorphosis on images.
It can perform efficient registration of images in 2D and 3D with or without topology differences.
We propose a flexible framework to perform Metamorphosis with different modeling
choices. Are provided:

- The classic LDDMM registration method for registering image having the same topologies as described by Trouvé et al. or Beg's 2005 paper.
- The classic Metamorphosis registration method for mixing diffeomorphic transport with intensities additions as described in Younes's book "Shape and Diffeomorphisms" (2019).
- A Weighted Metamorphosis registration method to control the intensity addition locally as described by me in `this paper <https://hal.science/hal-03971473>`_.
- A Constrained Metamorphosis registration method to control the intensity addition locally and guide the registration with a pre-computed vector field as described by me in `my thesis <https://u-paris.fr/theses/detail-dune-these/?id_these=5642>`_. (paper comming soon)
- and more to come...

The library is designed to be flexible and allow the user to define custom data cost functions, custom norms on the residual, custom vector fields, and custom geodesic integrators. Feel
free to contact me through github Issues if you have any questions or need help with the library.

Why the name Demeter ? Because in ancient Rome, Demeter was the goddess of agriculture.
Isn't it the perfect protection for or vectors fields ?

The Metamorphic framework [Holm et al., 2009; Trouvé and Younes, 2005; Younes, 2019]
can be seen as a relaxed version of `LDDMM <https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping>`_.
in which we add time-varying intensity variations
are added to the diffeomorphic flow, therefore allowing for topological changes. The
image evolution is not only modelled by deformation, we allow adding intensity at each
time for every voxel, making topological changes possible. Metamorphosis is solved through
an Hamiltonian formulation and the momentum control both the deformation and the intensity changes.





More
====
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
