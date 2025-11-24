from typing import List, Tuple

from numpy import ndarray
from torch import Tensor, eye, get_default_dtype

import domains as do
from dataclasses import dataclass

def fill_image(
        continuous_support : List[Tuple],  # make [(-1,1)]* dim default
        n_pixels : Tuple,
        val : Tensor,
        device : str = 'cpu'
        ) -> "Image":
    """
    Build an Image from raw values by defining a continuous domain and its discrete grid.

    Parameters
    ----------
    continuous_support : list[tuple]
        Bounds of the continuous domain, one tuple (min, max) per dimension.
    n_pixels : tuple
        Shape of the target discrete grid; must match the number of dimensions in `continuous_support`.
    val : Tensor or ndarray
        Raw values to wrap into a Field.
    device : str, optional
        Torch device for creating the manifold metric, default "cpu".

    Returns
    -------
    Image
        Container holding the field on the discrete grid and its associated manifolds.

    Notes
    -----
    Discretization is skipped when the target grid already matches the field or manifold domain.
    """
    # M : do.RiemannianManifold, grid : do.Domain, field: do.Field
    if len(continuous_support) == len(n_pixels):
        d = len(continuous_support)
    else:
        raise ValueError(f"len(continuous_support) != len(n_pixels) : continuous_support={continuous_support} and n_pixels={n_pixels}")
    Omega = do.Domain(type="continuous", dim=2, support=continuous_support)
    M = do.RiemannianManifold(
        domain=Omega,
        metric=eye(2,device=device,dtype=get_default_dtype())
    )

    if isinstance(val, Tensor):
        field = do.torch_to_field(val)
    elif isinstance(val, ndarray):
        field = do.numpy_to_field(val)
    else:
        raise ValueError(f"type(val) should be Tensor or ndarray, got {type(val)}")
    # Optional if grid == field.domain
    grid = do.Domain(type="discrete", dim=d, support= n_pixels)
    if grid == field.domain:
        phi = None
        image_val = field
    else:
        phi = do.Discretize(source=field.domain, target=grid)
        image_val = phi.pf(field)

    if grid == M.domain:
        phig = None
        M_discrete = M
    else:
        phig = do.Discretize(source=M.domain, target=grid)
        M_discrete = phig.pf(M)


    img = Image(
        field = image_val,
        continuous_manifold=M,
        discretize=phig,
        discrete_manifold=M_discrete
    )

    return img



@dataclass
class Image:
    """
    Lightweight container tying together a field, its continuous manifold, and the discretization
    used to obtain the corresponding discrete manifold.
    """
    field : do.Field # considered as a discrete field
    continuous_manifold : do.RiemannianManifold
    discretize : do.Discretize | None
    discrete_manifold : do.RiemannianManifold | None = None

    def __post_init__(self):
        if self.discrete_manifold is None and self.discretize is not None:
            self.discrete_manifold = self.discretize.pf(self.continuous_manifold)
        if self.discrete_manifold is None:
            raise ValueError("Image needs either a discretize operator or a precomputed discrete_manifold")

    def to_plt(self):
        return self.field.to_plt()