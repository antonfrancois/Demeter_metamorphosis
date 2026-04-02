import torch
from matplotlib import pyplot as plt

import demeter.utils.torchbox as tb


def _ensure_batched_param(x, B, length=None, device=None, dtype=None):
    """
    Convert parameter x to a batched tensor of shape [B, ...].

    Rules
    -----
    - scalar -> [B]
    - [k] -> [B, k] if B==1, else error
    - [B] or [B, k] kept as is
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)

    if x.ndim == 0:
        x = x.repeat(B)

    elif x.ndim == 1:
        if length is not None and x.numel() == length:
            if B == 1:
                x = x.unsqueeze(0)  # [1, length]
            elif x.numel() == B:
                pass  # interpreted as [B]
            else:
                raise ValueError(
                    f"Ambiguous 1D parameter of shape {tuple(x.shape)} for B={B}, length={length}."
                )
        else:
            if x.numel() == 1:
                x = x.repeat(B)
            elif x.numel() == B:
                pass
            else:
                raise ValueError(f"Expected scalar or batch-sized tensor, got shape {tuple(x.shape)}.")

    elif x.ndim == 2:
        if x.shape[0] != B:
            raise ValueError(f"Expected first dim = batch size {B}, got {tuple(x.shape)}.")
    else:
        raise ValueError(f"Unsupported parameter shape {tuple(x.shape)}.")

    return x


def rotation_matrix_2d(theta, device=None, dtype=None):
    """
    Batched 2D rotation matrices.

    Parameters
    ----------
    theta : scalar or tensor [B]
        Rotation angle in radians.

    Returns
    -------
    R : tensor [B, 2, 2]
    """
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=device, dtype=dtype)
    else:
        theta = theta.to(device=device, dtype=dtype)

    if theta.ndim == 0:
        theta = theta[None]

    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.zeros((theta.shape[0], 2, 2), device=theta.device, dtype=theta.dtype)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


def homogeneous_from_linear_translation(A, t):
    """
    Build homogeneous matrices from linear part and translation.

    Parameters
    ----------
    A : tensor [B, 2, 2]
    t : tensor [B, 2]

    Returns
    -------
    H : tensor [B, 3, 3]
    """
    B = A.shape[0]
    H = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(B, 1, 1)
    H[:, :2, :2] = A
    H[:, :2, 2] = t
    return H


def homogeneous_translation(t):
    """
    Translation homogeneous matrices.

    Parameters
    ----------
    t : tensor [B, 2]

    Returns
    -------
    T : tensor [B, 3, 3]
    """
    B = t.shape[0]
    T = torch.eye(3, device=t.device, dtype=t.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :2, 2] = t
    return T


def compose_homogeneous(H_left, H_right):
    """
    Batched composition H_left @ H_right.
    """
    return H_left @ H_right


def apply_homogeneous_to_grid(grid, H):
    """
    Apply a batched homogeneous transform to a batched grid.

    Parameters
    ----------
    grid : tensor [B,H,W,2]
        Grid in 2square convention.
    H : tensor [B,3,3]
        Forward transform in the same coordinate system.

    Returns
    -------
    out : tensor [B,H,W,2]
    """
    B, Hh, Wh, _ = grid.shape
    ones = torch.ones((B, Hh, Wh, 1), device=grid.device, dtype=grid.dtype)
    grid_h = torch.cat([grid, ones], dim=-1)  # [B,H,W,3]
    out_h = torch.einsum('bij,bhwj->bhwi', H, grid_h)
    return out_h[..., :2]


def center_translation_2square(img_shape, device=None, dtype=None):
    """
    Return the center of the image in tb's '2square' convention.

    Since tb.make_regular_grid(..., dx_convention='2square') spans [-1,1],
    the geometric center is simply (0,0).
    """
    return torch.zeros((1, 2), device=device, dtype=dtype)


def make_transform_matrices(
    img,
    rotation=None,
    translation=None,
    scale=None,
    full_affine=None,
):
    """
    Construct the four linear model matrices discussed.

    Parameters
    ----------
    img : tensor [B,C,H,W]
    rotation : scalar or tensor [B], optional
        Angle in radians.
    translation : tuple/list/tensor of shape [2] or [B,2], optional
        Translation in 2square coordinates.
    scale : scalar or tensor [B], optional
        Isotropic scale.
    full_affine : tensor [2,2] or [B,2,2], optional
        Arbitrary affine linear part.

    Returns
    -------
    mats : dict[str, tensor]
        Dictionary with keys:
        - 'full_affine'
        - 'rotation_translation_scaling'
        - 'rotation_translation'
        - 'rotation_scaling'

        Each value is a tensor [B,3,3].
    """
    if img.ndim != 4:
        raise ValueError(f"Expected img of shape [B,C,H,W], got {tuple(img.shape)}.")

    B, _, _, _ = img.shape
    device = img.device
    dtype = img.dtype

    if rotation is None:
        rotation = torch.zeros(B, device=device, dtype=dtype)
    rotation = _ensure_batched_param(rotation, B, device=device, dtype=dtype)

    if translation is None:
        translation = torch.zeros((B, 2), device=device, dtype=dtype)
    translation = _ensure_batched_param(translation, B, length=2, device=device, dtype=dtype)
    if translation.ndim != 2 or translation.shape[1] != 2:
        raise ValueError(f"translation must have shape [B,2], got {tuple(translation.shape)}.")

    if scale is None:
        scale = torch.ones(B, device=device, dtype=dtype)
    scale = _ensure_batched_param(scale, B, device=device, dtype=dtype)

    R = rotation_matrix_2d(rotation, device=device, dtype=dtype)  # [B,2,2]
    S = torch.zeros((B, 2, 2), device=device, dtype=dtype)
    S[:, 0, 0] = scale
    S[:, 1, 1] = scale

    A_rts = S @ R
    A_rt = R
    A_rs = S @ R

    if full_affine is None:
        A_full = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    else:
        if not torch.is_tensor(full_affine):
            full_affine = torch.tensor(full_affine, device=device, dtype=dtype)
        else:
            full_affine = full_affine.to(device=device, dtype=dtype)

        if full_affine.ndim == 2:
            A_full = full_affine.unsqueeze(0).repeat(B, 1, 1)
        elif full_affine.ndim == 3 and full_affine.shape[0] == B:
            A_full = full_affine
        else:
            raise ValueError(
                f"full_affine must be [2,2] or [B,2,2], got {tuple(full_affine.shape)}."
            )

    # In 2square coordinates the image center is 0, so "about center"
    # is equivalent to acting directly. I keep the homogeneous structure
    # explicit for extensibility.
    c = center_translation_2square(img.shape, device=device, dtype=dtype).repeat(B, 1)
    T_plus_c = homogeneous_translation(c)
    T_minus_c = homogeneous_translation(-c)

    H_full = compose_homogeneous(
        T_plus_c,
        compose_homogeneous(
            homogeneous_from_linear_translation(A_full, translation),
            T_minus_c,
        ),
    )

    H_rts = compose_homogeneous(
        T_plus_c,
        compose_homogeneous(
            homogeneous_from_linear_translation(A_rts, translation),
            T_minus_c,
        ),
    )

    H_rt = compose_homogeneous(
        T_plus_c,
        compose_homogeneous(
            homogeneous_from_linear_translation(A_rt, translation),
            T_minus_c,
        ),
    )

    # no translation for rotation+scaling only
    H_rs = compose_homogeneous(
        T_plus_c,
        compose_homogeneous(
            homogeneous_from_linear_translation(A_rs, torch.zeros_like(translation)),
            T_minus_c,
        ),
    )

    return {
        "full_affine": H_full,
        "rotation_translation_scaling": H_rts,
        "rotation_translation": H_rt,
        "rotation_scaling": H_rs,
    }


def apply_registration_models(
    img,
    rotation=None,
    translation=None,
    scale=None,
    full_affine=None,
    mode="bilinear",
    clamp=False,
):
    """
    Apply all discussed linear models to a [B,C,H,W] image.

    Parameters
    ----------
    img : tensor [B,C,H,W]
    rotation : scalar or tensor [B], optional
        Rotation angle in radians.
    translation : tuple/list/tensor [2] or [B,2], optional
        Translation in 2square coordinates.
    scale : scalar or tensor [B], optional
        Isotropic scale.
    full_affine : tensor [2,2] or [B,2,2], optional
        Arbitrary affine linear part.
    mode : str
        Interpolation mode passed to tb.imgDeform.
    clamp : bool
        Passed to tb.imgDeform.

    Returns
    -------
    out : dict[str, dict]
        For each model:
        {
            "matrix": [B,3,3],
            "grid":   [B,H,W,2],
            "image":  [B,C,H,W],
        }
    """
    if img.ndim != 4:
        raise ValueError(f"Expected img of shape [B,C,H,W], got {tuple(img.shape)}.")

    mats = make_transform_matrices(
        img=img,
        rotation=rotation,
        translation=translation,
        scale=scale,
        full_affine=full_affine,
    )

    B, _, H, W = img.shape
    id_grid = tb.make_regular_grid((B, H, W, 2), dx_convention="2square").to(img.device).to(img.dtype)

    out = {}
    for name, Hmat in mats.items():
        grid = apply_homogeneous_to_grid(id_grid, Hmat)
        deform = apply_homogeneous_to_grid(id_grid, torch.linalg.inv(Hmat))
        warped = tb.imgDeform(img, grid, dx_convention="2square", clamp=clamp, mode=mode)
        out[name] = {
            "matrix": Hmat,
            "grid": grid,
            "deform": deform,
            "image": warped,
        }

    return out

def show_deforms(img, res, keys ):
    fig, ax = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)

    for c, k in enumerate(keys):
        img = res[k]["image"]
        grid_t = res[k]["grid"]
        grid_inv = res[k]["deform"]
        H,W = img.shape[2:]

        # Row 1: transformed image
        ax[0, c].imshow(img[0,0], cmap="gray")
        ax[0, c].set_title(k)
        ax[0, c].axis("off")

        # Row 2: comparison with target (absolute difference)
        diff = tb.imCmp(target, img, 'seg')
        ax[1, c].imshow(diff[0], cmap="magma")
        ax[1, c].axis("off")

        # Row 3: deformation grid
        tb.gridDef_plot_2d(grid_inv, ax = ax[1, c],
                           step = 20,
                           color="white",
                           alpha =.3,
                           check_diffeo=False,
                           dx_convention='2square'
                           )

        ax[1, c].set_xlim(0, W - 1)
        ax[1, c].set_ylim(H - 1, 0)
        ax[1, c].set_aspect("equal", adjustable="box")
        ax[1, c].axis("off")

    ax[0, 0].set_ylabel("Image")
    ax[1, 0].set_ylabel("Image vs Target")
    # ax[2, 0].set_ylabel("Deformation grid")

    plt.show()