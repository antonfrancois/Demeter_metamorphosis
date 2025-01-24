import pytest
import torch
import matplotlib.pyplot as plt
import napari
import src.demeter.utils.torchbox as tb


plot = False

# @pytest.fixture
# def args():
#     return argparse.Namespace(plot=False)


# =============================================================================
#         TestLinearX_pixel
# =============================================================================
@pytest.fixture
def setup_linear_x_pixel():
    H, W = 10, 15
    yy, xx = torch.meshgrid(
        torch.arange(0, H, dtype=torch.long),
        torch.arange(0, W, dtype=torch.long),
        indexing="ij",
    )
    image = xx.clone()[None, None].to(torch.float64)
    derivative = tb.spatialGradient(image, dx_convention="pixel")

    return H, W, image, derivative


def test_plot_linear_x_pixel(setup_linear_x_pixel):
    if plot:
        H, W, image, derivative = setup_linear_x_pixel
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(image[0, 0], cmap="gray")
        ax[1, 0].imshow(derivative[0, 0, 0], cmap="gray")
        ax[2, 0].imshow(derivative[0, 0, 1], cmap="gray")

        ax[0, 1].plot(image[0, 0, H // 2])
        ax[1, 1].plot(derivative[0, 0, 0, :, W // 2])
        ax[2, 1].plot(derivative[0, 0, 1, H // 2])
        fig.suptitle(r"spatial convention on pixelic $[0,H-1]\times [0,W-1]$")
        plt.show()


def test_derivative_constant_direction_x_pixel(setup_linear_x_pixel):
    H, W, image, derivative = setup_linear_x_pixel
    assert derivative[0, 0, 0, :, W // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_x f(x,y) = x was not constant"


def test_derivative_constant_direction_y_pixel(setup_linear_x_pixel):
    H, W, image, derivative = setup_linear_x_pixel
    assert derivative[0, 0, 1, H // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_y f(x,y) = x was not constant"


def test_value_of_derivative_direction_x_pixel(setup_linear_x_pixel):
    H, W, image, derivative = setup_linear_x_pixel
    eps = 1e-7
    assert (
        derivative[0, 0, 0, H // 2, W // 2] - 1
    ).abs() < eps, f"The derivative of D_x f(x,y) = x should be 1 but was {derivative[0, 0, 0, H // 2, W // 2]}"


def test_value_of_derivative_direction_y_pixel(setup_linear_x_pixel):
    H, W, image, derivative = setup_linear_x_pixel
    assert (
        derivative[0, 0, 1, H // 2, W // 2] == 0
    ), f"The derivative of D_y f(x,y) = x should be 0 but was {derivative[0, 0, 1, H // 2, W // 2]}"


# =============================================================================
#         TestLinearX_square
# =============================================================================


@pytest.fixture
def setup_linear_x_1square():
    H, W = 10, 15
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    image = xx.clone()[None, None].to(torch.float64)
    derivative = tb.spatialGradient(image, dx_convention="square")
    return H, W, image, derivative


def test_derivative_constant_direction_x(setup_linear_x_1square):
    H, W, image, derivative = setup_linear_x_1square
    assert derivative[0, 0, 0, :, W // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_x f(x,y) = x was not constant"


def test_derivative_constant_direction_y(setup_linear_x_1square):
    H, W, image, derivative = setup_linear_x_1square
    assert derivative[0, 0, 1, H // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_y f(x,y) = x was not constant"


def test_value_of_derivative_direction_x(setup_linear_x_1square):
    H, W, image, derivative = setup_linear_x_1square
    eps = 1e-6
    assert (
        derivative[0, 0, 0, H // 2, W // 2] - 1
    ).abs() < eps, f"The derivative of D_x f(x,y)= x should be 1 but was {derivative[0, 0, 0, H // 2, W // 2]}"


def test_value_of_derivative_direction_y(setup_linear_x_1square):
    H, W, image, derivative = setup_linear_x_1square
    assert (
        derivative[0, 0, 1, H // 2, W // 2] == 0
    ), f"The derivative of D_y f(x,y) = x should be 0 but was {derivative[0, 0, 1, H // 2, W // 2]}"


# =============================================================================
#         TestLinearX_2square
# =============================================================================


@pytest.fixture
def setup_linear_x_2square():
    H, W = 10, 15
    yy, xx = torch.meshgrid(
        torch.linspace(0, 2, H), torch.linspace(0, 2, W), indexing="ij"
    )
    image = xx.clone()[None, None].to(torch.float64)
    derivative = tb.spatialGradient(image, dx_convention="2square")
    return H, W, image, derivative


def test_derivative_constant_direction_x_2square(setup_linear_x_2square):
    H, W, image, derivative = setup_linear_x_2square
    assert derivative[0, 0, 0, :, W // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_x f(x,y) = x was not constant"


def test_derivative_constant_direction_y_2square(setup_linear_x_2square):
    H, W, image, derivative = setup_linear_x_2square
    assert derivative[0, 0, 1, H // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_y f(x,y) = x was not constant"


def test_value_of_derivative_direction_x_2square(setup_linear_x_2square):
    H, W, image, derivative = setup_linear_x_2square
    eps = 1e-6
    assert (
        derivative[0, 0, 0, H // 2, W // 2] - 1
    ).abs() < eps, f"The derivative of D_x f(x,y)= x should be 1 but was {derivative[0, 0, 0, H // 2, W // 2]}"


def test_value_of_derivative_direction_y_2square(setup_linear_x_2square):
    H, W, image, derivative = setup_linear_x_2square
    assert (
        derivative[0, 0, 1, H // 2, W // 2] == 0
    ), f"The derivative of D_y f(x,y) = x should be 0 but was {derivative[0, 0, 1, H // 2, W // 2]}"


# =============================================================================
#         TestLinearY_pixel
# =============================================================================


@pytest.fixture
def setup_linear_y_pixel():
    H, W = 10, 15
    yy, xx = torch.meshgrid(
        torch.arange(0, H, dtype=torch.long),
        torch.arange(0, W, dtype=torch.long),
        indexing="ij",
    )
    image = yy.clone()[None, None].to(torch.float64)
    derivative = tb.spatialGradient(image, dx_convention="pixel")

    return H, W, image, derivative


def plot_linear_y_pixel(setup_linear_y_pixel):
    if plot:
        H, W, image, derivative = setup_linear_y_pixel

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(image[0, 0], cmap="gray")
        ax[1, 0].imshow(derivative[0, 0, 0], cmap="gray")
        ax[2, 0].imshow(derivative[0, 0, 1], cmap="gray")

        ax[0, 1].plot(image[0, 0, H // 2])
        ax[1, 1].plot(derivative[0, 0, 0, :, W // 2])
        ax[2, 1].plot(derivative[0, 0, 1, H // 2])
        fig.suptitle(r"spatial convention on pixelic $[0,H-1]\times [0,W-1]$")

        plt.show()


def test_derivative_constant_direction_x_pixel(setup_linear_y_pixel):
    H, W, image, derivative = setup_linear_y_pixel
    assert derivative[0, 0, 0, :, W // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_x f(x,y) = y was not constant"


def test_derivative_constant_direction_y_pixel(setup_linear_y_pixel):
    H, W, image, derivative = setup_linear_y_pixel
    assert derivative[0, 0, 1, H // 2].unique().shape == torch.Size(
        [1]
    ), "The derivative of D_y f(x,y) = y was not constant"


def test_value_of_derivative_direction_x_pixel(setup_linear_y_pixel):
    H, W, image, derivative = setup_linear_y_pixel
    assert (
        derivative[0, 0, 0, H // 2, W // 2] == 0
    ), f"The derivative of D_x f(x,y) = y should be 1 but was {derivative[0, 0, 0, H // 2, W // 2]}"


def test_value_of_derivative_direction_y_pixel(setup_linear_y_pixel):
    H, W, image, derivative = setup_linear_y_pixel
    eps = 1e-7
    assert (
        derivative[0, 0, 1, H // 2, W // 2] - 1
    ).abs() < eps, f"The derivative of D_y f(x,y) = y should be 0 but was {derivative[0, 0, 1, H // 2, W // 2]}"


# =============================================================================
#     TestCustom2D_pixel
# =============================================================================


@pytest.fixture
def setup_custom_2d_pixel():
    H, W = 10, 15
    yy, xx = torch.meshgrid(
        torch.arange(0, H, dtype=torch.long),
        torch.arange(0, W, dtype=torch.long),
        indexing="ij",
    )
    image = yy**2 + 2 * xx * yy
    image = image[None, None].to(torch.float64)
    theoretical_derivative_x = 2 * yy
    theoretical_derivative_y = 2 * yy + 2 * xx
    derivative = tb.spatialGradient(image, dx_convention="pixel")

    return H, W, image, derivative, theoretical_derivative_x, theoretical_derivative_y


def test_plot_custom_2d_pixel(setup_custom_2d_pixel):
    if plot:
        H, W, image, derivative, theoretical_derivative_x, theoretical_derivative_y = (
            setup_custom_2d_pixel
        )
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(image[0, 0], cmap="gray")
        ax[0, 0].set_title("f(x,y) = yy**2 + 2*xx*yy")
        ax[1, 0].imshow(derivative[0, 0, 0], cmap="gray")
        ax[1, 0].set_title("Spatial gradient x")
        ax[2, 0].imshow(derivative[0, 0, 1], cmap="gray")
        ax[2, 0].set_title("Spatial gradient y")

        ax[0, 1].plot(image[0, 0, :, W // 2])
        ax[1, 1].imshow(theoretical_derivative_x, cmap="gray")
        ax[1, 1].set_title("D_x f(x,y) = 2*yy")
        ax[2, 1].imshow(theoretical_derivative_y, cmap="gray")
        ax[2, 1].set_title("D_y f(x,y) = 2*yy + 2*xx")
        fig.suptitle(r"spatial convention on pixels $[0,H-1]\times [0,W-1]$")

        plt.show()


def test_values_of_derivative_direction_x(setup_custom_2d_pixel):
    H, W, image, derivative, theoretical_derivative_x, theoretical_derivative_y = (
        setup_custom_2d_pixel
    )
    eps = 1e-6
    assert (
        derivative[0, 0, 0, 1:-1, 1:-1] - theoretical_derivative_x[1:-1, 1:-1]
    ).abs().max() < eps, (
        "The derivative of D_x f(x,y) = yy**2 + 2*xx*yy should be "
        f"{theoretical_derivative_x} but was {derivative[0, 0, 0]}"
    )


def test_values_of_derivative_direction_y(setup_custom_2d_pixel):
    H, W, image, derivative, theoretical_derivative_x, theoretical_derivative_y = (
        setup_custom_2d_pixel
    )
    eps = 1e-6
    assert (
        derivative[0, 0, 1, 1:-1, 1:-1] - theoretical_derivative_y[1:-1, 1:-1]
    ).abs().max() < eps, (
        "The derivative of D_y f(x,y) = yy**2 + 2*xx*yy should be "
        f"{theoretical_derivative_y} but was {derivative[0, 0, 1]}"
    )


# =============================================================================
#        TestExp_custom
# =============================================================================


def test_dx_consistency_square_2d():
    rgi = tb.RandomGaussianImage(
        (300, 400),
        20,
        "square",
        # a=[-1, 1],
        # b=[15, 25],
        # c=[[.3*400 , .3*300], [.7*400, .7*300]]
    )
    image = rgi.image()
    theoretical_derivative = rgi.derivative()
    derivative = tb.spatialGradient(image, dx_convention="square")
    H, W = image.shape[-2:]
    # dx = torch.tensor([[1. / (H-1), 1. / (W-1)]], dtype=torch.float64)
    dx = torch.tensor([[1.0 / (W - 1), 1.0 / (H - 1)]], dtype=torch.float64)
    derivative_2 = tb.spatialGradient(image, dx_convention=dx)
    assert (derivative_2 - derivative).abs().max() < 1e-6, (
        "The derivative of D_x f(x,y) should be the same with dx = 1/(W-1), 1/(H-1)"
        f" and dx_convention = square; max difference is {(derivative_2 - derivative).abs().max()}"
    )


def test_dx_consistency_square_3d():
    rgi = tb.RandomGaussianImage((300, 400, 50), 20, "square")
    image = rgi.image()
    theoretical_derivative = rgi.derivative()
    derivative = tb.spatialGradient(image, dx_convention="square")
    D, H, W = image.shape[-3:]
    dx = torch.tensor(
        [[1.0 / (W - 1), 1.0 / (H - 1), 1.0 / (D - 1)]], dtype=torch.float64
    )
    derivative_2 = tb.spatialGradient(image, dx_convention=dx)
    print("3d der2 - der", (derivative_2 - derivative).abs().max())
    assert (derivative_2 - derivative).abs().max() < 1e-5, (
        "The derivative of D_x f(x,y) should be the same with dx = (1/(D-1), 1/(W-1), 1/(H-1)) and dx_convention = square; "
        f"max difference is {(derivative_2 - derivative).abs().max()}"
    )


def test_dx_consistency_pixel():
    rgi = tb.RandomGaussianImage(
        (300, 400),
        2,
        "pixel",
        a=[-1, 1],
        b=[15, 25],
        c=[[0.3 * 400, 0.3 * 300], [0.7 * 400, 0.7 * 300]],
    )
    image = rgi.image()
    theoretical_derivative = rgi.derivative()
    print(f"image shape : {image.shape}")
    derivative = tb.spatialGradient(image, dx_convention="pixel")
    dx = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    derivative_2 = tb.spatialGradient(image, dx_convention=dx)
    assert (
        derivative_2 - derivative
    ).abs().max() < 1e-6, "The derivative of D_x f(x,y) should be the same with dx = 1, 1 and dx_convention = pixel"


@pytest.fixture
def setup_exp2d(request):
    convention = request.param
    rgi = tb.RandomGaussianImage(
        (300, 400),
        2,
        convention,
        a=[-1, 1] if convention == "pixel" else [-1, 1],
        b=[15, 25] if convention == "pixel" else [0.1, 0.1],
        c=(
            [[0.3 * 400, 0.3 * 300], [0.7 * 400, 0.7 * 300]]
            if convention == "pixel"
            else [[0.5, 0.3], [0.5, 0.7]]
        ),
    )
    image = rgi.image()
    theoretical_derivative = rgi.derivative()
    derivative = tb.spatialGradient(image, dx_convention=convention)
    score = (
        derivative[..., 1:-1, 1:-1] - theoretical_derivative[..., 1:-1, 1:-1]
    ).abs()
    return image, theoretical_derivative, derivative, score


@pytest.mark.parametrize("setup_exp2d", ["pixel", "square"], indirect=True)
def test_values_of_derivative(setup_exp2d):
    image, theoretical_derivative, derivative, score = setup_exp2d
    eps = 1e-2
    assert (
        score.max() < eps
    ), f"The max difference between theoretical and finite difference derivative should be < {eps} but was {score.max()}"


@pytest.mark.parametrize("setup_exp2d", ["pixel", "square"], indirect=True)
def test_plot(setup_exp2d):
    if plot:
        image, theoretical_derivative, derivative, score = setup_exp2d
        fig, ax = plt.subplots(3, 2, constrained_layout=True)
        ax[0, 0].imshow(image[0, 0], origin="lower")
        ax[0, 0].set_title("Image")
        a = ax[1, 0].imshow(theoretical_derivative[0, 0], origin="lower")
        plt.colorbar(a, ax=ax[1, 0], fraction=0.046, pad=0.04)
        ax[1, 0].set_title("theoretical derivative_x")
        b = ax[2, 0].imshow(theoretical_derivative[0, 1], origin="lower")
        plt.colorbar(b, ax=ax[2, 0], fraction=0.046, pad=0.04)
        ax[2, 0].set_title("theoretical derivative_y")

        c = ax[1, 1].imshow(derivative[0, 0, 0], origin="lower")
        plt.colorbar(c, ax=ax[1, 1], fraction=0.046, pad=0.04)
        ax[1, 1].set_title("sobel derivative_x")
        d = ax[2, 1].imshow(derivative[0, 0, 1], origin="lower")
        plt.colorbar(d, ax=ax[2, 1], fraction=0.046, pad=0.04)
        ax[2, 1].set_title("sobel derivative_y")
        plt.show()


## 3D =============================================================================


@pytest.fixture
def setup_exp3d(request):
    convention = request.param
    rgi = tb.RandomGaussianImage((100, 80, 50), 10, convention)
    image = rgi.image()
    theoretical_derivative = rgi.derivative()
    derivative = tb.spatialGradient(image, dx_convention=convention)
    score = (
        derivative[..., 1:-1, 1:-1, 1:-1]
        - theoretical_derivative[..., 1:-1, 1:-1, 1:-1]
    ).abs()
    return image, theoretical_derivative, derivative, score


@pytest.mark.parametrize("setup_exp3d", ["square", "pixel", "2square"], indirect=True)
def test_values_of_derivative(setup_exp3d):
    image, theoretical_derivative, derivative, score = setup_exp3d
    eps = 0.1
    assert (
        score.mean() + score.std() < eps
    ), f"The max difference between theoretical and finite difference derivative should have mean + std < {eps} but was: max {score.max()} mean {score.mean()} and std {score.std()}"


@pytest.mark.parametrize("setup_exp3d", ["square", "pixel", "2square"], indirect=True)
def test_plot(setup_exp3d):
    if plot:
        image, theoretical_derivative, derivative, score = setup_exp3d
        print(
            f"TestExp3d_{setup_exp3d}, the difference between theoretical and finite difference derivative :"
        )
        print(
            f"\tscore : max {score.max()};\n\t min {score.min()};\n\t mean {score.mean()};\n\t std {score.std()}\n\t"
        )
        nv = napari.Viewer()
        nv.add_image(image[0, 0], name="image")
        nv.add_image(theoretical_derivative[0, 0], name="X theoretical_derivative")
        nv.add_image(theoretical_derivative[0, 1], name="Y theoretical_derivative")
        nv.add_image(theoretical_derivative[0, 2], name="Z theoretical_derivative")
        nv.add_image(derivative[0, 0, 0], name="X derivative")
        nv.add_image(derivative[0, 0, 1], name="Y derivative")
        nv.add_image(derivative[0, 0, 2], name="Z derivative")
        nv.add_image(score[0, 0, 0], name="X score")
        nv.add_image(score[0, 0, 1], name="Y score")
        nv.add_image(score[0, 0, 2], name="Z score")
        napari.run()


# =============================================================================
#     Test SpatialGradient shape
# =============================================================================

shape_list = [
    (1, 4, 100, 80, 50),
    (5, 1, 100, 80, 50),
    (2, 2, 100, 80, 50),
    (1, 4, 100, 80),
    (5, 1, 100, 80),
]
output_list = [
    (1, 4, 3, 100, 80, 50),
    (5, 1, 3, 100, 80, 50),
    "error",
    (1, 4, 2, 100, 80),
    (5, 1, 2, 100, 80),
]


@pytest.mark.parametrize("shape, output_shape", zip(shape_list, output_list))
def test_shape_spatialGradient(shape, output_shape):
    image = torch.randn(shape)
    if output_shape == "error":
        with pytest.raises(ValueError):
            derivative = tb.spatialGradient(image)
    else:
        derivative = tb.spatialGradient(image)
        assert (
            derivative.shape == output_shape
        ), f"expected shape {output_shape} but got {derivative.shape}"
