import pytest
import src.demeter.utils.torchbox as tb
import torch
#%% Test regular grid



@pytest.mark.parametrize("name,input,output", [
    ["3d_grid", (1,5,7,9,3), (1,5,7,9,3)],
    ["3d_ltl", (5,7,9), (1,5,7,9,3)],
    ["2d_grid", (1,4,6,2), (1,4,6,2)],
    ["3d_grid", (4,6), (1,4,6,2)]
])
def test_size(name, input, output):
    mesh = tb.make_regular_grid(input)
    assert mesh.shape == output

@pytest.mark.parametrize("name,size,dx,mini", [
    ["pixel_2d", (1,4,6,2), 'pixel', 0],
    ["pixel_3d", (1,5,7,9,3), 'pixel', 0],
    ["2square_2d", (1,4,6,2), '2square', -1],
    ["2square_2d", (1,4,6,2), '2square', -1],
    ["square", (1,4,6,2), 'square', 0],
    ["square", (1,5,7,9,3), 'square', 0]
])
def test_min(name, size, dx, mini):
    mesh = tb.make_regular_grid(size, dx_convention=dx)
    assert mesh.min() == mini

@pytest.mark.parametrize("name,size,dx,maxi", [
    ["pixel_2d", (1,4,6,2), 'pixel', 5],
    ["pixel_3d", (1,5,7,9,3), 'pixel', 8],
    ["2square_2d", (1,4,6,2), '2square', 1],
    ["2square_2d", (1,4,6,2), '2square', 1],
    ["square", (1,4,6,2), 'square', 1],
    ["square", (1,5,7,9,3), 'square', 1]
])
def test_max(name, size, dx, maxi):
    mesh = tb.make_regular_grid(size, dx_convention=dx)
    assert mesh.max() == maxi

#%% pixel 2 square conventions:


def sample_field_2d_pixel():
    """Fixture for a sample 2D field. shape = (1, 4, 6, 2)"""
    return torch.tensor([[
        [[0., 0.],
          [0., 0.],
          [0., 0.],
          [1., 1.],
          [0., 0.],
          [1., 2.]],
         [[0., 0.],
          [2., 1.],
          [0., 0.],
          [0., 0.],
          [0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.],
          [2., 2.],
          [0., 0.],
          [1., 0.],
          [0., 0.]],
         [[0., 0.],
          [1., -1.],
          [0., 0.],
          [0., 1.],
          [0., 0.],
          [0., 0.]]
    ]])

def sample_field_2d_square():
    """Fixture for a sample 2D field. shape = (1, 4, 6, 2)"""
    field = sample_field_2d_pixel()
    _,H,W,_ = field.shape
    field[...,0] = field[...,0]/(W - 1)
    field[...,1] = field[...,1]/(H - 1)
    return field

def sample_field_2d_2square():
    """Fixture for a sample 2D field. shape = (1, 4, 6, 2)"""
    field = sample_field_2d_pixel()
    _,H,W,_ = field.shape
    field[...,0] = 2 * field[...,0]/(W - 1)
    field[...,1] = 2 * field[...,1]/(H - 1)
    return field

@pytest.mark.parametrize("name,size", [
    ["2d", (1,4,6,2)],
    ["3d", (1,5,7,9,3)],
])
def test_pixel_to_2square_convention_1(name, size):
    field = tb.make_regular_grid(size, dx_convention='pixel')
    field = tb.pixel_to_2square_convention(field)
    assert field.shape == size
    assert field.min() == -1
    assert field.max() == 1



def test_pixel_to_2square_convention_2d_field():
    field = sample_field_2d_pixel()
    field = tb.pixel_to_2square_convention(field, is_grid=False)
    ref_field = sample_field_2d_2square()
    print(field - ref_field)
    assert torch.all(field == ref_field)

@pytest.mark.parametrize("from_field,converter, to_field", [
    [sample_field_2d_pixel, tb.pixel_to_square_convention, sample_field_2d_square],
    [sample_field_2d_pixel, tb.pixel_to_2square_convention, sample_field_2d_2square],
    [sample_field_2d_square, tb.square_to_pixel_convention, sample_field_2d_pixel],
    [sample_field_2d_square, tb.square_to_2square_convention, sample_field_2d_2square],
    [sample_field_2d_2square, tb.square2_to_pixel_convention, sample_field_2d_pixel],
    [sample_field_2d_2square, tb.square2_to_square_convention, sample_field_2d_square],
])
def test_convert_convention_2d_field(from_field, converter, to_field):
    field = from_field()
    field = converter(field, is_grid=False)
    ref_field = to_field()
    print(from_field.__name__,converter.__name__, to_field.__name__)
    print(field - ref_field)
    assert torch.allclose(field,ref_field, atol=1e-6)

@pytest.mark.parametrize("from_field,converter, to_field", [
    [sample_field_2d_pixel, tb.pixel_to_square_convention, sample_field_2d_square],
    [sample_field_2d_pixel, tb.pixel_to_2square_convention, sample_field_2d_2square],
    [sample_field_2d_square, tb.square_to_pixel_convention, sample_field_2d_pixel],
    [sample_field_2d_square, tb.square_to_2square_convention, sample_field_2d_2square],
    [sample_field_2d_2square, tb.square2_to_pixel_convention, sample_field_2d_pixel],
    [sample_field_2d_2square, tb.square2_to_square_convention, sample_field_2d_square],
])
def test_convert_convention_2d_grid(from_field, converter, to_field):
    field = from_field()
    dx_con_from = from_field.__name__[16:]
    dx_con_to = to_field.__name__[16:]

    grid = tb.make_regular_grid(field.shape, dx_convention=dx_con_from)
    n_grid = converter(grid + field, is_grid=True)
    ref_field = to_field()
    ref_grid = tb.make_regular_grid(ref_field.shape, dx_convention=dx_con_to)

    print(from_field.__name__,converter.__name__, to_field.__name__)
    ic(n_grid, ref_grid + ref_field,n_grid - (ref_grid + ref_field) )

    assert torch.allclose(n_grid, ref_grid + ref_field, atol=1e-6)
#
#
# def test_pixel_to_square_convention_2d_field():
#     field = sample_field_2d_pixel()
#     field = tb.pixel_to_square_convention(field, is_grid=False)
#     ref_field = sample_field_2d_square()
#     assert torch.all(field == ref_field)
#
# def test_square_to_pixel_convention_2d():
#     field = sample_field_2d_2square()
#     field = tb.square_to_pixel_convention(field,is_grid=False)
#     ref_field = sample_field_2d_pixel()
#     assert torch.all(field == ref_field)


