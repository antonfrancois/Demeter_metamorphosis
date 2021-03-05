import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample
from kornia.filters import SpatialGradient,filter2D
from kornia.geometry.transform import resize
import kornia.utils.grid as kg
from numpy import newaxis

from my_toolbox import rgb2gray

# ================================================
#        IMAGE BASICS
# ================================================

def reg_open(number, size = None,requires_grad= False):
    path = 'im2Dbank/reg_test_'+number+'.png'

    I = rgb2gray(plt.imread(path))
    I = torch.tensor(I[newaxis,newaxis,:],
                 dtype=torch.float,
                 requires_grad=requires_grad)
    if size is None:
        return I
    else:
        return resize(I,size)

def thresholding(image,bounds = (0,1)):
    return torch.maximum(torch.tensor(bounds[0]),
                         torch.minimum(
                             torch.tensor(bounds[1]),
                                          image
                                      )
                         )

def spacialGradient(image,dx_convention = 'pixel'):
    # TODO : make it work on 3D images
    grad_image = SpatialGradient(mode='sobel')(image)
    if dx_convention == '2square':
        _,_,H,W = image.size()
        grad_image[0,0,0] *= (H-1)/2
        grad_image[0,0,1] *= (W-1)/2
    return grad_image


# =================================================
#            PLOT
# =================================================
def imCmp(I1, I2):
    from numpy import concatenate,zeros
    _,_,M, N = I1.shape
    I1 = I1[0,0,:,:].detach().numpy()
    I2 = I2[0,0,:,:].detach().numpy()

    return concatenate((I2[:, :, None], I1[:, :, None], zeros((M, N, 1))), axis=2)

def checkDiffeo(field):
    _,H,W,_ = field.shape
    det_jaco = detOfJacobian(field_2d_jacobian(field))[0]
    I = .4 * torch.ones((H,W,3))
    I[:,:,0] = (det_jaco <=0) *0.83
    I[:,:,1] = (det_jaco >=0)

    return I

def gridDef_plot(deformation,
                 ax=None,
                 step=2,
                 add_grid=False,
                 check_diffeo=False,
                 title="",
                 color=None,
                 dx_convention='pixel'):
    """

    :param field: field to represent
    :param grid:
    :param saxes[1]:
    :param title:
    :return:
    """
    if not torch.is_tensor(deformation):
        raise TypeError("showDef has to be tensor object")
    if deformation.size().__len__() != 4 or deformation.size()[0] > 1:
        raise TypeError("deformation has to be a (1,H,W,2) or (1,H,W,D,3) "
                        "tensor object got "+str(deformation.size()))
    deform = deformation.clone()
    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        color = 'black'

    if add_grid:
        deform += make_regular_grid(deform.size(),dx_convention=dx_convention)

    if check_diffeo :
        cD = checkDiffeo(deform)

        title += 'diffeo = '+str(cD[:,:,0].sum()<=0)

        ax.imshow(cD,interpolation='none',origin='lower')

    ax.plot(deform[0,:,::step, 0].numpy(),
                 deform[0,:,::step, 1].numpy(), color=color)
    ax.plot(deform[0,::step,:, 0].numpy().T,
                 deform[0,::step,:, 1].numpy().T, color=color)

    ax.plot(deform[0,:,-1, 0].numpy(),
                 deform[0,:,-1, 1].numpy(), color=color)
    ax.plot(deform[0,-1,:, 0].numpy().T,
                 deform[0,-1,:, 1].numpy().T, color=color)
    ax.set_aspect('equal')
    ax.set_title(title)


    return ax

def quiver_plot(field,
                ax=None,
                step=2,
                title="",
                check_diffeo=False,
                color=None,
                real_scale=True):
    """

    :param field: field to represent
    :param grid:
    :param saxes[1]:
    :param title:
    :return:axes

    """
    if not torch.is_tensor(field):
        raise TypeError("field has to be tensor object")
    if field.size().__len__() != 4 or field.size()[0] > 1:
        raise TypeError("field has to be a (1",
                        "H,W,2) or (1,H,W,D,3) tensor object got ",
                        str(field.size()))

    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        color = 'black'

    reg_grid = make_regular_grid(field.size())
    if check_diffeo :
         cD = checkDiffeo(reg_grid+field)
         title += 'diffeo = '+str(cD[:,:,0].sum()<=0)

         ax.imshow(cD,interpolation='none',origin='lower')

    # real scale =1 means to plot quiver arrow with axis scale
    (scale_units,scale ) = ('xy',1) if real_scale else (None,None)

    ax.quiver(reg_grid[0,::step,::step,0],reg_grid[0,::step,::step,1],
            ((field[0,::step, ::step, 0] ).detach().numpy()),
            ((field[0,::step, ::step, 1] ).detach().numpy()),
            color=color,
            scale_units=scale_units, scale=scale)

    return ax

def deformation_show(deformation,step=2,
                     check_diffeo=False,title="",color=None):
    r"""

    :param deformation:
    :param step:
    :param check_diffeo:
    :return:

    Example :
    cms = mbs.getCMS_allcombinaision()

    H,W = 100,150
    # vector defomation generation
    v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)
    v *= 0.5

    deform_diff = vff.FieldIntegrator(method='fast_exp')(v.clone(),forward= True)

    deformation_show(deform_diff,step=4,check_diffeo=True)
    """

    fig, axes = plt.subplots(1,2)
    fig.suptitle(title)
    regular_grid = make_regular_grid(deformation.size())
    gridDef_plot(deformation,step=step,ax = axes[0],
                 check_diffeo=check_diffeo,
                 color=None)
    quiver_plot(deformation - regular_grid,step=step,
                ax = axes[1],check_diffeo=check_diffeo,
                color=None)
    plt.show()

def vectField_show(field,step=2,check_diffeo= False,title=""):
    r"""

    :param field: (1,H,W,2) tensor object
    :param step:
    :param check_diffeo: (bool)
    :return:

    Example :
    cms = mbs.getCMS_allcombinaision()

    H,W = 100,150
    # vector defomation generation
    v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)
    v *= 0.5

    vectField_show(v,step=4,check_diffeo=True)
    """

    fig, axes = plt.subplots(1,2)
    fig.suptitle(title)
    regular_grid = make_regular_grid(field.size())
    gridDef_plot(field + regular_grid,step=step,ax = axes[0],
                 check_diffeo=check_diffeo)
    quiver_plot(field ,step=step,
                ax = axes[1],check_diffeo=check_diffeo)
    plt.show()

def showDef(field,axes=None, grid=None, step=2, title="",check_diffeo=False,color=None):
     print("\033[93m WARNING"+
         "function deprecated /!\ do not use /!\ see defomation_show"+
         "\033[0m")
     return deformation_show(field)

# =================================================
#            FIELD RELATED FUNCTIONS
# =================================================

def fieldNorm2(field):
    return (field**2).sum(dim=-1)

def field2diffeo(in_vectField, N=None,save= False,forward=True):
   """function deprecated /!\ do not use /!\ see vector_field_to_flow"""
   print("\033[93m WARNING"+
         "function deprecated /!\ do not use /!\ see vector_field_to_flow"+
         "\033[0m")
   import vector_field_to_flow as vff
   return vff.FieldIntegrator(method='fast_exp')(in_vectField.clone(),forward= forward)


def imgDeform(I,field,dx_convention ='2square'):
    if dx_convention == 'pixel':
        field = pixel2square_convention(field)
    return grid_sample(I,field, padding_mode="border", align_corners=True)

def compose_fields(field,field_on):
    return im2grid(grid_sample(grid2im(field),field_on))


def field_2d_jacobian(field):
    r"""

    :param field: field.size (B,H,W,2)
    :return: output.size = (B,2,2,H,W)

    :example:
field = torch.zeros((100,100,2))
field[::2,:,0] = 1
field[:,::2,1] = 1

jaco =  field_2d_jacobian(field)


plt.rc('text',usetex=True)
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(jaco[0,0,0,:,:].detach().numpy(),cmap='gray')
axes[0,0].set_title(r"$\frac{\partial f_1}{\partial x}$")
axes[0,1].imshow(jaco[0,0,1,:,:].detach().numpy(),cmap='gray')
axes[0,1].set_title(r"$\frac{\partial f_1}{\partial y}$")
axes[1,0].imshow(jaco[0,1,0,:,:].detach().numpy(),cmap='gray')
axes[1,0].set_title(r"$\frac{\partial f_2}{\partial x}$")
axes[1,1].imshow(jaco[0,1,1,:,:].detach().numpy(),cmap='gray')
axes[1,1].set_title(r"$\frac{\partial f_2}{\partial y}$")

plt.show()
    """

    f_d = grid2im(field)
    return SpatialGradient()(f_d)

def field_2d_hessian(field_grad):
    r""" compute the hessian of a field from the jacobian

    :param field_grad: BxnxpxHxW tensor n = p = 2
    :return: Bx8x2xHxW tensor

    :example :

    hess = field_2d_hessian(I_g)
    print('hess.shape = '+str(hess.shape))
    fig, axes = plt.subplots(2,4)
    for x in range(2):
        for d in range(4):
            axes[x][d].imshow(hess[0,d,x,:,:].detach().numpy(),cmap='gray')
            axes[x][d].set_title(str((x,d)))
    plt.show()

    """
    N,_,_,H,W = field_grad.shape
    device = 'cuda' if field_grad.is_cuda else 'cpu'
    hess = torch.zeros((N,4,2,H,W),device=device)

    hess[:,:2,:,:,:] = SpatialGradient()(field_grad[:,0,:,:,:])
    hess[:,2:,:,:,:] = SpatialGradient()(field_grad[:,1,:,:,:])

    return hess

def detOfJacobian(jaco):
    """ compute the determinant of the jacobian from field_2d_jacobian

    :param jaco: B,2,2,H,W tensor
    :return: H,W tensor
    """
    return jaco[:,0,0,:,:] * jaco[:,1,1,:,:] - jaco[:,1,0,:,:] * jaco[:,0,1,:,:]

def field_divergence(field,dx_convention = 'pixel'):
    r"""
    make the divergence of a field, for each pixel $p$ in I
    $$div(I(p)) = \sum_{i=1}^C \frac{\partial I(p)_i}{\partial x_i}$$
    :param field: (B,H,W,2) tensor
    :return:

    cms = torch.tensor([  # control matrices
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, -1, 0, -1, 0, -1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, -1, 0, +1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, +1, 0, -1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, -1, 0, -1, 0, -1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     ],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, +1, 0, -1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, .2, .75, 1, 0],
     [0, -1, 0, -1, 0, -1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, -1, 0, -1, 0, -1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, +1, 0, -1, 0, +1, 0, +1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ],requires_grad=False,dtype=torch.float)

    field_size = (20,20)
    field = mbs.field2D_bspline(cms,field_size,
                                degree=(3,3),dim_stack=2).unsqueeze(0)

    # field_diff = vect_spline_diffeo(cms,field_size)
    H,W = field_size
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))

    div = field_2d_divergence(field)

    # _,d_ax = plt.subplots()
    fig,ax = plt.subplots()

    div_plot = ax.imshow(div[0,0,:,:],origin='lower')
    ax.quiver(field[0,:,:,0],field[0,:,:,1])
    fig.colorbar(div_plot)
    plt.show()
    """
    _,H,W,_ = field.shape
    x_sobel = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])/8
    field_as_im = grid2im(field)
    field_x_dx = filter2D(field_as_im[:,0,:,:].unsqueeze(1),
                          x_sobel.unsqueeze(0))# * (2/(H-1)))
    field_y_dy = filter2D(field_as_im[:,1,:,:].unsqueeze(1),
                          x_sobel.T.unsqueeze(0))# * (2/(W-1)))

    if dx_convention == '2square':
        _,H,W,_ = field.shape
        return field_x_dx*(H-1)/2 + field_y_dy*(W-1)/2
    else:
        return field_x_dx + field_y_dy

def pixel2square_convention(field,grid = True):
    """ Convert a field in spacial pixelic convention in one on as
    [-1,1]^2 square as requested by pytorch's gridSample

    :return:
    """
    field = field.clone()
    if field.shape[-1] == 2 :
        _,H,W,_ = field.shape
        mult = torch.tensor((2/(W-1),2/(H-1)))
        sub = 1 if grid else 0
        return field * mult[None,None,None] - sub
    else:
        raise NotImplementedError("Indeed")

def square2pixel_convention(field,grid=True):
    """ Convert a field on a square centred and from -1 to 1 convention
    as requested by pytorch's gridSample to one in pixelic convention

    :return:
    """
    field = field.clone()
    if field.shape[-1] == 2 :
        _,H,W,_ = field.shape
        mult = torch.tensor((W/2,H/2))
        add = 1 if grid else 0
        return (field + add) * mult[None,None,None]


    else:
        raise NotImplementedError("Indeed")


def grid2im(grid):
    """Reshape a grid tensor into an image tensor
        2D  [T,H,W,2] -> [T,2,H,W]
        3D  [T,D,H,W,2] -> [T,D,H,W,3]

    # grid to image
    T,D,H,W = (4,5,6,7)

    grid_2D = torch.rand((T,H,W,2))
    grid_3D = torch.rand((T,D,H,W,3))

    image_2D = torch.rand((T,2,H,W))
    image_3D = torch.rand((T,3,D,H,W))

    grid_2D_as_image = grid2im(grid_2D)
    grid_3D_as_image = grid2im(grid_3D)

    # check if the method works
    print('\n  GRID TO IMAGE')
    print(' ==== 2D ====\n')
    print('grid_2D.shape =',grid_2D.shape)
    print('grid_2D_as_image.shape =',grid_2D_as_image.shape)
    print('we have indeed the good shape')
    count = 0
    for i in range(T):
        count += (grid_2D[i,...,0] == grid_2D_as_image[i,0,...]).sum()
        count += (grid_2D[i,...,1] == grid_2D_as_image[i,1,...]).sum()

    print('count is equal to ',count/(T*H*W*2),'and should be equal to 1')

    print(' \n==== 3D ====\n')
    print('grid_3D.shape =',grid_3D.shape)
    print('grid_3D_as_image.shape =',grid_3D_as_image.shape)
    print('we have indeed the good shape')
    count = 0
    for i in range(T):
        count += (grid_3D[i,...,0] == grid_3D_as_image[i,0,...]).sum()
        count += (grid_3D[i,...,1] == grid_3D_as_image[i,1,...]).sum()
        count += (grid_3D[i,...,2] == grid_3D_as_image[i,2,...]).sum()


    print('count is equal to ',count/(T*H*W*D*3),'and should be equal to 1')

    """
    if grid.shape[-1] == 2: # 2D case
        return grid.transpose(2,3).transpose(1,2)

    elif grid.shape[-1] == 3: # 3D case
        return grid.transpose(3,4).transpose(2,3).transpose(1,2)
    else:
        raise ValueError("input argument expected is [N,H,W,2] or [N,D,H,W,3]",
                         "got "+str(grid.shape)+" instead.")

def im2grid(image):
    """Reshape an image tensor into a grid tensor
        2D case [T,2,H,W]   ->  [T,H,W,2]
        3D case [T,3,D,H,W] ->  [T,D,H,W,3]

    T,D,H,W = (4,5,6,7)

    grid_2D = torch.rand((T,H,W,2))
    grid_3D = torch.rand((T,D,H,W,3))

    image_2D = torch.rand((T,2,H,W))
    image_3D = torch.rand((T,3,D,H,W))

    # image to grid
    image_2D_as_grid = im2grid(image_2D)
    image_3D_as_grid = im2grid(image_3D)

    print('\n  IMAGE TO GRID')
    print(' ==== 2D ====\n')
    print('image_2D.shape = ',image_2D.shape)
    print('image_2D_as_grid.shape = ',image_2D_as_grid.shape)

    count = 0
    for i in range(T):
        count += (image_2D[i,0,...] == image_2D_as_grid[i,...,0]).sum()
        count += (image_2D[i,1,...] == image_2D_as_grid[i,...,1]).sum()
    print('count is equal to ',count/(T*H*W*2),'and should be equal to 1')

    print(' ==== 3D ====\n')
    print('image_3D.shape = ',image_3D.shape)
    print('image_3D_as_grid.shape = ',image_3D_as_grid.shape)

    count = 0
    for i in range(T):
        count += (image_3D[i,0,...] == image_3D_as_grid[i,...,0]).sum()
        count += (image_3D[i,1,...] == image_3D_as_grid[i,...,1]).sum()
        count += (image_3D[i,2,...] == image_3D_as_grid[i,...,2]).sum()
    print('count is equal to ',count/(T*H*W*D*3.0),'and should be equal to 1')
    """
    # No batch
    if image.shape[1] == 2:
        return image.transpose(1,2).transpose(2,3)
    elif image.shape[1] == 3:
        return image.transpose(1,2).transpose(2,3).transpose(3,4)
    else:
        raise ValueError("input argument expected is [1,2,H,W] or [1,3,D,H,W]",
                         "got "+str(image.shape)+" instead.")


def make_regular_grid(deformation_shape,device = torch.device('cpu'), dx_convention = 'pixel'):
    """API for create_meshgrid, it is the identity deformation

    :param deformation_shape: tuple such as
    (H,W) or (n,H,W,2) for 2D grid
    (D,H,W) or (n,D,H,W,3) for 3D grid
    :param device: device for selecting cpu or cuda usage
    :return: will return 2D identity deformation with size (1,H,W,2) or
    3D identity deformation with size (1,D,H,W,3)
    """
    if dx_convention == 'pixel':
        normalized_coordinates = False
    elif dx_convention == '2square':
        normalized_coordinates = True

    if len(deformation_shape) == 2 :
        H,W = deformation_shape
        return kg.create_meshgrid(H,W,
                                  normalized_coordinates=normalized_coordinates,device=device)
    elif len(deformation_shape) == 3:
        D,H,W = deformation_shape
        return kg.create_meshgrid3d(D,H,W,
                                    normalized_coordinates=normalized_coordinates,device=device)
    elif len(deformation_shape) == 4 or len(deformation_shape) == 5 :
        d = deformation_shape[-1]

        if d ==2:
            _,H,W,_ = deformation_shape
            return kg.create_meshgrid(H,W,
                                      normalized_coordinates=normalized_coordinates,device=device)
        elif d == 3:
            _,D,H,W,_ = deformation_shape
            return kg.create_meshgrid3d(D,H,W,
                                        normalized_coordinates=normalized_coordinates,device=device)

# =================================================================
#             LIE ALGEBRA
# =================================================================

def leviCivita_2Dderivative(v,w):
    """ Perform the operation $\nabla_w v$"""

    d_v = field_2d_jacobian(v)
    d_v_x = w[:,:,0]*d_v[0,0,0,:,:] + w[:,:,1]*d_v[0,0,1,:,:]
    d_v_y = w[:,:,0]*d_v[0,1,0,:,:] + w[:,:,1]*d_v[0,1,1,:,:]

    return torch.stack((d_v_x,d_v_y),dim=2)

def lieBracket(v,w):
    return leviCivita_2Dderivative(v,w) - leviCivita_2Dderivative(w,v)

def BCH(v,w,order = 0):
    """ Evaluate the Backer-Campbell-Hausdorff formula"""
    var = v + w
    if order >= 1:
        # print(lieBracket(v,w))
        var += 0.5*lieBracket(v,w)
    if order == 2:
        var += (lieBracket(v,lieBracket(v,w)) - lieBracket(w,lieBracket(v,w)))/12
    if order > 2:
        print('non')
    return var


# =================================================================
#             KERNELS
# =================================================================

