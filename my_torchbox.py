import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient,SpatialGradient3d,filter2d,filter3d
from kornia.geometry.transform import resize
import kornia.utils.grid as kg
from numpy import newaxis
from matplotlib.widgets import Slider
from nibabel import load as nib_load
from skimage.exposure import match_histograms

# import decorators
from my_toolbox import rgb2gray
# import my_bspline as mbs
import vector_field_to_flow as vff
import decorators as deco
from constants import ROOT_DIRECTORY

# ================================================
#        IMAGE BASICS
# ================================================

def reg_open(number, size = None,requires_grad= False,location = 'local',device='cpu'):

    # if location == 'local':
    #     path = '/home/turtlefox/Documents/Doctorat/'
    # elif location == 'bartlett':
    #     path = '/home/fanton/'
    path = ROOT_DIRECTORY
    path += '/im2Dbank/reg_test_'+number+'.png'

    I = rgb2gray(plt.imread(path))
    I = torch.tensor(I[newaxis,newaxis,:],
                 dtype=torch.float,
                 requires_grad=requires_grad,
                 device=device)
    if size is None:
        return I
    else:
        return resize(I,size)

def open_nib(folder_name,irm_type,data_base,format= '.nii.gz',normalize=True, to_torch =True):

    if data_base == 'brats':
        path = ROOT_DIRECTORY+ '/../data/brats/'
    elif data_base == 'brats_2021':
        path = ROOT_DIRECTORY+ '/../data/brats_2021/'
    else:
        path = data_base
    img_nib = nib_load(path+folder_name+'/'+folder_name+'_'+irm_type+format)
    img = img_nib.get_fdata()
    method = None
    if isinstance(normalize,str):
        method = normalize
        normalize = True
    if normalize: img = nib_normalize(img,method=method)
    if to_torch: return torch.Tensor(img)[None,None]
    else: return img_nib

def open_template_icbm152(ponderation = 't1',normalize= True):
    path = ROOT_DIRECTORY+ '/../data/template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/'
    format = '.mgz' if ponderation == 'segV' else '.nii'
    file_name = 'mni_icbm152_'+ponderation+'_tal_nlin_asym_09c'+format
    img = nib_load(path+file_name).get_fdata()
    seg = nib_load(path+'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii').get_fdata()
    seg = seg/seg.max() # segmentation value is exaclty one ...
    # take off the skull and all non brain elements
    img[seg == 0] = 0
    method = None
    if isinstance(normalize,str):
        method = normalize
        normalize = True
    if normalize: img = nib_normalize(img,method=method)
    return torch.Tensor(img)[None,None]

class parse_brats:

    def __init__(self,brats_list,
                 template_folder=None,
                 brats_folder=None,
                 modality='T1',
                 device = 'cpu'):
        """

        :param brats_list: list of stings containing the name of the folders
        :param template_folder: path to template folder
        :param brats_folder: path to brats db
        :param modality: modality of the IRM ex: `'T1'`,`'T2'`
        """
        if template_folder is None:
            template_folder = ROOT_DIRECTORY+'/../data/template/sri_spm8/templates/'
        if brats_folder is None:
            brats_folder = ROOT_DIRECTORY+'/../data/brats_2021/'
        # TODO : check that the list is correct by parsing with os.get_dir ...
        self.brats_list = brats_list
        self.modality = modality
        self.device = device

        template_nib = nib_load(template_folder+modality+"_brain.nii")
        self.template_affine = template_nib.affine
        self.template_img = template_nib.get_fdata()[:,::-1,:,0]

        template_seg = nib_load(template_folder+'seg_sri24.mgz').get_fdata()
        self.vesi_seg = torch.zeros(template_seg.shape)
        self.vesi_seg[template_seg==4] = 1
        self.vesi_seg[template_seg==43] = 1

    def get_template(self,normalised=True):
        if normalised:
            img_norm = (self.template_img - self.template_img.min()) / (self.template_img.max() - self.template_img.min())
            return torch.Tensor(img_norm,device=self.device)[None,None]
        else:
            return torch.Tensor(self.template_img,device=self.device)

    def get_template_vesi(self):
        return self.vesi_seg.flip(1)

    def get_vesicule_seg(self,index):
        brats_name = self.brats_list[index]

        vesi = open_nib(brats_name,'segV','brats_2021',
                          format='.nii',normalize=False,to_torch=True)
        # img_1 = torch.zeros(vesi.shape)
        # img_1[vesi==4] = 1
        # img_1[vesi==43] = 1
        return vesi[0,0]

    def __call__(self, index):
        """ Open the brats folder in self.brats_list at the desired index

        :param index: must be int < len(brats_list)
        :return: im
        """
        brats_name = self.brats_list[index]
        source = open_nib(brats_name,self.modality.lower(),'brats_2021',normalize=False,to_torch=False)
        segmentation_tumor = open_nib(brats_name,'seg','brats_2021',normalize=False,to_torch=True)

        # source = nib.load("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_000_T1.nii.gz")
        # histogram normalisation
        source_img = source.get_fdata()
        source_img[source_img > 0] = match_histograms(source_img[source_img > 0], self.template_img[self.template_img > 0])
        source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())

        return (torch.Tensor(source_img,device=self.device)[None,None],
                segmentation_tumor.to(self.device))
        # source = nib.Nifti1Image(source_img, self.template_affine)
        # return source.get_fdata()

def nib_normalize(img,method='mean'):
    if method == 'mean' or method is None:
        img = (img -img.mean())/img.std()
        img = img/img.max()
    elif method == 'min_max':
        img += img.min()
        img /= img.max()
    else:
        raise ValueError(f"method must be 'mean' or 'min_max' gor {method}")
    return img

def pad_to_same_size(img_1,img_2):
    """ Pad the two images in order to make images of the same size
    takes

    :param img_1: [T_1,C,D_1,H_1,W_1] or [D_1,H_1,W_1]  torch tensor
    :param img_2: [T_2,C,D_2,H_2,W_2] or [D_2,H_2,W_2]  torch tensor
    :return: will return both images with of shape
    [...,max(D_1,D_2),max(H_1,H_2),max(W_1,W_2)] in a tuple.
    """
    diff = [x - y for x,y in zip(img_1.shape[2:],img_2.shape[2:])]
    padding1,padding2 = (),()
    for d in reversed(diff):
        is_d_even = d % 2
        if d//2 < 0:
            padding1 += (-d//2 + is_d_even,-d//2)
            padding2 += (0,0)
        else:
            padding1 += (0,0)
            padding2 += (d//2 + is_d_even,d//2)
    img_1_padded = torch.nn.functional.pad(img_1[0,0],padding1)[None,None]
    img_2_padded = torch.nn.functional.pad(img_2[0,0],padding2)[None,None]
    return (img_1_padded,img_2_padded)

def addGrid2im(img, n_line,cst=0.1,method='dots'):
    """

    :param img:
    :param n_line:
    :param cst:
    :param method:
    :return:
    """

    if isinstance(img,tuple):
        img = torch.zeros(img)
    if len(img.shape) == 4:
        _,_,H,W = img.shape
        is_3D = False
    elif len(img.shape) == 5:
        _,_,H,W,D = img.shape
        is_3D = True
    else: raise ValueError(f"img should be [B,C,H,W] or [B,C,D,H,W] got {img.shape}")

    try:
        len(n_line)
    except:
        n_line = (n_line,)*(len(img.shape)-2)

    add_mat = torch.zeros(img.shape)
    row_mat = torch.zeros(img.shape,dtype=torch.bool)
    col_mat = torch.zeros(img.shape,dtype=torch.bool)

    row_centers = (torch.arange(n_line[0])+1) * H //n_line[0]
    row_width = int(max(H / 200, 1))
    for lr,hr in zip(row_centers-row_width,row_centers+row_width):
        row_mat[:,:,lr:hr] = True

    col_centers = (torch.arange(n_line[1])+1) * W //n_line[1]
    col_width = int(max(W / 200, 1))
    for lc,hc in zip(col_centers-col_width,col_centers+col_width):
        col_mat[:,:,:,lc:hc] = True

    if is_3D:
        depth_mat = torch.zeros(img.shape,dtype=torch.bool)
        depth_centers = (torch.arange(n_line[2])+1) * D //n_line[2]
        depth_width = int(max(D / 200, 1))
        for ld,hd in zip(depth_centers-depth_width,depth_centers+depth_width):
            depth_mat[:,:,:,:,ld:hd] = True

    if method == 'lines':
        add_mat[row_mat] += cst
        add_mat[col_mat] += cst
        if is_3D:
            add_mat[depth_mat] += cst
    elif method == 'dots':
        bool = torch.logical_and(row_mat, col_mat)
        if is_3D:
            bool = torch.logical_and(bool,depth_mat)
        add_mat[bool] = cst
    else:
        raise ValueError(f"method must be among `lines`,`dots` got {method}")

    #put the negative grid on the high values image
    add_mat[img > .5] *= -1

    return img + add_mat

def thresholding(image,bounds = (0,1)):
    return torch.maximum(torch.tensor(bounds[0]),
                         torch.minimum(
                             torch.tensor(bounds[1]),
                                          image
                                      )
                         )

def spacialGradient(image,dx_convention = 'pixel'):
    if len(image.shape) == 4 :
        return spacialGradient_2d(image,dx_convention)
    elif len(image.shape) == 5:
        return spacialGradient_3d(image,dx_convention)

def spacialGradient_2d(image,dx_convention = 'pixel'):
    """ Compute the spatial gradient on 2d images by applying
    a sobel kernel

    :param image: Tensor [B,C,H,W]
    :param dx_convention:
    :return: [B,C,2,H,W]
    """

    grad_image = SpatialGradient(mode='sobel')(image)
    # grad_image[:,0,1] *= -1
    if dx_convention == '2square':
        _,_,H,W = image.size()
        grad_image[:,0,0] *= (H-1)/2
        grad_image[:,0,1] *= (W-1)/2
    return grad_image


def spacialGradient_3d(image,dx_convention = 'pixel'):
    """

    :param image: Tensor [B,1,D,H,W]
    :param dx_convention:
    :return: Tensor [B,C,3,D,H,W]

    :Example:
    H,W,D = (50,75,100)
    image = torch.zeros((H,W,D))
    mX,mY,mZ = torch.meshgrid(torch.arange(H),
                              torch.arange(W),
                              torch.arange(D))

    mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4
    mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)
    mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4
    mask = mask_rond & mask_carre & mask_diamand
    image[mask] = 1


    grad_image = spacialGradient_3d(image[None,None])
    # grad_image_sum = grad_image.abs().sum(dim=1)
    # iv3d.imshow_3d_slider(grad_image_sum[0])

    """

    # sobel kernel is not implemented for 3D images yet in kornia
    # grad_image = SpatialGradient3d(mode='sobel')(image)
    kernel = get_sobel_kernel_3d().to(image.device).to(image.dtype)
    spatial_pad = [1,1,1,1,1,1]
    image_padded = F.pad(image,spatial_pad,'replicate').repeat(1,3,1,1,1)
    grad_image =  F.conv3d(image_padded,kernel,padding=0,groups=3,stride=1)
    if dx_convention == '2square':
        _,_,D,H,W, = image.size()
        grad_image[0,0,0] *= (D-1)/2
        grad_image[0,0,1] *= (W-1)/2
        grad_image[0,0,2] *= (H-1)/2

    return grad_image.unsqueeze(1)

def get_sobel_kernel_2d():
    return torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

def get_sobel_kernel_3d():
    return torch.tensor(
    [
        [[[-1,0,1],
          [-2,0,2],
          [-1,0,1]],

         [[-2,0,2],
          [-4,0,4],
          [-2,0,2]],

         [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]],

        [[[-1,-2,-1],
          [0,0,0],
          [1,2,1]],

         [[-2,-4,-2],
          [0,0,0],
          [2,4,2]],

         [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]],

        [[[-1,-2,-1],
          [-2,-4,-2],
          [-1,-2,-1]],

         [[0,0,0],
          [0,0,0],
          [0,0,0]],

         [[1,2,1],
          [2,4,2],
          [1,2,1]]]
    ]).unsqueeze(1)



# =================================================
#            PLOT
# =================================================
def imCmp(I1, I2):
    from numpy import concatenate,zeros
    _,_,M, N = I1.shape
    I1 = I1[0,0,:,:].detach().cpu().numpy()
    I2 = I2[0,0,:,:].detach().cpu().numpy()

    return concatenate((I2[:, :, None], I1[:, :, None], zeros((M, N, 1))), axis=2)

def checkDiffeo(field):
    _,H,W,_ = field.shape
    det_jaco = detOfJacobian(field_2d_jacobian(field))[0]
    I = .4 * torch.ones((H,W,3))
    I[:,:,0] = (det_jaco <=0) *0.83
    I[:,:,1] = (det_jaco >=0)

    return I

@deco.deprecated("Please specify the dimension by using gridDef_plot_2d ot gridDef_plot_3d")
def gridDef_plot(defomation,
                 ax=None,
                 step=2,
                 add_grid=False,
                 check_diffeo=False,
                 title="",
                 color=None,
                 dx_convention='pixel'):
    return gridDef_plot_2d(defomation,ax=ax,step=step,add_grid=add_grid,
                    check_diffeo=check_diffeo,title=title,
                    color=color,dx_convention=dx_convention)


def gridDef_plot_2d(deformation,
                 ax=None,
                 step=2,
                 add_grid=False,
                 check_diffeo=False,
                 dx_convention='pixel',
                 title="",
                 color=None,
                linewidth=None
                ):
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
        raise TypeError("deformation has to be a (1,H,W,2) "
                        "tensor object got "+str(deformation.size()))
    deform = deformation.clone()
    if ax is None:
        fig, ax = plt.subplots()

    if color is None: color = 'black'
    if linewidth is None: linewidth = 2

    if add_grid:
        reg_grid = make_regular_grid(deform.size(),dx_convention='pixel')
        if dx_convention == '2square':
            reg_grid = pixel2square_convention(reg_grid)
        deform += reg_grid

    if check_diffeo :
        cD = checkDiffeo(deform)

        title += 'diffeo = '+str(cD[:,:,0].sum()<=0)

        ax.imshow(cD,interpolation='none',origin='lower')

    ax.plot(deform[0,:,::step, 0].numpy(),
                 deform[0,:,::step, 1].numpy(), color=color,linewidth=linewidth)
    ax.plot(deform[0,::step,:, 0].numpy().T,
                 deform[0,::step,:, 1].numpy().T, color=color,linewidth=linewidth)

    # add the last lines on the right and bottom edges
    ax.plot(deform[0,:,-1, 0].numpy(),
                 deform[0,:,-1, 1].numpy(), color=color,linewidth=linewidth)
    ax.plot(deform[0,-1,:, 0].numpy().T,
                 deform[0,-1,:, 1].numpy().T, color=color,linewidth=linewidth)
    ax.set_aspect('equal')
    ax.set_title(title)


    return ax

def quiver_plot(field,
                ax=None,
                step=2,
                title="",
                check_diffeo=False,
                color=None,
                dx_convention='pixel',
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

    reg_grid = make_regular_grid(field.size(),dx_convention=dx_convention)
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

def vectField_show(field,step=2,check_diffeo= False,title="",
                   dx_convention = 'pixel'):
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
                 check_diffeo=check_diffeo,
                 dx_convention=dx_convention)
    quiver_plot(field ,step=step,
                ax = axes[1],check_diffeo=check_diffeo,
                dx_convention=dx_convention)
    plt.show()

def geodesic_3d_slider(mr):
    """ Display a 3d image

    exemple:
    mr = mt.load_optimize_geodesicShooting('2D_13_10_2021_m0t_m1_001.pk1')
    geodesic_3d_slider(mr)
    """
    image = mr.mp.image_stock.numpy()
    print(image.shape)
    residuals = mr.mp.residuals_stock.numpy()


    fig,ax = plt.subplots(1,2)

    kw_image_args = dict(cmap='gray',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
    kw_residuals_args = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=residuals.min(),
                      vmax=residuals.max())

    img_x = ax[0].imshow( image[0,0], **kw_image_args)
    img_y = ax[1].imshow(residuals[0], **kw_residuals_args)
    # img_z = ax[2].imshow( image_slice(image,init_z_coord,dim=2),origin='lower', **kw_image)
    ax[0].set_xlabel('X')
    ax[1].set_xlabel('Y')
    # ax[2].set_xlabel('Z')


    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1,right=.9, bottom=0.25)

    # Make sliders.
    axcolor = 'lightgoldenrodyellow'
    #place them [x_bottom,y_bottom,height,width]
    sl_t = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(
        valmin=0,
        valfmt='%0.0f'
    )
    t_slider = Slider(label='t',ax=sl_t,
                      valmax=image.shape[0]-1,valinit=0,
                      **kw_slider_args)

    # The function to be called anytime a slider's value changes
    def update(val):
        img_x.set_data(image[int(t_slider.val),0])
        img_y.set_data(residuals[int(t_slider.val)])
        # img_z.set_data(image_slice(image, z_slider.val, 2))

        fig.canvas.draw_idle()

    # register the update function with each slider
    t_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return t_slider

@deco.deprecated("function deprecated /!\ do not use /!\ see defomation_show")
def showDef(field,axes=None, grid=None, step=2, title="",check_diffeo=False,color=None):
     return deformation_show(field)

# =================================================
#            FIELD RELATED FUNCTIONS
# =================================================

def fieldNorm2(field):
    return (field**2).sum(dim=-1)

@deco.deprecated("function deprecated /!\ do not use /!\ see vector_field_to_flow")
def field2diffeo(in_vectField, N=None,save= False,forward=True):
   """function deprecated /!\ do not use /!\ see vector_field_to_flow"""
   return vff.FieldIntegrator(method='fast_exp')(in_vectField.clone(),forward= forward)


def imgDeform(I,field,dx_convention ='2square',clamp=True):
    if dx_convention == 'pixel':
        field = pixel2square_convention(field)
    deformed = F.grid_sample(I,field, padding_mode="border", align_corners=True)
    # if len(I.shape) == 5:
    #     deformed = deformed.permute(0,1,4,3,2)
    if clamp:
        max_val = 1 if I.max() <= 1 else 255
        # print(f"I am clamping max_val = {max_val}, I.max,min = {I.max(),I.min()},")
        deformed = torch.clamp(deformed,min=0,max=max_val)
    return deformed

def compose_fields(field,field_on):
    return im2grid(F.grid_sample(grid2im(field),field_on))

def vect_spline_diffeo(control_matrix,field_size, N = None,forward = True):
    field = mbs.field2D_bspline(control_matrix, field_size, dim_stack=2)[None]
    return vff.FieldIntegrator(method='fast_exp')(field.clone(),forward= forward)

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

class Field_divergence(torch.nn.Module):

    def __init__(self,dx_convention='pixel'):
        self.dx_convention = dx_convention

        super(Field_divergence, self).__init__()

    def __repr__(self):
        return (
            self.__class__.__name__
            +'(field dimension ='
            +self.field_dim+'d, '
            +'dx_convention ='
            +self.dx_convention
            +')'
        )

    def forward(self,field):
        field_as_im = grid2im(field)
        if field.shape[-1] == 2:
            x_sobel = get_sobel_kernel_2d().to(field.device)/8

            field_x_dx = filter2d(field_as_im[:,0,:,:].unsqueeze(1),
                          x_sobel.unsqueeze(0))# * (2/(H-1)))
            field_y_dy = filter2d(field_as_im[:,1,:,:].unsqueeze(1),
                          x_sobel.T.unsqueeze(0))# * (2/(W-1)))

            field_div = torch.stack([field_x_dx, field_y_dy],dim=0)

        elif field.shape[-1] == 3:
            x_sobel = get_sobel_kernel_3d().to(field.device)

            field_x_dx = filter3d(field_as_im[:,0].unsqueeze(1),
                                  x_sobel[0]/x_sobel[0].abs().sum())
            field_y_dy = filter3d(field_as_im[:,1].unsqueeze(1),
                                  x_sobel[1]/x_sobel[1].abs().sum()) # TODO : might be a kind of transposition of the thing
            field_z_dz = filter3d(field_as_im[:,2].unsqueeze(1),
                                  x_sobel[2]/x_sobel[2].abs().sum())
            field_div = torch.stack([field_x_dx, field_y_dy, field_z_dz],dim=0)

        if self.dx_convention == '2square':
            return torch.stack(
                [(s-1)/2*field_div[i] for i,s in enumerate(field_as_im.shape[2:])],
                dim=0).sum(dim=0)
        else:
            return field_div.sum(dim=0)

@deco.deprecated
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
    # _,H,W,_ = field.shape
    # x_sobel = get_sobel_kernel_2d()/8
    # # x_sobel = torch.tensor([[-1, 0, 1],
    # #                         [-2, 0, 2],
    # #                         [-1, 0, 1]])/8
    # field_as_im = grid2im(field)
    # field_x_dx = filter2d(field_as_im[:,0,:,:].unsqueeze(1),
    #                       x_sobel.unsqueeze(0))# * (2/(H-1)))
    # field_y_dy = filter2d(field_as_im[:,1,:,:].unsqueeze(1),
    #                       x_sobel.T.unsqueeze(0))# * (2/(W-1)))
    #
    # if dx_convention == '2square':
    #     _,H,W,_ = field.shape
    #     return field_x_dx*(H-1)/2 + field_y_dy*(W-1)/2
    # else:
    #     return field_x_dx + field_y_dy
    return Field_divergence(dx_convention)(field)

def pixel2square_convention(field,grid = True):
    """ Convert a field in spacial pixelic convention in one on as
    [-1,1]^2 square as requested by pytorch's gridSample

    :return:
    """
    field = field.clone()
    if field.shape[-1] == 2 :
        _,H,W,_ = field.shape
        mult = torch.tensor((2/(W-1),2/(H-1))).to(field.device)
        if not torch.is_tensor(field):
            mult = mult.numpy()
        sub = 1 if grid else 0
        return field * mult[None,None,None] - sub
    elif field.shape[-1] == 3 and len(field.shape) == 5:
        _,D,H,W,_ = field.shape
        mult = torch.tensor((2/(D-1),2/(H-1),2/(W-1))).to(field.device)
        if not torch.is_tensor(field):
            mult = mult.numpy()
        sub =1 if grid else 0
        return field * mult[None,None,None,None] - sub
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
        mult = torch.tensor(((W-1)/2,(H-1)/2))
        if not torch.is_tensor(field):
            mult = mult.numpy()
        add = 1 if grid else 0
        return (field + add) * mult[None,None,None]
    elif field.shape[-1] == 3 and len(field.shape) == 5:
        _,D,H,W,_ = field.shape
        mult = torch.tensor(((D-1)/2,(H-1)/2,(W-1)/2))
        if not torch.is_tensor(field):
            mult = mult.numpy()
        add = 1 if grid else 0
        return (field + add) * mult[None,None,None,None]
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
        raise ValueError("input argument expected is [B,2,H,W] or [B,3,D,H,W]",
                         "got "+str(image.shape)+" instead.")


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.
    """

    lx: torch.Tensor = torch.linspace(0, width - 1, depth, device=device, dtype=dtype)
    ly: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    lz: torch.Tensor = torch.linspace(0, depth - 1, width, device=device, dtype=dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        lx = (lx / (width - 1) - 0.5) * 2
        ly = (ly / (height - 1) - 0.5) * 2
        lz = (lz / (depth - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    mx,my,mz = torch.meshgrid([lx,ly,lz])
    return torch.stack((mz,my,mx),dim=-1)[None]


def make_regular_grid(deformation_shape,
                      device = torch.device('cpu'),
                      dx_convention = 'pixel'):
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
        # warnings.warn("There is a bug in kornia.create_meshgrid3d, if it is "
        #               "not fixed yet, use dx_convetion=='pixel and adapt with"
        #               "my_torchbox.pixel2square_convention")
        normalized_coordinates = True

    if len(deformation_shape) == 2 :
        H,W = deformation_shape
        return kg.create_meshgrid(H,W,
                                  normalized_coordinates=normalized_coordinates,device=device)
    elif len(deformation_shape) == 3:
        D,H,W = deformation_shape
        return create_meshgrid3d(D,H,W,
                                    normalized_coordinates=normalized_coordinates,device=device)
    elif len(deformation_shape) == 4 or len(deformation_shape) == 5 :
        d = deformation_shape[-1]

        if d ==2:
            _,H,W,_ = deformation_shape
            return kg.create_meshgrid(H,W,
                                      normalized_coordinates=normalized_coordinates,device=device)
        elif d == 3:
            _,D,H,W,_ = deformation_shape
            return create_meshgrid3d(D,H,W,
                                        normalized_coordinates=normalized_coordinates,device=device)



# =================================================================
#             GEOMETRIC HANDELER
# =================================================================

def make_ball_at_shape_center(img,
                              shape_value=None,
                              overlap_threshold=0.1,
                              r_min=None,
                              force_r=None,
                              verbose=False):
    """

    :param img:
    :param shape_value:
    :param overlap_threshold:
    :param r_min:
    :param verbose:
    :return:
    """
    # TODO : documentation
    if len(img.shape) not in [2,4]:
        raise NotImplementedError("Bad shape dude")
    elif len(img.shape) == 4:
        img = img[0,0]
    img = img.cpu()
    shape_value = img.max() if shape_value is None else shape_value
    indexes = (img == shape_value).nonzero(as_tuple=False)

    # La ligne d'avant ordonne naturellement les index.
    min_index_1,max_index_1 = (indexes[0,0],indexes[-1,0])
    # print(min_index_1,max_index_1)
    #Ici il y a plus de travail.
    min_index_2,max_index_2 = (torch.argmin(indexes[:,1]),torch.argmax(indexes[:,1]))
    min_index_2,max_index_2 =(indexes[min_index_2,1],indexes[max_index_2,1])
    centre = (
        (max_index_2 + min_index_2)//2,
        (max_index_1 + min_index_1)//2
    )
    Y,X = torch.meshgrid(torch.arange(img.shape[0]),
                     torch.arange(img.shape[1]))

    def overlap_percentage():
        img_supp = img >0
        overlap = torch.logical_and((img_supp).cpu(), bool_ball).sum()
        seg_sum = (img_supp).sum()
        return overlap/seg_sum

    if force_r is None:
        r = 5 if r_min is None else r_min
        # sum_threshold = 20
        bool_ball = torch.zeros(img.size(),dtype=torch.bool)
        while  overlap_percentage() < overlap_threshold:
            r += max(img.shape)//20
            bool_ball = ((X - centre[0])**2 + (Y - centre[1])**2) < r
            ball = bool_ball[None,None].to(img.dtype)
    else:
        r = force_r
        bool_ball = ((X - centre[0])**2 + (Y - centre[1])**2) < r
        ball = bool_ball[None,None].to(img.dtype)

    if verbose:
        print(f"centre = {centre}, r = {r} and the seg and ball have { torch.logical_and((img > 0).cpu(), bool_ball).sum()} pixels overlapping")
    return ball,centre+(r,)