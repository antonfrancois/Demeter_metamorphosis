import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from torch import is_tensor
import warnings
import vedo
import numpy as np
from numpy import newaxis
from math import prod

import utils.torchbox as tb
from utils.constants import DLT_KW_RESIDUALS
# TODO : Ajouter Residual_norm_function à la liste des imports dans __init__.py
from metamorphosis import Metamorphosis_Shooting,Residual_norm_function


# Utility function
def image_slice(I,coord,dim):
    coord = int(coord)
    if dim == 0:
        return I[coord,:,:]
    elif dim == 1:
        return I[:,coord,:]
    elif dim == 2:
        return I[:,:,coord]

# def line2segmentsCollection(x_coord,y_coord):


def grid_slice(grid,coord,dim):
    """ return a line collection

    :param grid:
    :param coord:
    :param dim:
    :return:
    """


    coord = int(coord)
    if dim == 0:
        return grid[0,coord,:,:,:]
    elif dim == 1:
        return grid[0,:,coord,:,:]
    elif dim == 2:
        return grid[0,:,:,coord,:]


def imshow_3d_slider(image,image_cmap = 'gray'):
    """ Display a 3d image

    :param image: (H,W,D) numpy array or tensor
    :param image_cmap: color map for the plot of the image
    :return: a slider. Note :it is important to store the sliders in order to
    # have them updating

    Exemple :
    # H,W,D = (100,75,50)
    # image = np.zeros((H,W,D))
    # mX,mY,mZ = np.meshgrid(np.arange(H),
    #                           np.arange(W),
    #                           np.arange(D))
    #
    # mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//6
    # mask_carre = (mX > H//6) & (mX < 5*H//6) & (mZ > D//6) & (mZ < 5*D//6)
    # mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//6
    # mask = mask_rond & mask_carre & mask_diamand
    # image[mask] = 1
    # # it is important to store the sliders in order to
    # # have them updating
    # slider = imshow_3d_slider(image)
    # plt.show()
    """
    if len(image.shape) > 5:
        raise TypeError("The image size is expected to be a [D,H,W] array,",
                        " [1,1,D,H,W] tensor object are tolerated.")
    if is_tensor(image) and len(image.shape) == 5:
        warnings.warn("Reshape to image[0,0]: Batch and channel are ignored")
        image = image[0,0].cpu().numpy()
    image = image.T
    # Create the figure and the line that we will manipulate
    fig,ax = plt.subplots(1,3)

    H,W,D = image.shape
    # Define initial parameters
    init_x_coord,init_y_coord,init_z_coord = H //2, W//2, D//2

    kw_image = dict(
        vmin=image.min(),vmax=image.max(),cmap=image_cmap
    )
    img_x = ax[0].imshow( image_slice(image,init_x_coord,dim=0), **kw_image)
    img_y = ax[1].imshow( image_slice(image,init_y_coord,dim=1),origin='lower', **kw_image)
    img_z = ax[2].imshow( image_slice(image,init_z_coord,dim=2),origin='lower', **kw_image)
    ax[0].set_xlabel('X')
    ax[1].set_xlabel('Y')
    ax[2].set_xlabel('Z')

    #add init lines

    line_color = 'white'
    l_x_v = ax[0].axvline(x= init_y_coord,color=line_color)
    l_x_h = ax[0].axhline(y= init_z_coord,color=line_color)
    l_y_v = ax[1].axvline(x= init_z_coord,color=line_color)
    l_y_h = ax[1].axhline(y= init_x_coord,color=line_color)
    l_z_v = ax[2].axvline(x= init_y_coord,color=line_color)
    l_z_h = ax[2].axhline(y= init_x_coord,color=line_color)


    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1,right=.9, bottom=0.25)

    # Make sliders.
    axcolor = 'lightgoldenrodyellow'
    #place them [x_bottom,y_bottom,height,width]
    sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(
        valmin=0,
        valfmt='%0.0f'
    )
    x_slider = Slider(label='x',ax=sl_x,
                      valmax=image.shape[0]-1,valinit=init_x_coord,
                      **kw_slider_args)
    y_slider = Slider(label='y',ax=sl_y,
                      valmax=image.shape[1]-1,valinit=init_y_coord,
                      **kw_slider_args)
    z_slider = Slider(label='z',ax=sl_z,
                      valmax=image.shape[2]-1,valinit=init_z_coord,
                      **kw_slider_args)


    # The function to be called anytime a slider's value changes
    def update(val):
        img_x.set_data(image_slice(image, x_slider.val, 0))
        img_y.set_data(image_slice(image, y_slider.val, 1))
        img_z.set_data(image_slice(image, z_slider.val, 2))

        #update lines
        l_x_v.set_xdata([z_slider.val,z_slider.val])
        l_x_h.set_ydata([y_slider.val,y_slider.val])
        l_y_v.set_xdata([z_slider.val,z_slider.val])
        l_y_h.set_ydata([x_slider.val,x_slider.val])
        l_z_v.set_xdata([y_slider.val,y_slider.val])
        l_z_h.set_ydata([x_slider.val,x_slider.val])
        fig.canvas.draw_idle()

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return x_slider

def _line2segment(x_line,y_line):
    points = np.array([x_line,y_line]).T.reshape(-1,1,2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

def _grid2segments(deformation_slice,horizontal_sampler,vertical_sampler):
    """

    :param defomation: a slice of deformation
    :param samplers (long): an array of n elements from 0 to N respecting n <N
    :return:
    """
    N_H , N_V = horizontal_sampler[-1], vertical_sampler[-1]
    hori_segments = np.zeros((len(horizontal_sampler) * (N_V+1),2,2))
    vert_segments = np.zeros((len(vertical_sampler)   * (N_H+1),2,2))

    print(deformation_slice.shape)
    print(horizontal_sampler)
    for i,nh in enumerate(horizontal_sampler):
        hori_segments[int(i*(N_V)):int(i*(N_V)+(N_V)),:,:] = _line2segment(
            deformation_slice[nh,:,0],deformation_slice[nh,:,1]
        )
    for i,nv in enumerate(vertical_sampler):
        vert_segments[int(i*(N_H)):int(i*(N_H))+(N_H),:,:] = _line2segment(
            deformation_slice[:,nv,0],deformation_slice[:,nv,1]
        )
    return np.concatenate([hori_segments,vert_segments],axis=0)


def gridDef_3d_slider(deformation,
                      add_grid :bool = False,
                      n_line :int = 20,
                      dx_convention = 'pixel'):
    """ Display a 3d grid with sliders

    :param image: (H,W,D) numpy array or tensor
    :param image_cmap: color map for the plot of the image
    :return: a slider. Note :it is important to store the sliders in order to
    # have them updating

    Exemple :
    # H,W,D = (100,75,50)
    # image = np.zeros((H,W,D))
    # mX,mY,mZ = np.meshgrid(np.arange(H),
    #                           np.arange(W),
    #                           np.arange(D))
    #
    # mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//6
    # mask_carre = (mX > H//6) & (mX < 5*H//6) & (mZ > D//6) & (mZ < 5*D//6)
    # mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//6
    # mask = mask_rond & mask_carre & mask_diamand
    # image[mask] = 1
    # # it is important to store the sliders in order to
    # # have them updating
    # slider = imshow_3d_slider(image)
    # plt.show()
    """

    if is_tensor(deformation) and len(deformation.shape) == 5\
            and deformation[-1] == 3:
        deformation = deformation.numpy()
    t,D,H,W,_ = deformation.shape
    if t >1:
        warnings.warn("Only deformation with one time step are supported,"
                      " only first entry had been taken into account.")
        deformation = deformation[0][newaxis]
    # deformation = deformation.T


    # Define initial coordinates
    init_d_coord, init_h_coord,init_w_coord = D//2, H//2, W//2

    # kw_image = dict(
    #     vmin=image.min(),vmax=image.max(),cmap=image_cmap
    # )

    d_sampler = np.linspace(0,D-1,n_line,dtype=np.long)
    h_sampler = np.linspace(0,H-1,n_line,dtype=np.long)
    w_sampler = np.linspace(0,W-1,n_line,dtype=np.long)

    segments_D = _grid2segments(deformation[0,init_d_coord,:,:,1:],h_sampler,w_sampler)
    lc_1 = LineCollection(segments_D,colors='black',linewidths=1)

    segments_H = _grid2segments(deformation[0,:,init_h_coord,:,:2],d_sampler,w_sampler)
    lc_2 = LineCollection(segments_H,colors='black',linewidths=1)

    segments_W = _grid2segments(deformation[0,:,:,init_d_coord,::2],d_sampler,h_sampler)
    lc_3 = LineCollection(segments_W,colors='black',linewidths=1)


    # Create the figure and the line that we will manipulate
    fig,ax = plt.subplots(1,3)
    line_d = ax[0].add_collection(lc_1)
    ax[0].autoscale()
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("W")

    line_h = ax[1].add_collection(lc_2)
    ax[1].autoscale()
    ax[1].set_xlabel("D")
    ax[1].set_ylabel("H")

    line_w = ax[2].add_collection(lc_3)
    ax[2].autoscale()
    ax[2].set_xlabel("W")
    ax[2].set_ylabel("D")


    #add init lines
    line_kwargs=dict(
        color = 'red',
        linestyle ='--'
    )

    l_x_v = ax[0].axvline(x= 2*init_h_coord/H-1,**line_kwargs)
    l_x_h = ax[0].axhline(y= 2*init_w_coord/W-1,**line_kwargs)
    l_y_v = ax[1].axvline(x= 2*init_w_coord/W-1,**line_kwargs)
    l_y_h = ax[1].axhline(y= 2*init_d_coord/D-1,**line_kwargs)
    l_z_v = ax[2].axvline(x= 2*init_h_coord/H-1,**line_kwargs)
    l_z_h = ax[2].axhline(y= 2*init_d_coord/D-1,**line_kwargs)


    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1,right=.9, bottom=0.25)

    # Make sliders.
    axcolor = 'lightgoldenrodyellow'
    #place them [x_bottom,y_bottom,height,width]
    sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(
        valmin=0,
        valfmt='%0.0f'
    )
    x_slider = Slider(label='D',ax=sl_x,
                      valmax=D-1,valinit=init_d_coord,
                      **kw_slider_args)
    y_slider = Slider(label='H',ax=sl_y,
                      valmax=H-1,valinit=init_h_coord,
                      **kw_slider_args)
    z_slider = Slider(label='W',ax=sl_z,
                      valmax=W-1,valinit=init_w_coord,
                      **kw_slider_args)


    # The function to be called anytime a slider's value changes
    def update(val):
        # print(x_slider.val,type(x_slider.val))
        # print(deformation[0,x_slider.val,:,:,1:].shape)
        segments_D =_grid2segments(deformation[0,int(x_slider.val),:,:,1:],
                                   h_sampler,w_sampler)
        lc_1.set_segments(segments_D)

        segments_H = _grid2segments(deformation[0,:,int(y_slider.val),:,:2],
                                   d_sampler,w_sampler)
        lc_2.set_segments(segments_H)

        segments_W = _grid2segments(deformation[0,:,:,int(z_slider.val),::2],
                                    d_sampler,h_sampler)
        lc_3.set_segments(segments_W)

        #update lines
        l_x_v.set_xdata([z_slider.val,z_slider.val])
        l_x_h.set_ydata([y_slider.val,y_slider.val])
        l_y_v.set_xdata([z_slider.val,z_slider.val])
        l_y_h.set_ydata([x_slider.val,x_slider.val])
        l_z_v.set_xdata([y_slider.val,y_slider.val])
        l_z_h.set_ydata([x_slider.val,x_slider.val])
        fig.canvas.draw_idle()

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return x_slider


# ======================================================================
#
#        ██    ██ ███████ ██████   ██████
#        ██    ██ ██      ██   ██ ██    ██
#        ██    ██ █████   ██   ██ ██    ██
#         ██  ██  ██      ██   ██ ██    ██
#          ████   ███████ ██████   ██████
#
#
# ======================================================================

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
vedo.embedWindow('ipyvtk')
vedo.settings.useDepthPeeling = True  # if you use transparency <1
class deformation_grid3D_vedo:

    def __init__(self,show_kwargs=None,
                    max_resolution=30,
                    color='yellow',
                    alpha=0.5,
                    show_all_surf = False,
                    addCutterTool=False):
                    # warnings_verbose = False):
        """ When you initialize the class you have the opportunity
        to select some constants that will define the plot.

        :param max_resolution: int or 3sized tuple of int. Maximum number of
        mesh summits in each dimension. If type(max_resolution) == int then
        its value will be the maximum in each dimension.
        :param color: Color of the surfaces
        :param alpha: alpha value of the surfaces
        :param show_all_surf: If true show the faces of the surface only if the points
        where transported, else it show all faces on the wireframe.
        """
        if show_kwargs is None:
            self.show_kwargs = dict(
                axes=8, bg2='lightblue', viewup='x', interactive=True
            )
        else:
            self.show_kwargs = show_kwargs
        self.max_resolution = max_resolution
        self.color = color
        self.alpha = alpha
        self.show_all_surf = show_all_surf
        self.addCutterTool = addCutterTool
        self.plot = vedo.Plotter()
        # if warnings_verbose:
        #     from warnings import simplefilter
        #     simplefilter(action='ignore', category=DeprecationWarning)

    def _make_all_faces(self,coord,dim_1,dim_2):
        faces = np.zeros((dim_1*dim_2,4))
        for d in range(dim_1-1):
            faces[d*dim_2:d*dim_2 + (dim_2-1),:] = np.stack(
                [coord[d,:-1],coord[d,1:],coord[d+1,1:],coord[d+1,:-1]],
                axis=1
            )
        return faces

    def _make_moved_faces(self,bool_slice,coord,dim_1,dim_2):
        faces = []
        for u in range(dim_1 - 1):
            for v in range(dim_2 -1):
                coord_moved = bool_slice[u:u+2,v:v+2].sum(dtype=bool)
                if coord_moved:
                    faces.append([coord[u,v],
                                  coord[u,v+1],
                                  coord[u+1,v+1],
                                  coord[u+1,v]])
        return faces

    def _construct_surface_coord(self, deformation_slice,reg_grid_slice = None):
        """ takes a deformation slice of the selected dimensions and
        prepare it to be plotted by vedo

        :param defomation_slice: [dim_1,dim_2,3]
        :return: an array with point coordinates and face links.
        """
        if deformation_slice.shape[-1] != 3:
            raise TypeError("deformation slice must have shape (dim_1,dim_2,3) got "+
                            str(deformation_slice.shape))
        dim_1,dim_2,_ = deformation_slice.shape

        points = deformation_slice.reshape((dim_1*dim_2,3))

        coord = np.arange(dim_1*dim_2).reshape((dim_1,dim_2))

        if reg_grid_slice is None:
            faces = self._make_all_faces(coord,dim_1,dim_2)
        else:
            bool_slice = (deformation_slice - reg_grid_slice).abs() > 0
            faces = self._make_moved_faces(bool_slice,coord,dim_1,dim_2)

        return [points,faces]

    def _slicer(self, deformation, dim, ind):
        r""" Slice the deformation in the given index at the given dimension.
        It also subsample the deformation at the max_resolution set by the user
        in init, default value is 30.

        :param deformation: [1,D,H,W,3] numpy array
        :param dim: int \in \{0,1,2\}
        :param ind: int index of deformation matrix
        :return: sliced deformation
        """
        _,D,H,W,_ =deformation.shape
        if type(self.max_resolution) == np.int:
            self.max_resolution = (self.max_resolution,)*3
        if dim == 0 :
            h_sampler = np.linspace(0,H-1,min(H,self.max_resolution[1]),dtype=np.long)
            w_sampler = np.linspace(0,W-1,min(W,self.max_resolution[2]),dtype=np.long)
            return deformation[0,ind,h_sampler,:,:][:,w_sampler,:]
        elif dim == 1:
            d_sampler = np.linspace(0,D-1,min(D,self.max_resolution[0]),dtype=np.long)
            w_sampler = np.linspace(0,W-1,min(W,self.max_resolution[2]),dtype=np.long)
            return deformation[0,d_sampler,ind,:,:][:,w_sampler,:]
        elif dim == 2:
            d_sampler = np.linspace(0,D-1,min(D,self.max_resolution[0]),dtype=np.long)
            h_sampler = np.linspace(0,H-1,min(H,self.max_resolution[1]),dtype=np.long)
            return deformation[0,d_sampler,:,ind,:][:,h_sampler,:]
        else:
            raise IndexError("dim has to be {0,1,2} got "+str(dim))

    def make_surface_mesh(self,deformation,dim,n_surf):
        surf_indexes = np.linspace(0,deformation.shape[dim+1]-1,n_surf,
                                   dtype=np.long)
        surfaces = [None] * n_surf
        for i,ind in enumerate(surf_indexes):
            if self.show_all_surf:
                reg_grid_slice = None
            else:
                reg_grid_slice = self._slicer(self.reg_grid,dim,ind)

            mesh_pts = self._construct_surface_coord(
                self._slicer(deformation,dim,ind),
                reg_grid_slice
            )
            surfaces[i] = vedo.Mesh(mesh_pts)

        surf_meshes = vedo.merge(surfaces).computeNormals()
        surf_meshes.lineWidth(0.1).alpha(self.alpha).c(self.color).lighting('off')
        return surf_meshes

    def time_slider(self,widget,event):
        value = widget.GetRepresentation().GetValue()
        for s,surfs in enumerate(self.all_time_surf):
            if s == int(value):
                self.all_time_surf[s].on()
            else:
                self.all_time_surf[s].off()

        self.plot.render()

    def buttonfunc(self):
        name_file = 'last_deformation_saved'
        self.plot.export(name_file+".npz") # vedo 3d format, use command line "vedo scene.npz"
        self.plot.screenshot(name_file+".png",scale=1)
        vedo.printc("save exported "+name_file+".npz and screenshot to "
                    +name_file+".png", c='g')

    def __call__(self,deformation,
                dim=0,
                n_surf=5,
                add_grid= False,
                dx_convention='pixel'):
        if len(deformation.shape) != 5 and deformation.shape[-1] !=3:
            raise TypeError("deformation shape must be of the form [1,D,H,W,3]",
                            "got array of dim "+ str(deformation.shape))
        if deformation.shape[0] > 1:
            # deformation = deformation[0][None]
            # print("Warning, only first Batch dimension will be considered")

            self.plot.addSlider2D(self.time_slider,
                                  xmin=0,xmax=deformation.shape[0]-1,value=0,
                                  pos=[(0.35,0.06),(0.65,0.06)],title="time")
        # _,D,H,W,_ =deformation.shape
        self.reg_grid = tb.make_regular_grid(deformation.shape,dx_convention=dx_convention)
        if add_grid:
            deformation = self.reg_grid + deformation

        self.all_time_surf = []
        for t in range(deformation.shape[0]):
            surf_mesh = self.make_surface_mesh(deformation[t][None],dim,n_surf)
            elevations = surf_mesh.points()[:,0]
            surf_mesh.cmap("YlOrBr_r", elevations).addScalarBar("Dimension")
            surf_mesh.off()  # switch off new meshes of
            self.all_time_surf.append(surf_mesh)
        self.all_time_surf[0].on()
        # plots

        self.bu = self.plot.addButton(self.buttonfunc,
                                      pos=(0.15, 0.05),  # x,y fraction from bottom left corner
                                      states=["export/save"],
                                      c=["w"],
                                      bc=["dg"],  # colors of states
                                      font="times",   # arial, courier, times
                                      size=25,
                                      bold=True,
                                      italic=False,
        )
        # add some coloring

        self.plot.show(self.all_time_surf,
                       vedo.Points(surf_mesh.points(), c='black', r=1,alpha=.5),
                       **self.show_kwargs)
         # axes=8, bg2='lightblue', viewup='x', interactive=False)
        
        if self.addCutterTool:
            self.plot.addCutterTool(self.all_time_surf,'box') # comment this line for using the class in jupyter notebooks
        vedo.interactive()  # stay interactive
        return



def deformation_grid3D_surfaces(deformation,dim =0,n_surf=10,add_grid = False,
                                dx_convention= 'pixel'):
    """ function API for `deformation_grid3D_vedo` class

    :param deformation: [1,D,H,W,3] numpy array
    :param dim: Dimension you want to show the surfaces along.
    :param n_surf: Numbers of surfaces you want to plot.
    :return:
    """

    return deformation_grid3D_vedo()(deformation,dim=dim,n_surf=n_surf,
        add_grid =add_grid,dx_convention=dx_convention)


def image_slice_vedo(image,interpolate= False,bg_color=('white','lightblue'),close=True):

    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()
    if len(image.shape) == 5:
        image = image[0,0]
    vol = vedo.Volume(image)
    vol.addScalarBar3D()

    plot = vedo.applications.SlicerPlotter( vol,
                     bg=bg_color[0], bg2=bg_color[1],
                     cmaps=("bone","bone_r","jet","Spectral_r","hot_r","gist_ncar_r"),
                     useSlider3D=False,
                     # map2cells=not interpolate, #buggy
                     clamp=False
                   )
    vedo.interactive()
    if close:
        plot.show().close()
    else:
        return plot


def _slice_rotate_image_(image,dim,index,alpha=1,time=None):
    """ Utilitary class for slicing and positioning images at their rightful locations

    :param image:
    :param dim:
    :param index:
    :param alpha:
    :param time:
    :return:
    """
    if time is None:
        T = image.shape[0] - 1
    elif time > image.shape[0] - 1:
        T = min(time,image.shape[0]-1)
    else:
        T = time
    if dim == 2:
        img = image[T,index,::-1]*255
        pic = vedo.Picture(img,flip=(len(img.shape)== 2))
        return pic.z(index).alpha(alpha)
    elif dim == 1:
        img = image[T,::-1,index]*255
        pic = vedo.Picture(img,flip=(len(img.shape)== 2))
        return pic.rotateX(90).y(index).alpha(alpha)
    elif dim == 0:
        img = image[T,::-1,::1,index]*255
        pic = vedo.Picture(img,flip=(len(img.shape)== 2))
        return pic.rotateX(90).rotateZ(90).x(index).alpha(alpha)
    else:
        raise ValueError(' dim must be an int equal to 0,1 or 2')

def make_cmp_image(img_1,img_2):
    # if T is different, then the lowest is equal to 1
    print(f"make_cmp_image >> img1:{img_1.shape} , img2:{img_2.shape}")
    if img_1.shape[0] > img_2.shape[0]:
        img_2 = np.repeat(img_2,img_1.shape[0],axis=0)
    elif img_1.shape[0] < img_2.shape[0]:
        img_1 = np.repeat(img_1,img_2.shape[0],axis=0)
    print(f"make_cmp_image.2 >> img1:{img_1.shape} , img2:{img_2.shape}")
    return np.concatenate(
        (img_1[:,:,:,:,None],
         img_2[:,:,:,:,None],
         np.zeros(img_1.shape+(1,))),
        axis=-1
        )

class compare_3D_images_vedo:

    def __init__(self,image1,image2,alpha=0.8,close = True):

        # TODO : reduire la taille de l'image
        self.image1 = self._prepare_image_(image1)
        self.image2 = self._prepare_image_(image2)
        self._check_dimensions_()
        _,D,H,W = self.image1.shape
        T = max(self.image1.shape[0],self.image2.shape[0])
        if T == 1: T = 0

        self.flag_cmp = False # is comparison with target image activated or not
        # stack images on different color channels
        self.cmp_image = make_cmp_image(self.image1,self.image2)

        self.flag_def = False # show and update deformation flow
        self.alpha = alpha if alpha >= 0 and alpha <= 1 else 1
        vedo.settings.immediateRendering = False  # faster for multi-renderers
        bg_s= [(57,62,58), (82,87,83)]
        custom_shape = [
            dict(bottomleft=(0.0, 0.0), topright=(0.5, 1), bg=bg_s[0], bg2=bg_s[1]),  # ren0
            dict(bottomleft=(0.5, 0.0), topright=(1, 1), bg=bg_s[0], bg2=bg_s[1]),  # ren1
            dict(bottomleft=(0.4, 0), topright=(0.6, 0.2), bg="white"),  # ren2
        ]
        self.plotter = vedo.Plotter(shape=custom_shape,#N=2,
                                    bg=bg_s[0],bg2=bg_s[1],
                                    screensize=(1200,1000),
                                    interactive=False
                                    )

        vol_1 = vedo.Volume(self.image1[0]).addScalarBar3D()
        vol_2 = vedo.Volume(self.image2[0]).addScalarBar3D()
        box1 = vol_1.box().wireframe().alpha(0)
        box2 = vol_2.box().wireframe().alpha(0)
        self.plotter.show(box1,at=0,viewup='x',axes=7)
        # self.plotter.interactive = True
        self.plotter.show(box2,at=1,viewup='x',axes=7)
        self.plotter.addInset(vol_1,at=0,pos=1,
                              c='w',draggable=True)
        self.plotter.addInset(vol_2,at=1,pos=2,
                              c='w',draggable=True)

        # ===== Image 1 initialisation =============
        self.actual_t,self._d,self._h,self._w = (T, 0, 0, W//2)
        pic_1_D = _slice_rotate_image_(self.image1,0,self._w,self.alpha)
        self.plotter.renderers[0].AddActor(pic_1_D)
        # pic_1_H = _slice_rotate_image_(self.image1,1,self._h,self.alpha)
        # self.plotter.renderers[0].AddActor(pic_1_H)
        # pic_1_W = _slice_rotate_image_(self.image1,2,self._d,self.alpha)
        # self.plotter.renderers[0].AddActor(pic_1_W)

        # ===== Image 2 initialisation =============
        pic_2_D = _slice_rotate_image_(self.image2,0,self._w,self.alpha)
        self.plotter.renderers[1].AddActor(pic_2_D)

        self.visibles = [[pic_1_D,None,None],
                         [pic_2_D,None,None],
                         None]
        self.plotter.show(self.visibles[0],at=0)
        self.plotter.show(self.visibles[1],at=1)



        # Add 2D sliders
        cx, cy, cz, ct, ch = 'dr', 'dg', 'db', 'fdf1', (0.3,0.3,0.3)
        if np.sum(self.plotter.renderer.GetBackground()) < 1.5:
            cx, cy, cz, ct = 'lr', 'lg', 'lb', 'sdfs'
            ch = (0.8,0.8,0.8)
        # X,Y,Z dimentional sliders
        self.plotter.renderer = self.plotter.renderers[2]
        x_m,x_p,y,y_s = 0.01,0.19,0.02,0.04
        vscale = 6
        slid_d = self.plotter.addSlider2D(self._sliderfunc_d,
                    xmin=0,xmax= W,value=self._d,
                    title='X', titleSize=3,
                    pos=[(x_m, y + 2*y_s), (x_p, y + 2*y_s)],
                    showValue=True,
                    c=cx,
        )
        self._set_slider_size(slid_d,vscale)
        slid_h = self.plotter.addSlider2D(self._sliderfunc_h,
                    xmin=0,xmax= H,value=self._h,
                    title='Y', titleSize=3,
                    pos=[(x_m, y + y_s), (x_p, y + y_s)],
                    showValue=True,
                    c=cy,
        )
        self._set_slider_size(slid_h,vscale)
        slid_w = self.plotter.addSlider2D(self._sliderfunc_w,
                    xmin=0,xmax= D,value=self._w,
                    title='Z', titleSize=3,
                    pos=[(x_m,y), (x_p,y)],
                    showValue=True,
                    c=cz,
        )
        self._set_slider_size(slid_w,vscale)
        # TIME dimentional slider
        if self._is_1_temporal or self._is_2_temporal:
            slid_t = self.plotter.addSlider2D(self._sliderfunc_t,
                        xmin=0,xmax= T,value=self.actual_t,
                        title='T', titleSize=3,
                        pos=[(x_m, y + 3.5*y_s), (x_p, y + 3.5*y_s)],
                        showValue=True,
                        c=ct,
            )
            self._set_slider_size(slid_t,vscale)

        self.plotter.renderer = self.plotter.renderers[1]
        self._bu_cmp = self.plotter.addButton(self._button_func_,
                    pos=(0.27, 0.95),
                    states=['compare','stop comparing'],
                    # c=["db"]*len(cmaps),
                    # bc=["lb"]*len(cmaps),  # colors of states
                    size=14,
                    bold=True,
                )



        hist1 = vedo.pyplot.cornerHistogram(self.image1, s=0.2,
                               bins=25, logscale=1, pos=(0.03, 0.01),
                               c=ch, bg=ch, alpha=0.7)
        hist2 = vedo.pyplot.cornerHistogram(self.image2, s=0.2,
                               bins=25, logscale=1, pos=(0.8, 0.01),
                               c=ch, bg=ch, alpha=0.7)
        self.plotter.show(hist1,at=0)
        self.plotter.show(hist2,at=1)
        if close:
            self.plotter.show(interactive=True).close()


    def show_deformation_flow(self,deformation,at,step = None):
        """

        :param deformation: grid like numpy array
        :param step:
        :return:
        """
        # check and downsize deformation
        # reg_grid = tb.make_regular_grid(deformation[0][None].shape,
        #                                 dx_convention='pixel')
        self.flag_def = True
        if step is None:
            step = max(deformation.shape[1:-1])
        print("Adding flow :")
        self.deformation_stepped = deformation[:,::step,::step,::step]
        # reg_grid = reg_grid[:,::step,::step,::step]
        self.visibles[2] = self._make_deformation_flow_()
        self.plotter.show(self.visibles[2],at=at)

    def _make_deformation_flow_(self):
        fT,D,H,W,d = self.deformation_stepped.shape

        lines = self.deformation_stepped.reshape(fT,H*W*D,d)
        length = ((lines[1:] - lines[:-1])**2).sum(dim=-1).sqrt().sum(dim=0)
        colors = (length/length.max()).numpy()
        med_length = np.median(colors)

        lines_col = []
        for i in range(D*H*W):
            if med_length < colors[i]:
                t = min(max(self.actual_t,0),fT-1)
                lines_col.append(vedo.Line(lines[:t,i,:].numpy(),
                                            c=(colors[i],0,1-colors[i]),
                                            lw=3)
                                 )
        return vedo.merge(lines_col)


    def _update_image_along(self,image,renderer,dim,time,index):
        if self.flag_cmp and renderer ==1:
            image = self.cmp_image
        pic = _slice_rotate_image_(image,dim,index, time=time,alpha=self.alpha)
        self.plotter.renderers[renderer].AddActor(pic)
        self.visibles[renderer][dim] = pic

    def _sliderfunc_d(self,widget, event):
        self._d = int(widget.GetRepresentation().GetValue())
        self.plotter.renderers[0].RemoveActor(self.visibles[0][0])
        self.plotter.renderers[1].RemoveActor(self.visibles[1][0])
        if self._d and self._d<self.image1.shape[1]:
            self._update_image_along(self.image1,0,0,self.actual_t,self._d)
            self._update_image_along(self.image2,1,0,self.actual_t,self._d)

    def _sliderfunc_h(self,widget, event):
        self._h = int(widget.GetRepresentation().GetValue())
        self.plotter.renderers[0].RemoveActor(self.visibles[0][1])
        self.plotter.renderers[1].RemoveActor(self.visibles[1][1])
        if self._h and self._h<self.image1.shape[2] :
            self._update_image_along(self.image1,0,1,self.actual_t,self._h)
            self._update_image_along(self.image2,1,1,self.actual_t,self._h)

    def _sliderfunc_w(self,widget, event):
        self._w = int(widget.GetRepresentation().GetValue())
        self.plotter.renderers[0].RemoveActor(self.visibles[0][2])
        self.plotter.renderers[1].RemoveActor(self.visibles[1][2])
        if self._w and self._w<self.image1.shape[3] :
            self._update_image_along(self.image1,0,2,self.actual_t,self._w)
            self._update_image_along(self.image2,1,2,self.actual_t,self._w)

    def _sliderfunc_t(self,widget,event):
        if widget is not None:
            self.actual_t = int(widget.GetRepresentation().GetValue())
        my_nor = not (self._is_1_temporal or self._is_2_temporal)
        if self._is_1_temporal or my_nor:
            if self._d and self._d<self.image1.shape[1]:
                self.plotter.renderers[0].RemoveActor(self.visibles[0][0])
                self._update_image_along(self.image1,0,0,self.actual_t,self._d)
            if self._h and self._h<self.image1.shape[2] :
                self.plotter.renderers[0].RemoveActor(self.visibles[0][1])
                self._update_image_along(self.image1,0,1,self.actual_t,self._h)
            if self._w and self._w<self.image1.shape[3] :
                self.plotter.renderers[0].RemoveActor(self.visibles[0][2])
                self._update_image_along(self.image1,0,2,self.actual_t,self._w)
        if self._is_2_temporal or my_nor:
            if self._d and self._d<self.image2.shape[1]:
                self.plotter.renderers[1].RemoveActor(self.visibles[1][0])
                self._update_image_along(self.image2,1,0,self.actual_t,self._d)
            if self._h and self._h<self.image2.shape[2] :
                self.plotter.renderers[1].RemoveActor(self.visibles[1][1])
                self._update_image_along(self.image2,1,1,self.actual_t,self._h)
            if self._w and self._w<self.image2.shape[3] :
                self.plotter.renderers[1].RemoveActor(self.visibles[1][2])
                self._update_image_along(self.image2,1,2,self.actual_t,self._w)
        if self.flag_def:
            self.plotter.renderers[1].RemoveActor(self.visibles[2])
            self.visibles[2] = self._make_deformation_flow_()
            self.plotter.renderers[1].AddActor(self.visibles[2])

    def _button_func_(self):
        self._bu_cmp.switch()
        self.flag_cmp= not self.flag_cmp
        self._sliderfunc_t(None,None)

    def _set_slider_size(self,slider,scale):
        sliderRep = slider.GetRepresentation()
        sliderRep.SetSliderLength(0.003 * scale)  # make it thicker
        sliderRep.SetSliderWidth(0.025 * scale)
        sliderRep.SetEndCapLength(0.001 * scale)
        sliderRep.SetEndCapWidth(0.025 * scale)
        sliderRep.SetTubeWidth(0.0075 * scale)



    def _prepare_image_(self,image):
        if torch.is_tensor(image):
            if len(image.shape) != 5:
                raise ValueError('If the image is a torch tensor'
                                     'it have to have shape [T,1,D,H,W]'
                                     f'got {image.shape}')
            else:
                image = image[:,0].cpu().numpy()
        elif isinstance(image, np.ndarray):
            if len(image.shape) != 3:
                raise ValueError(f'If the image is a numpy array it must be 3D of shape [D,H,W] got {str(image.shape)}')
            else:
                image = image[None]
        else:
            raise AttributeError('image must be numpy array or torch tensor')
        return image

    def _check_dimensions_(self):
        if self.image1.shape[1:] != self.image2.shape[1:]:
            raise ValueError('Both image have to have same dimensions'
                             'image1.shape = '+ str(self.image1.shape)+' and '
                             'image2.shape = '+ str(self.image2.shape)+'.')
        # check if there are temporal images
        T1,T2 = self.image1.shape[0],self.image2.shape[0]
        self._is_1_temporal = (T1 >1)
        self._is_2_temporal = (T2 >1)
        if self._is_1_temporal or self._is_2_temporal:
            if T1 != T2 and min(T1,T2) > 1:
                raise ValueError('If both images are temporal'
                                 'they must have same time dimensions'
                                 'got image1.shape ='+ self.image1.shape+' and '
                                 'image2.shape ='+ self.image2.shape+'.')

class Visualize_geodesicOptim:


    def __init__(self,geoShoot : Metamorphosis_Shooting,
                 alpha = 0.8,
                 convergence_plot=True,
                 close = True):
        # TODO : check if geoShoot is really a subclass of Optimize_metamorphosis
        self.gs = geoShoot.cpu()

        # Get values to show later
        self.image = self.gs.mp.image_stock[:,0].numpy()
        print(f"Visu_geodesic min : {self.image.min()}, max: {self.image.max()}")
        # self.image = (self.image - self.image.min()) /(self.image - self.image.min()).max()
        self.image = np.clip(self.image,a_min=0,a_max=1)
        T,D,H,W = self.image.shape
        res_np = self.gs.mp.residuals_stock.numpy()
        self.res_max_abs = res_np.__abs__().max()
        self.residual = [vedo.Volume(res_t).addScalarBar3D() for res_t in res_np]

        # flag for discriminating against different kinds of Optimizers
        try:
            isinstance(self.gs.mp.rf,Residual_norm_function)
            self.is_pure_meta = False
        except AttributeError:
            self.is_pure_meta = True

        # Booleans for buttons switches
        self.flag_cmp_target = False # is comparison with target image activated or not
        self.flag_cmp_source = False
        self.flag_cmp_mask = False   #is comparison with source image activated. (only for Weighted metamoprhoshis)
        self.show_res = False # If True show image else show residual

        # stack images on different color channels
        self.cmp_image_target = make_cmp_image(self.image,self.gs.target[:,0].numpy())
        self.cmp_image_source = make_cmp_image(self.image,self.gs.source[:,0].numpy())
        if not self.is_pure_meta:
            self.cmp_image_mask = make_cmp_image(self.image,self.gs.mp.rf.mask[:,0].numpy())

        self.alpha = alpha if alpha >= 0 and alpha <= 1 else 1
        vedo.settings.immediateRendering = True

        bg_s= [(57,62,58), (82,87,83)]
        self.plotter = vedo.Plotter(N=1,bg=bg_s[0],bg2=bg_s[1],
                                    screensize=(1200,1000),
                                    interactive=False
                                    )
        vol = vedo.Volume(self.image[-1]).addScalarBar3D()
        box = vol.box().wireframe().alpha(0)
        self.plotter.show(box,self.gs.__repr__(),at=0,viewup='x',axes=7)
        # self.plotter.interactive = True
        self.plotter.addInset(vol,at=0,pos=2,
                              c='w',draggable=True)

        # ===== Image 1 initialisation =============
        self.actual_t,self._d,self._h,self._w = T,W//2,H//2,D//2
        pic_1_D = _slice_rotate_image_(self.image,0,self._d,self.alpha)
        self.plotter.renderer.AddActor(pic_1_D)

        self.visibles = [[pic_1_D,None,None], # Is used for the tree panels of images
                         [None,None,None], # Is used for the tree panels of residuals
                         None] # Is used for the flow visualisation
        self.plotter.show(self.visibles[0][1],at=0)

        # ==== show convergence ==========
        plots  = self.gs.get_total_cost()
        conv_points = np.stack([np.arange(max(plots.shape)),plots]).T
        plot = vedo.pyplot.cornerPlot(conv_points,pos = 3,c='w',s=0.175)
        plot.GetXAxisActor2D().SetFontFactor(0.3)
        plot.GetYAxisActor2D().SetLabelFactor(0.3)
        self.plotter.show(plot)

        # ==== Slider handeling =========
        cx, cy, cz, ct, ch = 'dr', 'dg', 'db', 'fdf1', (0.3,0.3,0.3)
        if np.sum(self.plotter.renderer.GetBackground()) < 1.5:
            cx, cy, cz, ct = 'lr', 'lg', 'lb', 'sdfs'
            ch = (0.8,0.8,0.8)
        # X,Y,Z dimentional sliders
        x_m,x_p,y,y_s = 0.8,0.9,0.02,0.02
        vscale = 6
        slid_d = self.plotter.addSlider2D(self._sliderfunc_d,
                    xmin=0,xmax= W,value=self._w,
                    title='X', titleSize=1,
                    pos=[(x_m, y + 2*y_s), (x_p, y + 2*y_s)],
                    showValue=True,
                    c=cx,
        )
        self._set_slider_size(slid_d,vscale)
        slid_h = self.plotter.addSlider2D(self._sliderfunc_h,
                    xmin=0,xmax= H,value=self._h,
                    title='Y', titleSize=1,
                    pos=[(x_m, y + y_s), (x_p, y + y_s)],
                    showValue=True,
                    c=cy,
        )
        self._set_slider_size(slid_h,vscale)
        slid_w = self.plotter.addSlider2D(self._sliderfunc_w,
                    xmin=0,xmax= D,value=self._d,
                    title='Z', titleSize=1,
                    pos=[(x_m,y), (x_p,y)],
                    showValue=True,
                    c=cz,
        )
        self._set_slider_size(slid_w,vscale)
        # TIME dimentional slider
        slid_t = self.plotter.addSlider2D(self._sliderfunc_t,
                    xmin=0,xmax= T,value=self.actual_t,
                    title='T', titleSize=1,
                    pos=[(x_m, y + 3.5*y_s), (x_p, y + 3.5*y_s)],
                    showValue=True,
                    c=ct,
        )
        self._set_slider_size(slid_t,vscale)


        # ======== BUTTONS ============
        self._bu_residuals = self.plotter.addButton(self._button_residuals_,
            pos=(0.27, 0.0),
            states=['image','residual'],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_cmp_target = self.plotter.addButton(self._button_cmp_target_,
            pos=(0.37, 0.0),
            states=['Compare target OFF','Compare target ON'],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_cmp_source = self.plotter.addButton(self._button_cmp_source_,
            pos=(0.47, 0.0),
            states=['Compare source OFF','Compare source ON'],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        if not self.is_pure_meta:
            self._bu_cmp_mask = self.plotter.addButton(self._button_cmp_mask_,
                pos=(0.57, 0.0),
                states=['Compare mask OFF','Compare mask ON'],
                # c=["db"]*len(cmaps),
                # bc=["lb"]*len(cmaps),  # colors of states
                size=14,
                bold=True,
            )

        if close:
            self.plotter.show(interactive=True).close()

    def _update_image_along(self,dim,time,index):
        if self.flag_cmp_target:
            image = self.cmp_image_target
        elif self.flag_cmp_source:
            image = self.cmp_image_source
        elif self.flag_cmp_mask:
            image = self.cmp_image_mask
        else:
            image = self.image

        # image = self.cmp_image if self.flag_cmp_target else self.image
        # self.plotter.renderer.RemoveActor(self.visibles[0][dim])
        # self.visibles[1][dim] = None # remove residual from visuals (may be useless sometime)
        pic = _slice_rotate_image_(image,dim,index, time=time)
        self.plotter.renderer.AddActor(pic)
        self.visibles[0][dim] = pic

    def _update_residuals_along(self,dim,time,index):
        # self.plotter.renderer.RemoveActor(self.visibles[1][dim])
        # self.visibles[0][dim] = None # remove image from visuals (may be useless sometime)
        if dim == 0: res = self.residual[time-1].xSlice(index)
        elif dim == 1: res = self.residual[time-1].ySlice(index)
        elif dim == 2: res = self.residual[time-1].zSlice(index)
        else: raise ValueError("dim must be int in {0,1,2}")
        la, ld = 0.7, 0.3 #ambient, diffuse
        res.alpha(self.alpha).lighting('', la, ld, 0)
        res.cmap(DLT_KW_RESIDUALS['cmap'], vmin=-self.res_max_abs, vmax=self.res_max_abs)
        if index: # and index < self.residual.shape[dim+1]:
            self.plotter.renderer.AddActor(res)
            self.visibles[1][dim] = res

    def _update_along_(self,dim,time,index):
        if self.show_res:
            self._update_residuals_along(dim,time,index)
        else:
            self._update_image_along(dim,time,index)

    def _remove_actor(self,obj,dim):
            self.plotter.renderer.RemoveActor(self.visibles[obj][dim])
            self.visibles[obj][dim] = None

    def _sliderfunc_d(self,widget, event):
        self._d = int(widget.GetRepresentation().GetValue())
        # vis_ind = 0 if self.show_res else 1
        self._remove_actor(0,0)
        self._remove_actor(1,0)
        if self._d and self._d<self.image.shape[1]:
            self._update_along_(0,self.actual_t,self._d)

    def _sliderfunc_h(self,widget, event):
        self._h = int(widget.GetRepresentation().GetValue())
        self._remove_actor(0,1)
        self._remove_actor(1,1)
        if self._h and self._h<self.image.shape[2] :
            self._update_along_(1,self.actual_t,self._h)

    def _sliderfunc_w(self,widget, event):
        self._w = int(widget.GetRepresentation().GetValue())
        self._remove_actor(0,2)
        self._remove_actor(1,2)
        if self._w and self._w<self.image.shape[3] :
            self._update_along_(2,self.actual_t,self._w)

    def _sliderfunc_t(self,widget,event):
        if widget is not None:
            self.actual_t = int(widget.GetRepresentation().GetValue())
        # my_nor = not (self._is_1_temporal or self._is_2_temporal)
        # if self._is_1_temporal or my_nor:
        if self._d and self._d<self.image.shape[1]:
            self._remove_actor(0,0)
            self._remove_actor(1,0)
            self._update_along_(0,self.actual_t,self._d)
        if self._h and self._h<self.image.shape[2] :
            self._remove_actor(0,1)
            self._remove_actor(1,1)
            self._update_along_(1,self.actual_t,self._h)
        if self._w and self._w<self.image.shape[3] :
            self._remove_actor(0,2)
            self._remove_actor(1,2)
            self._update_along_(2,self.actual_t,self._w)

        # if self.flag_def:
        #     self.plotter.renderers[1].RemoveActor(self.visibles[2])
        #     self.visibles[2] = self._make_deformation_flow_()
        #     self.plotter.renderers[1].AddActor(self.visibles[2])

    def _set_slider_size(self,slider,scale):
        sliderRep = slider.GetRepresentation()
        # sliderRep.SetSliderLength(0.003 * scale)  # make it thicker
        # sliderRep.SetSliderWidth(0.025 * scale)
        # sliderRep.SetEndCapLength(0.001 * scale)
        # sliderRep.SetEndCapWidth(0.025 * scale)
        # sliderRep.SetTubeWidth(0.0075 * scale)

    def _button_residuals_(self):
        self._bu_residuals.switch()
        self.show_res = not self.show_res
        # to_residuals = all(x is None for x in self.visibles[1])
        self._sliderfunc_t(None,None)

    def _button_cmp_target_(self):
        if self.show_res:
            print('Impossible to compare target with residuals')
        else:
            self._bu_cmp_target.switch()
            self.flag_cmp_target = not self.flag_cmp_target
            self.flag_cmp_source = False
            self.flag_cmp_mask = False
            self._sliderfunc_t(None,None)

    def _button_cmp_source_(self):
        if self.show_res:
            print('Impossible to compare source with residuals')
        else:
            self._bu_cmp_source.switch()
            self.flag_cmp_source = not self.flag_cmp_source
            self.flag_cmp_target = False
            self.flag_cmp_mask = False
            self._sliderfunc_t(None,None)

    def _button_cmp_mask_(self):
        if self.show_res:
            print('Impossible to compare target with residuals')
        else:
            self._bu_cmp_mask.switch()
            self.flag_cmp_mask = not self.flag_cmp_mask
            self.flag_cmp_target = False
            self.flag_cmp_source = False
            self._sliderfunc_t(None,None)


    def _prepare_image_(self,image):
         return image[:,0].cpu().numpy()
