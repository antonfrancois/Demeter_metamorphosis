import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, sobel
from scipy.interpolate import griddata
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import matplotlib
import sys
import torch

DEFAULT_DATA_PATH = '/home/turtlefox/Documents/Doctorat/gliomorph/im2Dbank/'


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def my_imread(file, path=DEFAULT_DATA_PATH):
    file_path = path + file
    return rgb2gray(imread(file_path))


# comparaison of two images
def imCmp(I1, I2):
    M, N = I1.shape
    return np.concatenate((I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2)


def full_ellipse(x, y, a, b, center, theta=.0):
    """
    Return a boolean matrix of the size given by x,y

    :param x, y: grid from meshgrid
    :param a:  constant, control of the wideness of the ellipse
    :param b: constant, control of the wideness of the ellipse
    :param center: has to have a length of two, center coordinates of the ellispe
    :param theta: inclinaison de l'ellispe
    :return:
    """
    if theta == 0:
        return (x - center[0]) ** 2 / a ** 2 + (y - center[1]) ** 2 / b ** 2 < 1
    else:
        tmp = ((x - center[0]) * np.cos(theta) + (y - center[1]) * np.sin(theta)) ** 2 / a ** 2
        tmp += ((x - center[0]) * np.sin(theta) + (y - center[1]) * np.cos(theta)) ** 2 / b ** 2
        return tmp < 1


def imdef(I, Xdef, Ydef):
    # n, m = I.shape
    P = np.stack((Ydef.flatten(), Xdef.flatten()))

    Idef = map_coordinates(I, P, order=1)
    Idef = Idef.reshape(I.shape)
    return Idef


""" old version
def vect2Dinterp_mc(Vx,Vy,Xdef,Ydef):
    print(str(Ydef.shape) +' =?= '+ str(Vy.shape))
    print(str(Xdef.shape) +' =?= '+ str(Vx.shape))
    P = np.array([(Ydef).flatten(),(Xdef).flatten()])
    # P = np.array([(Xdef+Vx).flatten(),(Ydef+Vy).flatten()])
    Vxdef = map_coordinates(Vx,P,order=1)
    Vxdef = Vxdef.reshape(Vx.shape)
    Vydef = map_coordinates(Vy,P,order=1)
    Vydef = Vydef.reshape(Vy.shape)
    return Vxdef, Vydef
"""


def vect2Dinterp(vect_x, vect_y, origin_grid, new_grid=None):
    N, M = vect_x.shape
    origin_grid_x, origin_grid_y = origin_grid

    if new_grid is None:
        new_grid = (origin_grid_x + vect_x,
                    origin_grid_y + vect_y)

    points = np.stack((origin_grid_x.flatten(), origin_grid_y.flatten())).T

    interp_vect_x = griddata(points, vect_x.flatten(), new_grid, fill_value=0)
    interp_vect_y = griddata(points, vect_y.flatten(), new_grid, fill_value=0)

    return interp_vect_x, interp_vect_y


def vect2Dinterp_mc(vect_x, vect_y, origin_grid, new_grid=None):
    N, M = vect_x.shape
    origin_grid_x, origin_grid_y = origin_grid

    if new_grid is None:
        new_grid = (origin_grid_x + vect_x,
                    origin_grid_y + vect_y)

    new_grid = np.stack((new_grid[1].flatten(), new_grid[0].flatten()))

    interp_vect_x = map_coordinates(vect_x, new_grid, order=1)
    interp_vect_y = map_coordinates(vect_y, new_grid, order=1)

    return interp_vect_x.reshape((N, M)), interp_vect_y.reshape((N, M))


def summarizeFlow(vectField, init_grid=None):
    print("WARNING : summarizeFlow deprecated, Use computeFlow instead")
    return computeFlow(vectField, init_grid)


def computeFlow(vectField, init_grid=None, mask=None):
    """

    # :param vectField: TxMxNxD numpy ar
    # where T is the number time max; M and N the dimensions of the vector field
    # and D is the dimensiondef_xx)
    # :return: A MxN vector field that is the composition of all fields in vectField.
    """
    T, D, N, M = vectField.shape  # at this stage of the dev, D = 2

    if init_grid is None:
        # regular grid
        init_grid = np.meshgrid(np.arange(M).astype('float'),
                                np.arange(N).astype('float'))

        def_xx, def_yy = init_grid[0], init_grid[1]

        if mask is None:
            def_xx += vectField[0, 0, :, :]
            def_yy += vectField[0, 1, :, :]
        else:
            def_xx += mask * vectField[0, 0, :, :]
            def_yy += mask * vectField[0, 1, :, :]

        cst = 1
    else:  # interp_vect_x = griddata(points, vect_x.flatten(), new_grid)
        # interp_vect_y = griddata(points, vect_y.flatten(), new_grid)
        def_xx, def_yy = init_grid[0].astype('float'), init_grid[1].astype('float')
        cst = 0

    for t in range(cst, T):
        interp_Field_x, interp_Field_y = vect2Dinterp_mc(
            vectField[t, 0, :, :],
            vectField[t, 1, :, :],
            init_grid,  # origin_grid
            (def_xx, def_yy)  # deformed grid
        )

        if mask is None:
            def_xx += interp_Field_x
            def_yy += interp_Field_y
        else:
            def_xx += mask * interp_Field_x
            def_yy += mask * interp_Field_y

    return np.stack((def_xx, def_yy))


def addGrid2im(img, n_line):
    N, M = img.shape
    I = np.zeros((N, M))
    xx, yy = np.meshgrid(range(M), range(N))

    # cerveau
    a, b = np.array([0.3, 0.35])
    center = 0.5 * np.ones((2))  # +(2*np.random.random((2))-1)*rdm_sigma

    bool1 = full_ellipse(xx / (M - 1), yy / (N - 1), a, b, center)
    # séparation centrale
    bool1[:, int(center[0] * M - M / 200):int(center[0] * M + M / 200)] = False
    cst = 0.1
    for n in range(n_line[0] + 1):
        w = int(np.maximum(N / 200, 1))
        range_m = int(np.maximum(n * N / n_line[0] - w, 0))
        range_p = int(np.minimum(n * N / n_line[0] + w, N - 1))
        img[range_m:range_p, :] = img[range_m:range_p, :] + cst

    for m in range(n_line[1] + 1):
        w = int(np.maximum(M / 200, 1))
        range_m = int(np.maximum(m * M / n_line[1] - w, 0))
        range_p = int(np.minimum(m * M / n_line[1] + w, M - 1))
        img[:, range_m:range_p] = img[:, range_m:range_p] + cst

    return img


def showDeformation(vectField):
    D, N, M = vectField.shape

    # xx,yy = np.meshgrid(np.arange(M).astype('float'),np.arange(N).astype('float'))

    grid = addGrid2im(np.zeros((N, M)), [int(N / 10), int(M / 10)])
    grid_def = imdef(grid, vectField[0, ::], vectField[1, ::])

    fig = plt.figure()
    # ax1 = fig.add_subplot(131)
    # ax1.imshow(grid,cmap='gray')

    ax2 = fig.add_subplot(121)
    ax2.imshow(imCmp(grid / np.max(grid), grid_def / np.max(grid)))

    ax3 = fig.add_subplot(122)
    ax3.imshow(grid_def, cmap='gray')

    fig.show()


def smoothErod(bool_mat, sigma=2):
    """

    :param bool_mat: boolean matrix
    :param sigma: sigma parameterer for the gaussian convolution
    :return:
    """
    dRx = gaussian_filter(sobel(bool_mat.astype('float'), axis=0, mode='constant'), sigma)
    dRy = gaussian_filter(sobel(bool_mat.astype('float'), axis=1, mode='constant'), sigma)

    norm_I = np.maximum(np.abs(dRx), np.abs(dRy))
    norm_I = norm_I / np.max(norm_I)
    norm_I[np.logical_not(bool_mat)] = 0
    smooth_bool = bool_mat - norm_I.astype('float')

    # FOR DEBUG
    # plt.imshow(imCmp(norm_I,dRy))
    #
    # plt.plot(50*bool_mat[100,:])
    # plt.plot(50*norm_I[100,:])
    # plt.plot(50*smooth_bool[100,:])
    # plt.show()

    return smooth_bool



# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    c = str(seconds - int(seconds))[:5]+"cents"
    return "{:d}:{:02d}:{:02d}s and ".format(int(h), int(m), int(s)) + c

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # get the minimum value to then highlight it
    data_min = data.min()

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] ==data_min:
                kw.update(color='tab:red')
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


    return texts

def TestCuda(verbose = True):
    use_cuda = torch.cuda.is_available()
    if(verbose):
        print("Use cuda : ",use_cuda)
    torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
    torchdtype = torch.float32
    KernelMethod = 'CPU'

    if(use_cuda):
        torch.cuda.set_device(torchdeviceId)
        KernelMethod = 'auto'
    # PyKeOps counterpart
    KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.str().split('.')[1]  # 'float32'

    #print(KeOpsdtype)
    return use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod