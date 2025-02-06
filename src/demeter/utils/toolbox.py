import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import torch
from demeter.constants import ROOT_DIRECTORY
DEFAULT_DATA_PATH = 'put_file_path_here'


def rgb2gray(rgb):
    if rgb.shape[-1] > 1:
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    else: return rgb


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

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress,message = None):
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
    text = "\rProgress: [{0}] {1:6.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    if not message is None:
        text += f' ({message[0]} ,{message[1]:8.4f}).'
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

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = torch.tensor([int(x.split()[2]) for x in open('tmp', 'r').readlines()])

    return 'cuda:'+str(int(torch.argmax(memory_available)))

# comparaison of two images
def imCmp(I1, I2):
    M, N = I1.shape
    return np.concatenate((I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2)


# La méthode de PIL ne fonctionne pas pour les images rgb.
# Cette fonction nécessite d'installer la commande Image magic.
def save_gif_with_plt(image_list,file_name,folder=None, delay = 20,duplicate=True,verbose = False,
                      image_args=None,clean=True):
    """  Convert a list og images to a gif

    :param image_list:
    :param file_name: (str) file name withour '.gif' specified
    :param folder: (str) the folder to save the gif, (default: will be saved in
    'gliomorph/figs/gif_box/`file_name`'
    :param delay: (int) millisecond of image duration
    :param duplicate: (bool) duplicate the first and last frame for longer duration.
    :return:
    """
    path = ROOT_DIRECTORY + '/examples/gifs/'
    if folder is None: folder = file_name
    if image_args is None: image_args = dict()
    #make folder if not existing
    if not os.path.exists(path+folder): os.makedirs(path+folder)
    folder = path + folder

    for i in range(len(image_list)):
        plt.imsave(f'{folder}/{file_name}_{i:03d}.png', image_list[i], **image_args)

    # Check that the images were created
    for i in range(len(image_list)):
        if not os.path.exists(f'{folder}/{file_name}_{i:03d}.png'):
            raise FileNotFoundError(f'Image {folder}/{file_name}_{i:03d}.png not found')

    print(f'convert -delay {delay} -loop 0 {folder}/{file_name}_\d{3}.png {folder}/{file_name}.gif')
    os.system(f'convert -delay {delay} -loop 0 {folder}/{file_name}_*.png {folder}/{file_name}.gif')
    if duplicate: os.system(f'convert -duplicate 1,-1 {folder}/{file_name}.gif {folder}/{file_name}.gif')
    # clean
    if not clean: return folder+'/',file_name+'.gif'
    if verbose: print("\tCleaning saved files.")
    for file in os.listdir(folder):
        if file_name in file and '.png' in file:
            os.remove(folder+'/'+file)
    if verbose: print(f"Your gif was successfully saved at : {folder}/{file_name}.gif ")
    return folder+'/',file_name+'.gif'

def fig_to_image(fig,ax):
    # I know it is not PEP to put an import statement inside function. But
    # this function usage being rare, and not time critical, it do keep the
    # import here.
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    ax.axis('off')
    fig.tight_layout(pad=0)

    ax.margins(0) # To remove the huge white borders

    ## Figure is done, we are now taking the render
    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer

    # image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Utiliser tostring_argb et convertir ARGB en RGB
    image_from_plot = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_from_plot = image_from_plot[:, :, [1, 2, 3]]  # Convertir ARGB en RGB
    return image_from_plot
