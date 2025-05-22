import os, re
from demeter.utils.torchbox import resize_image
import demeter.utils.image_3d_plotter as i3p
import matplotlib.pyplot as plt
import nibabel as nib
import torch

ORDER  = [
    "necrosis",             #1
    "wm",                   #4
    "basal",                #7
    "ventricles",       #6
    "gm",                   #5
    "gde",                      #3
    "thalamus",         #8
    "wmh_edema",       #2
]

def open_nii(path):
    img = nib.load(path)
    # affine = img.affine
    # return img.get_fdata(), affine
    img_ras = nib.as_closest_canonical(img)
    return img_ras.get_fdata()

def find_open_mri(path):
    """
    Find the open mri file in the path
    :param path: str
    :return: str
    """
    def find():
        for file in os.listdir(path):
            if 'FLAIR' in file:
                return file

    path_flair = os.path.join(path,find())
    return nib.load(path_flair).get_fdata()

def open_probabilities(path, key='prob'):
    """
    Open the probabilities files
    :param path: str
    :return: np.array
    """
    dict_list = []
    for file in os.listdir(path):
        if key not in file:
            continue
        d = {
            'name': file,
            # 'data': nib.load(os.path.join(path,file)).get_fdata()
            'data': open_nii(os.path.join(path,file)),
        }
        dict_list.append(d)
    return dict_list


def probability_to_simplex(probabilities, key='prob', resize_factors = None):
    d = len(probabilities)

    # simplex = np.zeros((d,) + proba[0]['data'].shape)
    proba_f = []
    pattern = re.compile(fr"{key}_([a-z_]*).nii.gz")
    for o in ORDER:
        for p in probabilities:
            file_id = pattern.search(p["name"])
            if file_id:
                result = file_id.group(1)
            else:
                result = 'non'
                continue

            if o == result:
                print(f"\tAdding : {p["name"]}")
                proba_f.append(p)
                continue
    print("\t len(proba_f)",len(proba_f))
    # for pf in proba_f:
    #     print(pf['name'])
    if resize_factors is None:
        simplex = torch.stack([
            torch.tensor(proba['data'].copy()).clip( 0,1)
            for proba in proba_f
        ],dim=0)[None]
    else:
        img_list = []
        for proba in proba_f:
                img  = torch.tensor(proba['data'].copy())[None, None]
                # ic(img.shape)
                re_img = resize_image(img, resize_factors)
                re_img = re_img.clip(0, 1)
                # ic(re_img.shape)
                img_list.append( re_img )
        simplex = torch.cat(
            img_list,dim=1
        )

    sum_simplex = simplex.sum(dim=1,keepdim=True).clip(0,1)
    # complete the simplex with a background class
    simplex = torch.cat([simplex, 1 - sum_simplex],dim=1)
    return simplex

def path_to_simplex(path, key='prob', resize_factors = None):
    probabilities = open_probabilities(path, key)

    simplex = probability_to_simplex(probabilities, key, resize_factors)
    return simplex

def plot_simplex(image, slc):
    image_rgb =  i3p.SimplexToHSV(image,is_last_background=True).to_rgb()
    # source_rgb = i3p.SimplexToHSV(target,is_last_background=True).to_rgb()

    fig, ax = plt.subplots(1,3, constrained_layout=True, figsize=(15,5))
    ax[0].imshow(image_rgb[0,:,:,slc[2]].transpose(1,0,2), aspect="auto", origin="lower")
    ax[1].imshow(image_rgb[0,:, slc[1], :].transpose(1,0,2), aspect="auto", origin="lower")
    ax[2].imshow(image_rgb[0,slc[0],:,:].transpose(1,0,2), aspect="auto", origin="lower")

    line_color = "red"
    _l_x_v = ax[0].axvline(x=slc[0], color=line_color, alpha=0.6)
    _l_x_h = ax[0].axhline(y=slc[1], color=line_color, alpha=0.6)
    _l_y_v = ax[1].axvline(x=slc[0], color=line_color, alpha=0.6)
    _l_y_h = ax[1].axhline(y=slc[2], color=line_color, alpha=0.6)
    _l_z_v = ax[2].axvline(x=slc[1], color=line_color, alpha=0.6)
    _l_z_h = ax[2].axhline(y=slc[2], color=line_color, alpha=0.6)

    # title = name+ f" t = {t}"
    # fig.suptitle(title, fontsize=20)
    # fig.savefig("imgs/pixyl_reg/"+title+".png")
    plt.show()




