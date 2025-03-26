import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import demeter.utils.image_3d_plotter as i3p
import demeter.utils.torchbox as tb
from demeter.constants import *

def test_angle_average():
    alpha = torch.tensor([[0.1, 0.8]])
    result = i3p.angle_average(alpha)
    expected = torch.tensor([[0.45]])
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

    result = i3p.angle_average(torch.linspace(0, 1, 10))
    expected = torch.tensor(0.5)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

    result = i3p.angle_average(torch.rand(1, 3, 4, 5), dim=1)
    assert result.shape == (1, 4, 5), f"Expected shape (1, 4, 5), but got {result.shape}"

    result = i3p.angle_average(torch.tensor([torch.nan, 0.2, 0.6]))
    expected = torch.tensor(0.4)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"




# if __name__ == '__main__':
def test_SimplexToHSV():

    add_background = True
    size = (100, 100)
    n_channels = 9
    centers = torch.tensor([
        [15 * torch.sin(t) + 50, 15 * torch.cos(t) + 50] for t in torch.linspace(0, 2 * np.pi, n_channels + 1)[:-1]
    ]).T
    image_channels = [
        tb.RandomGaussianImage(
            size, 1, 'pixel',
            a=[1], b=[10], c=centers[:, i][None]
        ).image() for i in range(n_channels)
    ]
    image_channels = [
        img[0, 0] / img.max() for img in image_channels
    ]
    image = torch.stack(image_channels, dim=0)[None]
    if add_background:
        complementary = torch.ones(size) * image.max() - image.sum(dim=1)
        image = torch.cat([image, complementary[None]], dim=1)
        image /= image.sum(dim=1, keepdim=True)

    d = image.shape[1]
    fig,ax = plt.subplots(1,d)
    for i in range(d):
        ax[i].imshow(image[0,i].cpu(),cmap='gray')
    # plt.show()

    simplex_to_hsv = i3p.SimplexToHSV(image, add_background)

    satu = simplex_to_hsv.prepare_saturation()
    print(satu.shape)
    valu = simplex_to_hsv.prepare_value()
    print(valu.shape)
    fig, ax = plt.subplots(1,2)
    a = ax[0].imshow(satu, cmap='gray')
    fig.colorbar(a, ax=ax[0])
    ax[0].set_title('Saturation')
    a = ax[1].imshow(valu, cmap='gray')
    fig.colorbar(a, ax=ax[1])
    ax[1].set_title('Value')

    plt.show()



    hsv_image = simplex_to_hsv.to_hsv()
    rgb_image = simplex_to_hsv.to_rgb()
    ic(hsv_image.shape)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_title('Hue')
    a = ax[0, 0].imshow(hsv_image[:, :, 0], cmap='gray')
    fig.colorbar(a, ax=ax[0, 0])
    ax[1, 0].imshow(hsv_image[:, :, 1], cmap='gray')
    ax[1, 0].set_title('Saturation')
    ax[0, 1].imshow(hsv_image[:, :, 2], cmap='gray')
    ax[0, 1].set_title('Value')
    ax[1, 1].imshow(rgb_image)
    ax[1, 1].set_title('RGB')
    plt.show()


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

def open_probabilities(path):
    """
    Open the probabilities files
    :param path: str
    :return: np.array
    """
    dict_list = []
    for file in os.listdir(path):
        d = {
            'name': file,
            'data': nib.load(os.path.join(path,file)).get_fdata()
        }
        dict_list.append(d)
    return dict_list


def probability_to_simplex(probabilities):
    d = len(probabilities)

    # simplex = np.zeros((d,) + proba[0]['data'].shape)
    simplex = torch.stack([
        torch.tensor(proba['data'])
        for proba in probabilities
        # if not proba['name'] == 'pred.nii.gz'
        if 'prob' in proba['name']
    ],dim=0)
    sum_simplex = simplex.sum(dim=0,keepdim=True)
    # complete the simplex with a background class
    simplex = torch.cat([simplex, 1 - sum_simplex],dim=0)[None]
    return simplex

def path_to_simplex(path):
    probabilities = open_probabilities(path)
    return probability_to_simplex(probabilities)



if __name__ == '__main__':
#%%
    path_folder = ROOT_DIRECTORY + "/../data/pixyl/predictions_pixyl/PSL_006_M03/"
    # source_path = os.path.join(path_files,source_name)
    print(path_folder)
    # target_path = os.path.join(path_files,target_name)

    image = path_to_simplex(path_folder).to(torch.double)
    print(image.shape)

    simplex_to_hsv = i3p.SimplexToHSV(image, is_last_background = True)
    hsv_image = simplex_to_hsv.to_hsv()
    rgb_image = simplex_to_hsv.to_rgb()

    slice = tb.image_slice(rgb_image, 23, dim= 2)
    ic(slice.shape)
    # #%%
    fig, ax = plt.subplots(1,1)
    ax.imshow(rgb_image[0,:,:,12])
    plt.show()
    #%%
    # i3p.imshow_3d_slider(rgb_image)
    # plt.show()
